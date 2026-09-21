/**
 * @file CpuMoeOp.ixx
 * @brief CPU mixture-of-experts bank (FP32): each token through its selected experts, weighted and summed.
 *
 * The reference implementation the CUDA grouped path is gated against, through MixtureOfExperts.
 * See Specifications/Gemma4MoE.md Phase 6.
 */

module;
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

export module Compute.CpuMoeOp;

import Dnn.Components.MixtureOfExpertsConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Per token t and selected slot k, with e = indices[t, k]:
     *
     *   [gate | up] = gate_up_projection[e] * x_t                 ( [2I] )
     *   hidden      = TFunctor( gate ) * up                       ( [I]  )
     *   y_t        += weights[t, k] * ( down_projection[e] * hidden )
     *
     * The gate is the first half of the fused projection, as HuggingFace's `chunk(2)` takes it.
     *
     * @tparam TFunctor POD functor from Mila::Dnn::Activations exposing fwd(x).
     */
    export template<typename TFunctor>
    class CpuMoeOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using OperationBaseType = Operation<DeviceType::Cpu, TensorDataType::FP32>;

        CpuMoeOp( IExecutionContext* context, const MixtureOfExpertsConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuMoeOp requires a CPU execution context" );
            }

            config_.validate();
        }

        /**
         * @brief Bind the stacked expert projections: gate_up [E, 2I, H] and down [E, H, I].
         */
        void setParameters( ITensor* gate_up_projection, ITensor* down_projection ) override
        {
            gate_up_projection_ = gate_up_projection;
            down_projection_ = down_projection;
        }

        void build( const BuildContext& build_context ) override
        {
            OperationBaseType::build( build_context );
        }

        /**
         * @brief Route every token of @p input through its experts into @p output.
         *
         * @param input    FP32 [..., H].
         * @param weights  FP32 [..., top_k] combine weights.
         * @param indices  INT32 [..., top_k] expert indices.
         * @param output   FP32 [..., H], overwritten.
         */
        void forward( const ITensor& input, const ITensor& weights, const ITensor& indices, ITensor& output ) const
        {
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( gate_up_projection_ == nullptr || down_projection_ == nullptr )
            {
                throw std::logic_error( "CpuMoeOp::forward: expert projections were never bound" );
            }

            if ( gate_up_projection_->size() != experts * 2 * intermediate * hidden
                 || down_projection_->size() != experts * hidden * intermediate )
            {
                throw std::invalid_argument( "CpuMoeOp::forward: expert projections do not match the configured geometry" );
            }

            if ( input.size() % hidden != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CpuMoeOp::forward: {} inputs is not a whole number of {}-wide tokens", input.size(), hidden ) );
            }

            const dim_t tokens = input.size() / hidden;

            if ( output.size() != input.size() || weights.size() != tokens * top_k || indices.size() != tokens * top_k )
            {
                throw std::invalid_argument( std::format(
                    "CpuMoeOp::forward: {} tokens need an output of {} and routing of {} elements; got {}, {} weights, {} indices",
                    tokens, input.size(), tokens * top_k, output.size(), weights.size(), indices.size() ) );
            }

            const auto* x = static_cast<const float*>( input.rawData() );
            const auto* combine = static_cast<const float*>( weights.rawData() );
            const auto* selected = static_cast<const std::int32_t*>( indices.rawData() );
            const auto* gate_up = static_cast<const float*>( gate_up_projection_->rawData() );
            const auto* down = static_cast<const float*>( down_projection_->rawData() );
            auto* y = static_cast<float*>( output.rawData() );

            gated_.resize( static_cast<std::size_t>( intermediate ) );

            for ( dim_t token = 0; token < tokens; ++token )
            {
                const float* token_input = x + token * hidden;
                float* token_output = y + token * hidden;

                for ( dim_t j = 0; j < hidden; ++j )
                {
                    token_output[ j ] = 0.0f;
                }

                for ( dim_t slot = 0; slot < top_k; ++slot )
                {
                    const std::int32_t expert = selected[ token * top_k + slot ];

                    if ( expert < 0 || expert >= experts )
                    {
                        throw std::out_of_range( std::format(
                            "CpuMoeOp::forward: token {} selects expert {} of {}", token, expert, experts ) );
                    }

                    const float* expert_gate_up = gate_up + expert * 2 * intermediate * hidden;
                    const float* expert_down = down + expert * hidden * intermediate;

                    for ( dim_t i = 0; i < intermediate; ++i )
                    {
                        const float* gate_row = expert_gate_up + i * hidden;
                        const float* up_row = expert_gate_up + ( intermediate + i ) * hidden;

                        float gate_value = 0.0f;
                        float up_value = 0.0f;

                        for ( dim_t c = 0; c < hidden; ++c )
                        {
                            gate_value += gate_row[ c ] * token_input[ c ];
                            up_value += up_row[ c ] * token_input[ c ];
                        }

                        gated_[ static_cast<std::size_t>( i ) ] = functor_.fwd( gate_value ) * up_value;
                    }

                    const float weight = combine[ token * top_k + slot ];

                    for ( dim_t j = 0; j < hidden; ++j )
                    {
                        const float* down_row = expert_down + j * intermediate;
                        float projected = 0.0f;

                        for ( dim_t i = 0; i < intermediate; ++i )
                        {
                            projected += down_row[ i ] * gated_[ static_cast<std::size_t>( i ) ];
                        }

                        token_output[ j ] += weight * projected;
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::MoeOp;
        }

        std::string getName() const override
        {
            return "Cpu::MoeOp";
        }

    private:
        IExecutionContext* context_;
        MixtureOfExpertsConfig config_;
        TFunctor functor_{};

        ITensor* gate_up_projection_{ nullptr };
        ITensor* down_projection_{ nullptr };

        // One expert's gated activations, reused across calls; the op holds no other state.
        mutable std::vector<float> gated_;
    };
}
