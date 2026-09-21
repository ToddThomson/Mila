/**
 * @file CpuRouterOp.ixx
 * @brief CPU mixture-of-experts selection (FP32): router logits to top-k experts and combine weights.
 *
 * The reference implementation the CUDA op is gated against. See Specifications/Gemma4MoE.md Phase 5.
 */

module;
#include <string>
#include <stdexcept>
#include <format>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

export module Compute.CpuRouterOp;

import Dnn.Components.RouterConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Selects each token's experts and their combine weights from router logits.
     *
     * Per row of `num_experts` logits: softmax over all experts, keep the top_k, renormalize those
     * to sum to one, then multiply each by its expert's `per_expert_scale`. The final weights
     * therefore do not sum to one.
     *
     * Ranking uses the logits, not the probabilities. Softmax is monotone, so the two orders agree
     * wherever the probabilities are distinct; ranking the logits keeps experts apart that rounded
     * probabilities would tie. Equal logits rank the lower expert index first.
     *
     * Output order within a row is by descending logit. Consumers must not depend on it: the
     * combine is a sum over the selected experts.
     */
    export class CpuRouterOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        CpuRouterOp( IExecutionContext* context, const RouterConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuRouterOp requires a CPU execution context" );
            }

            config_.validate();
        }

        /**
         * @brief Bind the per-expert combine scale [num_experts]. There is no bias.
         */
        void setParameters( ITensor* per_expert_scale, ITensor* bias ) override
        {
            if ( bias != nullptr )
            {
                throw std::invalid_argument( "CpuRouterOp: routing has no bias" );
            }

            per_expert_scale_ = per_expert_scale;
        }

        /**
         * @brief Route every row of @p logits.
         *
         * @param logits   FP32 [..., num_experts].
         * @param weights  FP32 [..., top_k], written.
         * @param indices  INT32 [..., top_k], written.
         */
        void forward( const ITensor& logits, ITensor& weights, ITensor& indices ) const
        {
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( per_expert_scale_ == nullptr )
            {
                throw std::logic_error( "CpuRouterOp::forward: per_expert_scale was never bound" );
            }

            if ( per_expert_scale_->size() != experts )
            {
                throw std::invalid_argument( std::format(
                    "CpuRouterOp::forward: per_expert_scale has {} elements, expected {}",
                    per_expert_scale_->size(), experts ) );
            }

            if ( logits.size() % experts != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CpuRouterOp::forward: {} logits is not a whole number of {}-expert rows",
                    logits.size(), experts ) );
            }

            const dim_t rows = logits.size() / experts;

            if ( weights.size() != rows * top_k || indices.size() != rows * top_k )
            {
                throw std::invalid_argument( std::format(
                    "CpuRouterOp::forward: outputs must hold {} x {} elements, got {} weights and {} indices",
                    rows, top_k, weights.size(), indices.size() ) );
            }

            const auto* logit_data = static_cast<const float*>( logits.rawData() );
            const auto* scale_data = static_cast<const float*>( per_expert_scale_->rawData() );
            auto* weight_data = static_cast<float*>( weights.rawData() );
            auto* index_data = static_cast<std::int32_t*>( indices.rawData() );

            std::vector<std::int32_t> order( static_cast<std::size_t>( experts ) );
            std::vector<float> probabilities( static_cast<std::size_t>( experts ) );

            for ( dim_t row = 0; row < rows; ++row )
            {
                const float* row_logits = logit_data + row * experts;

                softmax( row_logits, experts, probabilities.data() );

                std::iota( order.begin(), order.end(), 0 );
                std::partial_sort( order.begin(), order.begin() + top_k, order.end(),
                    [row_logits]( std::int32_t left, std::int32_t right )
                    {
                        if ( row_logits[ left ] != row_logits[ right ] )
                        {
                            return row_logits[ left ] > row_logits[ right ];
                        }

                        return left < right;
                    } );

                float selected_mass = 0.0f;

                for ( dim_t slot = 0; slot < top_k; ++slot )
                {
                    selected_mass += probabilities[ static_cast<std::size_t>( order[ slot ] ) ];
                }

                for ( dim_t slot = 0; slot < top_k; ++slot )
                {
                    const std::int32_t expert = order[ slot ];
                    const float renormalized = probabilities[ static_cast<std::size_t>( expert ) ] / selected_mass;

                    weight_data[ row * top_k + slot ] = renormalized * scale_data[ expert ];
                    index_data[ row * top_k + slot ] = expert;
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::RouterOp;
        }

        std::string getName() const override
        {
            return "Cpu::RouterOp";
        }

    private:
        // Max-shifted and accumulated in double, so a row of large logits neither overflows nor
        // loses the small probabilities the renormalization divides by.
        static void softmax( const float* row, dim_t count, float* out )
        {
            const double max_logit = *std::max_element( row, row + count );
            double total = 0.0;

            for ( dim_t i = 0; i < count; ++i )
            {
                total += std::exp( static_cast<double>( row[ i ] ) - max_logit );
            }

            for ( dim_t i = 0; i < count; ++i )
            {
                out[ i ] = static_cast<float>( std::exp( static_cast<double>( row[ i ] ) - max_logit ) / total );
            }
        }

        IExecutionContext* context_;
        RouterConfig config_;
        ITensor* per_expert_scale_{ nullptr };
    };
}
