/**
 * @file MetalGatedDeltaRuleOp.ixx
 * @brief The gated delta rule on Metal, Qwen's DeltaNet recurrence, its state FP32 in either precision.
 */

export module Compute.MetalGatedDeltaRuleOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.GatedDeltaRuleConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::DeltaNet
{
    struct GatedDeltaRuleShape
    {
        std::uint32_t batch;
        std::uint32_t steps;
        std::uint32_t key_heads;
        std::uint32_t value_heads;
        std::uint32_t head_key_dim;
        std::uint32_t head_value_dim;
    };

    /**
     * @brief Advance the gated delta rule over a chunk, the state updated in place.
     *
     * One forward serves prefill and decode, in the recurrent form, which CUDA holds its chunked form to. The state is FP32
     * whatever the precision, as the reference keeps it (`mamba_ssm_dtype`).
     */
    export template<TensorDataType TPrecision>
    class MetalGatedDeltaRuleOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        using ConfigType = GatedDeltaRuleConfig;

        /// @throws std::invalid_argument When a head's key dimension is past 256, the most the kernel holds.
        MetalGatedDeltaRuleOp( IExecutionContext* context, const GatedDeltaRuleConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalGatedDeltaRuleOp" ) ),
              config_( config )
        {
            config_.validate();

            if ( config_.getHeadKeyDim() > kMostHeadKeyDim )
            {
                throw std::invalid_argument( std::format(
                    "MetalGatedDeltaRuleOp: a head's key dimension of {} is past the {} Metal's kernel holds", config_.getHeadKeyDim(), kMostHeadKeyDim ) );
            }

            const dim_t slots_needed = ( config_.getHeadKeyDim() + kKeySplit - 1 ) / kKeySplit;
            const int slots = slots_needed <= 8 ? 8 : slots_needed <= 32 ? 32 : 64;

            pipeline_ = context_->device().kernels().pipeline(
                std::format( "gated_delta_rule_{}_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type, slots ) );
        }

        /// A_log and dt_bias, both [num_value_heads]. The rule derives g and beta from them.
        void setParameters( ITensor* a_log, ITensor* dt_bias ) override
        {
            a_log_ = a_log;
            dt_bias_ = dt_bias;
        }

        /**
         * @param q,k   [B, T, num_key_heads * head_key_dim], not expanded to the value heads.
         * @param v     [B, T, num_value_heads * head_value_dim].
         * @param a,b   [B, T, num_value_heads], the raw projections.
         * @param state [B, num_value_heads, head_key_dim, head_value_dim], FP32.
         * @param out   [B, T, num_value_heads * head_value_dim].
         *
         * @throws std::runtime_error When the parameters are not set or a shape does not match the configuration.
         */
        void forward( const ITensor& q, const ITensor& k, const ITensor& v, const ITensor& a, const ITensor& b, ITensor& state, ITensor& out ) const
        {
            if ( !a_log_ || !dt_bias_ )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - parameters not set" );
            }

            const auto& q_shape = q.shape();

            if ( q_shape.size() != 3 )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - q must be rank 3 [B, T, key_width]" );
            }

            const dim_t batch = q_shape[0];
            const dim_t steps = q_shape[1];

            if ( q_shape[2] != config_.getKeyWidth() || k.shape() != q_shape )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - q/k must both be [B, T, key_width]" );
            }

            if ( v.shape()[2] != config_.getValueWidth() || out.shape() != v.shape() )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - v/out must be [B, T, value_width]" );
            }

            if ( a.shape()[2] != config_.getNumValueHeads() || b.shape() != a.shape() )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - a/b must be [B, T, num_value_heads]" );
            }

            if ( state.getDataType() != TensorDataType::FP32 || state.size() != batch * config_.getStateElementsPerBatch() )
            {
                throw std::runtime_error( "MetalGatedDeltaRuleOp::forward - state does not match the built geometry" );
            }

            if ( steps == 0 )
            {
                return;
            }

            const GatedDeltaRuleShape shape{
                static_cast<std::uint32_t>( narrowToKernelIndex( batch ) ), static_cast<std::uint32_t>( narrowToKernelIndex( steps ) ),
                static_cast<std::uint32_t>( config_.getNumKeyHeads() ), static_cast<std::uint32_t>( config_.getNumValueHeads() ),
                static_cast<std::uint32_t>( config_.getHeadKeyDim() ), static_cast<std::uint32_t>( config_.getHeadValueDim() ) };

            const std::array<const void*, 9> buffers{
                q.rawData(), k.rawData(), v.rawData(), a.rawData(), b.rawData(), a_log_->rawData(), dt_bias_->rawData(), state.rawData(), out.rawData() };

            const std::size_t column_groups = static_cast<std::size_t>( ( config_.getHeadValueDim() + kColumnsPerGroup - 1 ) / kColumnsPerGroup );

            context_->dispatchThreadgroups( pipeline_, buffers, std::as_bytes( std::span( &shape, 1 ) ),
                static_cast<std::size_t>( shape.batch ) * shape.value_heads, column_groups,
                static_cast<std::size_t>( kColumnsPerGroup * kKeySplit ) / kSimdgroupWidth );
        }

        void build( const BuildContext& /*context*/ ) override
        {
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::GatedDeltaRuleOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.GatedDeltaRule";
        }

    private:

        // As GatedDeltaRule.metal: four lanes to a column, 32 columns to a threadgroup, a head's keys up to 256.
        static constexpr dim_t kKeySplit = 4;
        static constexpr dim_t kColumnsPerGroup = 32;
        static constexpr dim_t kMostHeadKeyDim = 256;
        static constexpr std::size_t kSimdgroupWidth = 32;

        MetalExecutionContext* context_;
        GatedDeltaRuleConfig config_;
        MetalPipeline pipeline_{ nullptr };
        ITensor* a_log_{ nullptr };
        ITensor* dt_bias_{ nullptr };
    };
}
