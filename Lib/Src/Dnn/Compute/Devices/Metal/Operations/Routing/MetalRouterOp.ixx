/**
 * @file MetalRouterOp.ixx
 * @brief Mixture-of-experts selection on Metal: router logits to the top-k experts and their combine weights.
 */

export module Compute.MetalRouterOp;

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
import Dnn.Components.RouterConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Routing
{
    /**
     * @brief Per row: the top_k experts by logit, the lower index on a tie, their softmax renormalized to sum to one and each multiplied
     *        by its per-expert scale, in descending-logit order.
     */
    export template<TensorDataType TPrecision>
    class MetalRouterOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        /// Largest top_k the kernel holds per lane, as CUDA's.
        static constexpr dim_t kMaximumTopK = 16;

        /// @throws std::invalid_argument For a top_k past kMaximumTopK.
        MetalRouterOp( IExecutionContext* context, const RouterConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalRouterOp" ) ),
              config_( config )
        {
            config_.validate();

            if ( config_.getTopK() > kMaximumTopK )
            {
                throw std::invalid_argument( std::format(
                    "MetalRouterOp: top_k {} exceeds the kernel's {}", config_.getTopK(), kMaximumTopK ) );
            }
        }

        /// Bind the per-expert combine scale [num_experts]. @throws std::invalid_argument For a bias: routing has none.
        void setParameters( ITensor* per_expert_scale, ITensor* bias ) override
        {
            if ( bias != nullptr )
            {
                throw std::invalid_argument( "MetalRouterOp: routing has no bias" );
            }

            per_expert_scale_ = per_expert_scale;
        }

        void build( const BuildContext& build_context ) override
        {
            Operation<DeviceType::Metal, TPrecision>::build( build_context );
        }

        /**
         * @brief Route every row of `logits` [..., num_experts] into `weights` [..., top_k] and INT32 `indices` [..., top_k].
         *
         * @throws std::logic_error When the scale was never bound.
         * @throws std::invalid_argument When the tensors do not match the configured experts and top_k.
         */
        void forward( const ITensor& logits, ITensor& weights, ITensor& indices ) const
        {
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( per_expert_scale_ == nullptr )
            {
                throw std::logic_error( "MetalRouterOp::forward: per_expert_scale was never bound" );
            }

            if ( per_expert_scale_->size() != experts )
            {
                throw std::invalid_argument( std::format(
                    "MetalRouterOp::forward: per_expert_scale has {} elements, expected {}", per_expert_scale_->size(), experts ) );
            }

            if ( logits.size() % experts != 0 )
            {
                throw std::invalid_argument( std::format(
                    "MetalRouterOp::forward: {} logits is not a whole number of {}-expert rows", logits.size(), experts ) );
            }

            const dim_t rows = logits.size() / experts;

            if ( weights.size() != rows * top_k || indices.size() != rows * top_k )
            {
                throw std::invalid_argument( std::format(
                    "MetalRouterOp::forward: outputs must hold {} x {} elements, got {} weights and {} indices",
                    rows, top_k, weights.size(), indices.size() ) );
            }

            if ( rows == 0 )
            {
                return;
            }

            const Shape shape{ narrowToKernelIndex( rows ), narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ) };
            const std::array<const void*, 4> buffers{
                logits.rawData(), static_cast<const ITensor*>( per_expert_scale_ )->rawData(), weights.rawData(), indices.rawData() };

            context_->dispatchThreadgroups(
                context_->device().kernels().pipeline( std::format( "router_select_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &shape, 1 ) ), static_cast<std::size_t>( rows ), 1 );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::RouterOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Router";
        }

    private:

        // The kernel's RouterShape.
        struct Shape
        {
            std::int32_t rows;
            std::int32_t experts;
            std::int32_t top_k;
        };

        MetalExecutionContext* context_;
        RouterConfig config_;
        ITensor* per_expert_scale_{ nullptr };
    };
}
