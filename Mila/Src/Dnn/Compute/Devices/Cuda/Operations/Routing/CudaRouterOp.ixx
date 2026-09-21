/**
 * @file CudaRouterOp.ixx
 * @brief CUDA mixture-of-experts selection: router logits to top-k experts and combine weights.
 *
 * Same contract as CpuRouterOp, which it is gated against through Router (Gemma4MoE.md Phase 5).
 */

module;
#include "Kernels/Router.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include <format>
#include <cstdint>
#include <string>

export module Compute.CudaRouterOp;
import :Dispatch;

import Dnn.Components.RouterConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Dnn.Component;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Routing
{
    export template <TensorDataType TPrecision>
    class CudaRouterOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using OperationBaseType = Operation<DeviceType::Cuda, TPrecision>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaRouterOp( IExecutionContext* context, const RouterConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRouterOp" ) ),
              config_( config )
        {
            config_.validate();

            if ( config_.getTopK() > kMaximumTopK )
            {
                throw std::invalid_argument( std::format(
                    "CudaRouterOp: top_k {} exceeds the kernel's register budget of {}",
                    config_.getTopK(), kMaximumTopK ) );
            }
        }

        /**
         * @brief Bind the per-expert combine scale [num_experts]. There is no bias.
         */
        void setParameters( ITensor* per_expert_scale, ITensor* bias ) override
        {
            if ( bias != nullptr )
            {
                throw std::invalid_argument( "CudaRouterOp: routing has no bias" );
            }

            per_expert_scale_ = per_expert_scale;
        }

        void build( const BuildContext& build_context ) override
        {
            OperationBaseType::build( build_context );
        }

        /**
         * @brief Route every row of @p logits.
         *
         * @param logits   [..., num_experts] at the op's precision.
         * @param weights  [..., top_k] at the op's precision, written.
         * @param indices  INT32 [..., top_k], written.
         */
        void forward( const ITensor& logits, ITensor& weights, ITensor& indices ) const
        {
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( per_expert_scale_ == nullptr )
            {
                throw std::logic_error( "CudaRouterOp::forward: per_expert_scale was never bound" );
            }

            if ( per_expert_scale_->size() != experts )
            {
                throw std::invalid_argument( std::format(
                    "CudaRouterOp::forward: per_expert_scale has {} elements, expected {}",
                    per_expert_scale_->size(), experts ) );
            }

            if ( logits.size() % experts != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CudaRouterOp::forward: {} logits is not a whole number of {}-expert rows",
                    logits.size(), experts ) );
            }

            const dim_t rows = logits.size() / experts;

            if ( weights.size() != rows * top_k || indices.size() != rows * top_k )
            {
                throw std::invalid_argument( std::format(
                    "CudaRouterOp::forward: outputs must hold {} x {} elements, got {} weights and {} indices",
                    rows, top_k, weights.size(), indices.size() ) );
            }

            if ( rows == 0 )
            {
                return;
            }

            Detail::cuda_router_select_impl<NativeType>::forward(
                static_cast<const NativeType*>( logits.rawData() ),
                static_cast<const NativeType*>( per_expert_scale_->rawData() ),
                static_cast<NativeType*>( weights.rawData() ),
                static_cast<int32_t*>( indices.rawData() ),
                narrowToKernelIndex( rows ), narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ),
                context_->getStream() );
        }

        OperationType getOperationType() const override
        {
            return OperationType::RouterOp;
        }

        std::string getName() const override
        {
            return "Cuda::RouterOp";
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        RouterConfig config_;
        ITensor* per_expert_scale_{ nullptr };
    };
}
