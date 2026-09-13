/**
 * @file CudaMoeOp.ixx
 * @brief CUDA mixture-of-experts bank: the two-pass kernel over the stacked expert tensors.
 *
 * Same contract as CpuMoeOp, gated against HuggingFace through MixtureOfExperts (Gemma4MoE.md Phase 6).
 */

module;
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include "Kernels/Moe.cuh"

export module Compute.CudaMoeOp;

import Dnn.Components.MixtureOfExpertsConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Moe
{
    /**
     * @tparam TPrecision Activation and weight precision (FP32 or BF16).
     * @tparam TFunctor   POD gate functor from Mila::Dnn::Activations exposing fwd(x).
     */
    export template<TensorDataType TPrecision, typename TFunctor>
    class CudaMoeOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using OperationBaseType = Operation<DeviceType::Cuda, TPrecision>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ScratchTensorType = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        CudaMoeOp( IExecutionContext* context, const MixtureOfExpertsConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaMoeOp" ) ),
              config_( config )
        {
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

        /**
         * @brief Size the FP32 gated scratch for the widest input this build admits.
         */
        void build( const BuildContext& build_context ) override
        {
            gated_ = std::make_shared<ScratchTensorType>(
                context_->getDeviceId(), shape_t{ gatedElements( build_context ) }, "moe.gated" );

            OperationBaseType::build( build_context );
        }

        std::size_t getStateMemorySize() const override
        {
            return gated_ ? gated_->getStorageSize() : 0;
        }

        std::size_t getRequiredStateMemorySize( const BuildContext& build_context ) const override
        {
            return storageBytes<TensorDataType::FP32>( gatedElements( build_context ) );
        }

        /**
         * @brief Route every token of @p input through its experts into @p output.
         *
         * @param input    [..., H] at the op's precision.
         * @param weights  [..., top_k] combine weights at the op's precision.
         * @param indices  INT32 [..., top_k] expert indices. An out-of-range index yields NaN.
         * @param output   [..., H] at the op's precision, overwritten.
         */
        void forward( const ITensor& input, const ITensor& weights, const ITensor& indices, ITensor& output ) const
        {
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( gate_up_projection_ == nullptr || down_projection_ == nullptr || !gated_ )
            {
                throw std::logic_error( "CudaMoeOp::forward: expert projections were never bound or the op was never built" );
            }

            if ( gate_up_projection_->size() != experts * 2 * intermediate * hidden
                 || down_projection_->size() != experts * hidden * intermediate )
            {
                throw std::invalid_argument( "CudaMoeOp::forward: expert projections do not match the configured geometry" );
            }

            if ( input.size() % hidden != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::forward: {} inputs is not a whole number of {}-wide tokens", input.size(), hidden ) );
            }

            const dim_t tokens = input.size() / hidden;

            if ( output.size() != input.size() || weights.size() != tokens * top_k || indices.size() != tokens * top_k )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::forward: {} tokens need an output of {} and routing of {} elements; got {}, {} weights, {} indices",
                    tokens, input.size(), tokens * top_k, output.size(), weights.size(), indices.size() ) );
            }

            if ( tokens * top_k * intermediate > gated_->size() )
            {
                throw std::runtime_error( "CudaMoeOp::forward: more tokens than build() sized the gated scratch for" );
            }

            const cudaStream_t stream = context_->getStream();
            auto* gated = static_cast<float*>( gated_->rawData() );
            const auto* index_data = static_cast<const int32_t*>( indices.rawData() );

            launch_moe_gated_forward<NativeType, TFunctor>(
                static_cast<const NativeType*>( input.rawData() ),
                static_cast<const NativeType*>( gate_up_projection_->rawData() ),
                index_data, gated,
                narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ),
                functor_, stream );

            launch_moe_combine_forward<NativeType>(
                gated,
                static_cast<const NativeType*>( down_projection_->rawData() ),
                static_cast<const NativeType*>( weights.rawData() ),
                index_data,
                static_cast<NativeType*>( output.rawData() ),
                narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ),
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::MoeOp;
        }

        std::string getName() const override
        {
            return "Cuda::MoeOp";
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        MixtureOfExpertsConfig config_;
        TFunctor functor_{};

        ITensor* gate_up_projection_{ nullptr };
        ITensor* down_projection_{ nullptr };

        std::shared_ptr<ScratchTensorType> gated_{ nullptr };

        dim_t gatedElements( const BuildContext& build_context ) const
        {
            return elementCount( build_context.inputShape() ) / config_.getHiddenSize()
                * config_.getTopK() * config_.getExpertIntermediateSize();
        }
    };
}
