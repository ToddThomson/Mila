/**
 * @file CudaCausalConv1dOp.ixx
 * @brief CUDA implementation of the depthwise causal 1-D convolution.
 *
 * Two entry points rather than one forward: `forward` convolves a chunk against an
 * optional left context, and `updateState` refreshes that context afterwards. They are
 * separate because the caller decides whether the convolution carries memory at all --
 * a standalone use convolves with zero padding and never touches a state buffer.
 */

module;
#include "Kernels/CausalConv1d.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include <cstdint>
#include <string>

export module Compute.CudaCausalConv1dOp;
import :Dispatch;

import Dnn.Components.CausalConv1dConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
// MSVC WORKAROUND, not a design decision: a consumer instantiating this operation must
// complete ExecutionContext<Cuda>, and MSVC 14.51 demands the type be VISIBLE where the
// standard -- and Clang 19+ -- accept it being merely reachable. Restore the plain import
// when that is fixed; nothing else here depends on the export.
export import Compute.ExecutionContext;
import Compute.OperationType;
import Dnn.Component;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Convolution
{
    export template <TensorDataType TPrecision>
    class CudaCausalConv1dOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = CausalConv1dConfig;

        CudaCausalConv1dOp( IExecutionContext* context, const CausalConv1dConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaCausalConv1dOp" ) ),
              config_( config )
        {
            if ( !context_ )
            {
                throw std::invalid_argument(
                    "CudaCausalConv1dOp requires a non-null CUDA execution context" );
            }

            channels_ = static_cast<int>( config_.getChannels() );
            kernel_width_ = static_cast<int>( config_.getKernelWidth() );
        }

        void setParameters( ITensor* weight, ITensor* bias )
        {
            weight_ = weight;
            bias_ = bias;
        }

        /**
         * @brief Convolve a chunk, optionally against a retained left context.
         *
         * @param input  [B, T, C].
         * @param state  [B, K-1, C] holding the previous K-1 input rows, or nullptr for a
         *               sequence that starts here (missing rows are zero).
         * @param output [B, T, C].
         */
        void forward( const ITensor& input, const ITensor* state, ITensor& output ) const
        {
            const NativeType* x = static_cast<const NativeType*>( input.rawData() );
            NativeType* y = static_cast<NativeType*>( output.rawData() );

            if ( !x || !y )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::forward - null tensor data pointer" );
            }

            if ( !weight_ )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::forward - parameters not set" );
            }

            const auto& shape = input.shape();

            if ( shape.size() != 3 )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::forward - input must be rank 3 [B, T, C]" );
            }

            if ( shape[ 2 ] != channels_ )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::forward - channel count mismatch" );
            }

            if ( output.size() != input.size() )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::forward - input/output size mismatch" );
            }

            const int B = static_cast<int>( shape[ 0 ] );
            const int T = static_cast<int>( shape[ 1 ] );

            const NativeType* state_data = state
                ? static_cast<const NativeType*>( state->rawData() )
                : nullptr;

            Detail::cuda_causal_conv1d_impl<NativeType>::forward(
                y, x, state_data,
                static_cast<const NativeType*>( weight_->rawData() ),
                bias_ ? static_cast<const NativeType*>( bias_->rawData() ) : nullptr,
                B, T, channels_, kernel_width_,
                context_->getStream() );
        }

        /**
         * @brief Refresh @p state to the last K-1 rows of [state ; input].
         *
         * Must run after forward() for the same chunk -- it overwrites the rows that pass
         * reads.
         */
        void updateState( const ITensor& input, ITensor& state ) const
        {
            const NativeType* x = static_cast<const NativeType*>( input.rawData() );
            NativeType* s = static_cast<NativeType*>( state.rawData() );

            if ( !x || !s )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::updateState - null tensor data pointer" );
            }

            const auto& shape = input.shape();

            if ( shape.size() != 3 )
            {
                throw std::runtime_error( "CudaCausalConv1dOp::updateState - input must be rank 3" );
            }

            const int B = static_cast<int>( shape[ 0 ] );
            const int T = static_cast<int>( shape[ 1 ] );

            Detail::cuda_causal_conv1d_impl<NativeType>::updateState(
                s, x, B, T, channels_, kernel_width_, context_->getStream() );
        }

        void build( const BuildContext& /*context*/ ) override
        {
        }

        OperationType getOperationType() const override
        {
            return OperationType::CausalConv1dOp;
        }

        std::string getName() const override
        {
            return "Cuda::CausalConv1dOp";
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        CausalConv1dConfig config_;

        ITensor* weight_{ nullptr };
        ITensor* bias_{ nullptr };

        int channels_{ 0 };
        int kernel_width_{ 0 };
    };
}
