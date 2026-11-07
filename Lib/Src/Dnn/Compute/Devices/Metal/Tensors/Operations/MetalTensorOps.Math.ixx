/**
 * @file MetalTensorOps.Math.ixx
 * @brief Elementwise add, subtract, multiply and divide of Metal tensors, and the sum of one.
 */

export module Compute.MetalTensorOps:Math;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalKernelLimits;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal
{
    export struct MathOps
    {
        /// Encode result = a + b. The result may be either input. With a context the caller synchronizes; without one this waits.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result, IExecutionContext* exec_context = nullptr )
        {
            elementwise( "add", a, b, result, exec_context );
        }

        /// Encode result = a - b, as add.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void subtract( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result, IExecutionContext* exec_context = nullptr )
        {
            elementwise( "subtract", a, b, result, exec_context );
        }

        /// Encode result = a * b, as add.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void multiply( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result, IExecutionContext* exec_context = nullptr )
        {
            elementwise( "multiply", a, b, result, exec_context );
        }

        /// Encode result = a / b, as add.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void divide( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result, IExecutionContext* exec_context = nullptr )
        {
            elementwise( "divide", a, b, result, exec_context );
        }

        /// The sum of every element, in 32 bits. Waits for the work encoded before it, as it returns a value.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static float sum( const Tensor<TDataType, TMemoryResource>& tensor, IExecutionContext* exec_context = nullptr )
        {
            if ( tensor.size() == 0 )
            {
                return 0.0f;
            }

            std::optional<MetalExecutionContext> owned;
            MetalExecutionContext* const context = exec_context
                ? validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal sum" )
                : &owned.emplace( tensor.getDeviceId() );

            Tensor<TensorDataType::FP32, MetalMemoryResource> total( tensor.getDeviceId(), shape_t{ 1 } );

            const std::array<const void*, 2> buffers{ static_cast<const ITensor&>( tensor ).rawData(), static_cast<ITensor&>( total ).rawData() };
            const std::uint32_t count = static_cast<std::uint32_t>( narrowToKernelIndex( tensor.size() ) );

            const std::string kernel = std::format( "sum_{}", MetalTensorDataTypeMap<TDataType>::kernel_type );
            const MetalKernelLimits limits = context->device().kernels().limits( kernel );

            context->dispatchThreadgroups( context->device().kernels().pipeline( kernel ), buffers, std::as_bytes( std::span( &count, 1 ) ), 1, 1,
                std::min( kMostSimdgroups, limits.max_total_threads_per_threadgroup / limits.thread_execution_width ) );
            context->synchronize();

            return static_cast<const float*>( static_cast<const ITensor&>( total ).rawData() )[0];
        }

    private:

        // The kernel's threadgroup memory holds one partial per SIMD-group, 32 of them.
        static constexpr std::size_t kMostSimdgroups = 32;

        template<TensorDataType TDataType, typename TMemoryResource>
        static void elementwise( std::string_view operation, const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result, IExecutionContext* exec_context )
        {
            if ( a.shape() != b.shape() || a.shape() != result.shape() )
            {
                throw std::invalid_argument( std::format( "{}: all three tensors must have the same shape", operation ) );
            }

            if ( a.size() == 0 )
            {
                return;
            }

            std::optional<MetalExecutionContext> owned;
            MetalExecutionContext* const context = exec_context
                ? validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal elementwise math" )
                : &owned.emplace( a.getDeviceId() );

            const std::array<const void*, 3> buffers{
                static_cast<const ITensor&>( a ).rawData(), static_cast<const ITensor&>( b ).rawData(), static_cast<ITensor&>( result ).rawData() };

            context->dispatch( context->device().kernels().pipeline( std::format( "{}_{}", operation, MetalTensorDataTypeMap<TDataType>::kernel_type ) ),
                buffers, {}, static_cast<std::size_t>( narrowToKernelIndex( a.size() ) ) );

            if ( owned )
            {
                owned->synchronize();
            }
        }
    };
}
