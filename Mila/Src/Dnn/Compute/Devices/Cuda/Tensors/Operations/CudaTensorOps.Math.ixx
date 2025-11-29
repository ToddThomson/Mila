/**
 * @file CudaTensorOps.Math.ixx
 * @brief CUDA tensor mathematical operations partition
 *
 * Implements CUDA-specific tensor mathematical operations using device kernels for
 * efficient parallel computation. Supports element-wise operations, reductions,
 * and activation functions with automatic type handling.
 *
 * Implementation strategy:
 * - Element-wise operations use grid-stride loop kernels
 * - Reduction operations use shared memory and warp-level primitives
 * - Stream-based asynchronous execution for pipeline optimization
 * - Zero-overhead borrowing of ExecutionContext
 * - Automatic fallback to default stream when no context provided
 */

module;
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cmath>
#include <vector>
#include "Kernels/Math.Elementwise.h"

export module Compute.CudaTensorOps:Math;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.DeviceTraits;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.ExecutionContext;
//import Compute.CudaExecutionContext;
import Compute.DeviceType;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CUDA specialization of TensorOps for mathematical operations
     *
     * Provides CUDA-specific implementations of tensor mathematical operations using
     * optimized device kernels for parallel execution on NVIDIA GPUs. Supports all
     * CUDA-compatible tensor data types with automatic type handling.
     *
     * Key features:
     * - Element-wise binary operations (add, subtract, multiply, divide)
     * - Element-wise unary operations (negate, abs, sqrt)
     * - Scalar operations (add scalar, multiply scalar)
     * - Activation functions (ReLU, Sigmoid, Tanh)
     * - Reduction operations (sum, mean, max, min)
     * - Stream-based asynchronous execution
     * - Zero-overhead ExecutionContext borrowing (raw pointer)
     * - Automatic fallback to default stream
     */
    export struct MathOps
    {
        // ================================================================
        // Element-wise Binary Operations
        // ================================================================

        /**
         * @brief Element-wise addition of two tensors
         *
         * Computes result[i] = a[i] + b[i] for all elements using CUDA kernels.
         * Tensors must have identical shapes. Borrows execution context for stream
         * control with zero overhead.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Output tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         *
         * @throws std::invalid_argument If tensor shapes don't match
         * @throws std::runtime_error If CUDA operations fail
         *
         * @note exec_context must outlive this function call
         * @note When exec_context provided, caller controls synchronization
         * @note When exec_context is null, uses default stream and synchronizes before returning
         *
         * Example:
         * @code
         * auto ctx = std::make_unique<CudaExecutionContext>(0);
         * add(tensor_a, tensor_b, result, ctx.get());
         * ctx->synchronize();
         * @endcode
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void add(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            // Validate tensor compatibility
            if (a.shape() != b.shape() || a.shape() != result.shape())
            {
                throw std::invalid_argument(
                    "All tensors must have the same shape for element-wise addition"
                );
            }

            if (a.size() == 0)
            {
                return; // Nothing to compute
            }

            // Determine stream and synchronization requirements
            cudaStream_t stream;
            bool needs_sync = false;
            int device_id = -1;

            if (exec_context)
            {
                // Caller-provided context - borrow stream, let caller control sync
                stream = exec_context->getStream();
                device_id = exec_context->getDeviceId();
            }
            else
            {
                // No context - use default stream with explicit device setting
                auto device = std::dynamic_pointer_cast<CudaDevice>(a.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for math operations"
                    );
                }

                device_id = device->getDeviceId();
                Cuda::setCurrentDevice( device_id );
                stream = nullptr;  // Default stream
                needs_sync = true;  // Must sync default stream before returning
            }

            // Get data pointers through ITensor interface
            const void* a_data = static_cast<const ITensor&>(a).rawData();
            const void* b_data = static_cast<const ITensor&>(b).rawData();
            void* result_data = static_cast<ITensor&>(result).rawData();

            if (!a_data || !b_data || !result_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for add operation" );
            }

            // Perform operation
            addImpl<TDataType>( a_data, b_data, result_data, a.size(), stream, device_id );

            if (needs_sync)
            {
                cudaStreamSynchronize( stream );
            }
        }

        /**
         * @brief Element-wise subtraction of two tensors
         *
         * Computes result[i] = a[i] - b[i] for all elements using CUDA kernels.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First input tensor (minuend)
         * @param b Second input tensor (subtrahend)
         * @param result Output tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void subtract(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (a.shape() != b.shape() || a.shape() != result.shape())
            {
                throw std::invalid_argument(
                    "All tensors must have the same shape for element-wise subtraction"
                );
            }

            if (a.size() == 0)
            {
                return;
            }

            cudaStream_t stream;
            bool needs_sync = false;
            int device_id = -1;

            if (exec_context)
            {
                stream = exec_context->getStream();
                device_id = exec_context->getDeviceId();
            }
            else
            {
                auto device = std::dynamic_pointer_cast<CudaDevice>(a.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for math operations"
                    );
                }

                device_id = device->getDeviceId();
                Cuda::setCurrentDevice( device_id );
                stream = nullptr;
                needs_sync = true;
            }

            // Get data pointers through ITensor interface
            const void* a_data = static_cast<const ITensor&>(a).rawData();
            const void* b_data = static_cast<const ITensor&>(b).rawData();
            void* result_data = static_cast<ITensor&>(result).rawData();

            if (!a_data || !b_data || !result_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for subtract operation" );
            }

            subtractImpl<TDataType>( a_data, b_data, result_data, a.size(), stream, device_id );

            if (needs_sync)
            {
                cudaStreamSynchronize( stream );
            }
        }

        /**
         * @brief Element-wise multiplication of two tensors
         *
         * Computes result[i] = a[i] * b[i] for all elements using CUDA kernels.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Output tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void multiply(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (a.shape() != b.shape() || a.shape() != result.shape())
            {
                throw std::invalid_argument(
                    "All tensors must have the same shape for element-wise multiplication"
                );
            }

            if (a.size() == 0)
            {
                return;
            }

            cudaStream_t stream;
            bool needs_sync = false;
            int device_id = -1;

            if (exec_context)
            {
                stream = exec_context->getStream();
                device_id = exec_context->getDeviceId();
            }
            else
            {
                auto device = std::dynamic_pointer_cast<CudaDevice>(a.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for math operations"
                    );
                }

                device_id = device->getDeviceId();
                Cuda::setCurrentDevice( device_id );
                stream = nullptr;
                needs_sync = true;
            }

            // Get data pointers through ITensor interface
            const void* a_data = static_cast<const ITensor&>(a).rawData();
            const void* b_data = static_cast<const ITensor&>(b).rawData();
            void* result_data = static_cast<ITensor&>(result).rawData();

            if (!a_data || !b_data || !result_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for multiply operation" );
            }

            multiplyImpl<TDataType>( a_data, b_data, result_data, a.size(), stream, device_id );

            if (needs_sync)
            {
                cudaStreamSynchronize( stream );
            }
        }

        /**
         * @brief Element-wise division of two tensors
         *
         * Computes result[i] = a[i] / b[i] for all elements using CUDA kernels.
         * Follows IEEE 754 standards for floating-point division by zero.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First input tensor (dividend)
         * @param b Second input tensor (divisor)
         * @param result Output tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         *
         * @throws std::invalid_argument If tensor shapes don't match
         * @throws std::runtime_error If CUDA operations fail
         *
         * @note For floating-point types, division by zero produces infinity or NaN per IEEE 754
         * @note For integer types, division by zero behavior depends on kernel implementation
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void divide(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (a.shape() != b.shape() || a.shape() != result.shape())
            {
                throw std::invalid_argument(
                    "All tensors must have the same shape for element-wise division"
                );
            }

            if (a.size() == 0)
            {
                return;
            }

            cudaStream_t stream;
            bool needs_sync = false;
            int device_id = -1;

            if (exec_context)
            {
                stream = exec_context->getStream();
                device_id = exec_context->getDeviceId();
            }
            else
            {
                auto device = std::dynamic_pointer_cast<CudaDevice>(a.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for math operations"
                    );
                }

                device_id = device->getDeviceId();
                Cuda::setCurrentDevice( device_id );
                stream = nullptr;
                needs_sync = true;
            }

            // Get data pointers through ITensor interface
            const void* a_data = static_cast<const ITensor&>(a).rawData();
            const void* b_data = static_cast<const ITensor&>(b).rawData();
            void* result_data = static_cast<ITensor&>(result).rawData();

            if (!a_data || !b_data || !result_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for divide operation" );
            }

            divideImpl<TDataType>( a_data, b_data, result_data, a.size(), stream, device_id );

            if (needs_sync)
            {
                cudaStreamSynchronize( stream );
            }
        }

        // ================================================================
        // Reduction Operations
        // ================================================================

        /**
         * @brief Computes sum of all tensor elements
         *
         * Reduces tensor to a single scalar value representing the sum of all elements.
         * Uses optimized CUDA reduction with shared memory and warp primitives.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param tensor Input tensor
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         * @return Sum of all elements as float
         *
         * @note Always returns after synchronization (even with exec_context)
         * @note Result is returned as float for consistency across data types
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static float sum(
            const Tensor<TDataType, TMemoryResource>& tensor,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (tensor.size() == 0)
            {
                return 0.0f;
            }

            cudaStream_t stream;
            int device_id = -1;

            if (exec_context)
            {
                stream = exec_context->getStream();
                device_id = exec_context->getDeviceId();
            }
            else
            {
                auto device = std::dynamic_pointer_cast<CudaDevice>(tensor.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for math operations"
                    );
                }

                device_id = device->getDeviceId();
                Cuda::setCurrentDevice( device_id );
                stream = nullptr;
            }

            // Get data pointer through ITensor interface
            const void* tensor_data = static_cast<const ITensor&>(tensor).rawData();
            if (!tensor_data)
            {
                throw std::runtime_error( "Invalid tensor data pointer for sum operation" );
            }

            return sumImpl<TDataType>( tensor_data, tensor.size(), stream, device_id );
        }

    private:
        // ================================================================
        // Implementation Methods
        // ================================================================

        template<TensorDataType TDataType>
        static void addImpl(
            const void* a_data,
            const void* b_data,
            void* result_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!a_data || !b_data || !result_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;
            const auto* typed_a = static_cast<const NativeType*>(a_data);
            const auto* typed_b = static_cast<const NativeType*>(b_data);
            auto* typed_result = static_cast<NativeType*>(result_data);

            Kernels::launch_elementwise_add_kernel<NativeType>(
                typed_a, typed_b, typed_result, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void subtractImpl(
            const void* a_data,
            const void* b_data,
            void* result_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!a_data || !b_data || !result_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;
            const auto* typed_a = static_cast<const NativeType*>(a_data);
            const auto* typed_b = static_cast<const NativeType*>(b_data);
            auto* typed_result = static_cast<NativeType*>(result_data);

            Kernels::launch_elementwise_subtract_kernel<NativeType>(
                typed_a, typed_b, typed_result, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void multiplyImpl(
            const void* a_data,
            const void* b_data,
            void* result_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!a_data || !b_data || !result_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;
            const auto* typed_a = static_cast<const NativeType*>(a_data);
            const auto* typed_b = static_cast<const NativeType*>(b_data);
            auto* typed_result = static_cast<NativeType*>(result_data);

            Kernels::launch_elementwise_multiply_kernel<NativeType>(
                typed_a, typed_b, typed_result, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void divideImpl(
            const void* a_data,
            const void* b_data,
            void* result_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!a_data || !b_data || !result_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;
            const auto* typed_a = static_cast<const NativeType*>(a_data);
            const auto* typed_b = static_cast<const NativeType*>(b_data);
            auto* typed_result = static_cast<NativeType*>(result_data);

            launch_elementwise_divide_kernel<NativeType>(
                typed_a, typed_b, typed_result, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static float sumImpl(
            const void* tensor_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
			// TODO: Implement sum reduction kernel.
            throw std::runtime_error( "CUDA sum reduction kernels are not implemented yet" );

            if (!tensor_data || count == 0)
            {
                return 0.0f;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;
            const auto* typed_src = static_cast<const NativeType*>(tensor_data);

            constexpr int block = 256;
            const int grid = static_cast<int>((count + block - 1) / block);

            // Allocate device memory for partial sums
            float* d_partial_sums = nullptr;
            cudaError_t status = cudaMallocAsync(
                reinterpret_cast<void**>(&d_partial_sums),
                grid * sizeof( float ),
                stream
            );
            cudaCheckStatus( status, std::source_location::current() );

            float result = 0.0f;
            try
            {
                //Kernels::launch_sum_reduction_kernel<NativeType>(
                //    typed_src, d_partial_sums, count, grid, block,
                //    block * sizeof( float ), stream
                //);

                // Copy partial sums to host
                std::vector<float> h_partial_sums( grid );
                status = cudaMemcpyAsync(
                    h_partial_sums.data(), d_partial_sums,
                    grid * sizeof( float ),
                    cudaMemcpyDeviceToHost,
                    stream
                );
                cudaCheckStatus( status, std::source_location::current() );

                // Synchronize to ensure copy completes
                cudaStreamSynchronize( stream );

                // Final reduction on host
                for (float partial : h_partial_sums)
                {
                    result += partial;
                }

                // Free device memory
                cudaFreeAsync( d_partial_sums, stream );
            }
            catch (...)
            {
                cudaFreeAsync( d_partial_sums, stream );
                throw;
            }

            return result;
        }
    };
}