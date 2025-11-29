/**
 * @file CudaTensorOps.Fill.ixx
 * @brief CUDA tensor fill operations partition
 *
 * Implements CUDA-specific tensor fill operations using device kernels for
 * efficient parallel initialization of tensor data. Supports both scalar
 * broadcast fills and element-wise array copies with automatic type conversion
 * and quantization.
 *
 * Implementation strategy:
 * - Scalar fills use optimized constant kernels (no temporary device memory)
 * - Array fills use chunked staging for memory efficiency on large tensors
 * - Stream-based asynchronous execution for pipeline optimization
 * - Automatic host-to-device type conversion via CUDA kernels
 * - Pure compile-time type dispatch eliminates runtime overhead
 */

module;
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>
#include "Kernels/TensorOps.Fill.h"

export module Compute.CudaTensorOps:Fill;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.ExecutionContext;
//import Compute.CudaExecutionContext;
import Compute.DeviceType;
import Compute.CudaTensorDataType;
import Cuda.Helpers;

namespace Mila::Dnn::Compute::Cuda
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CUDA specialization of TensorOps for initialization operations
     *
     * Provides CUDA-specific implementations of tensor fill operations using
     * optimized device kernels for parallel execution on NVIDIA GPUs. Supports
     * all CUDA-compatible tensor data types with automatic type conversion and
     * quantization from host representations.
     *
     * Key features:
     * - Asynchronous kernel execution using CUDA streams
     * - Zero-overhead borrowing of ExecutionContext (raw pointer semantics)
     * - Automatic fallback to default stream when no context provided
     * - Memory-efficient chunked processing for large arrays
     * - Automatic host-to-device type conversion in kernels
     * - Compile-time type dispatch for zero runtime overhead
     * - Support for FP32, FP16, BF16, FP8, and integer types
     */
    export struct FillOps
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<
            TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

        /**
         * @brief Fill tensor with array of host values using CUDA kernels
         *
         * Copies contiguous host values into a CUDA device tensor, performing
         * automatic type conversion and quantization as needed. Borrows execution
         * context for stream control with zero overhead.
         *
         * Implementation:
         * - Integer types: Use int32_t host representation with kernel conversion
         * - Float types: Use float host representation with kernel conversion
         * - Chunked processing limits temporary device memory usage
         * - Asynchronous execution via provided or default CUDA stream
         * - Compile-time type dispatch based on tensor data type
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param tensor Destination CUDA device tensor to fill
         * @param host_values Span of host values in canonical host representation
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         *
         * @note Host values are automatically converted to device native types
         * @note Uses CUDA stream from exec_context if provided, default stream otherwise
         * @note When using default stream, synchronizes before returning
         * @note When exec_context provided, caller controls synchronization
         * @note exec_context must outlive this function call
         *
         * Example:
         * @code
         * // With explicit context (caller manages sync)
         * auto ctx = std::make_unique<CudaExecutionContext>(0);
         * fill(tensor, values, ctx.get());
         * // ... queue more operations on same stream
         * ctx->synchronize();
         * 
         * // Without context (automatic sync)
         * fill(tensor, values);  // Uses default stream, returns after sync
         * @endcode
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            std::span<const host_value_t<TDataType>> host_values,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (tensor.size() == 0 || host_values.empty())
                return;

            cudaStream_t stream;
            bool needs_sync = false;

            if (exec_context) {
                // Caller-provided context - borrow stream, let caller control sync
                stream = exec_context->getStream();
            }
            else
            {
                // No context - use default stream with explicit device setting
                auto device = std::dynamic_pointer_cast<CudaDevice>(tensor.getDevice());
                if (!device)
                {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for fill operation"
                    );
                }

                Cuda::setCurrentDevice( device->getDeviceId() );

                stream = nullptr;  // nullptr = default stream (stream 0)
                needs_sync = true;  // Must sync default stream before returning
            }

            const size_t count = std::min(tensor.size(), host_values.size());
            
            void* raw_dst = static_cast<ITensor&>(tensor).rawData();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                Cuda::launch_array_fill_typed<NativeType, int32_t>(
                    raw_dst, host_values.data(), count, stream);
            }
            else {
                Cuda::launch_array_fill_typed<NativeType, float>(
                    raw_dst, host_values.data(), count, stream);
            }

            if (needs_sync) {
                // Synchronize default stream to ensure completion
                cudaStreamSynchronize(stream);
            }
        }

        /**
         * @brief Fill tensor with scalar host value using CUDA kernels
         *
         * Broadcasts a single host scalar value to all elements of a CUDA device
         * tensor using optimized constant fill kernels. No temporary device memory
         * is required - conversion happens directly in the kernel. Borrows execution
         * context for stream control with zero overhead.
         *
         * Implementation:
         * - Integer types: Use int32_t host representation with kernel conversion
         * - Float types: Use float host representation with kernel conversion
         * - Grid-stride loop kernels for scalability across tensor sizes
         * - Asynchronous execution via provided or default CUDA stream
         * - Compile-time type dispatch based on tensor data type
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param tensor Destination CUDA device tensor to fill
         * @param host_value Scalar value in canonical host representation
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         *
         * @note Host value is automatically converted to device native type
         * @note Uses CUDA stream from exec_context if provided, default stream otherwise
         * @note When using default stream, synchronizes before returning
         * @note When exec_context provided, caller controls synchronization
         * @note Optimized for constant broadcasts - no temporary memory allocation
         * @note exec_context must outlive this function call
         *
         * Example:
         * @code
         * // With explicit context (caller manages sync)
         * auto ctx = std::make_unique<CudaExecutionContext>(0);
         * fill(tensor1, 0.0f, ctx.get());
         * fill(tensor2, 1.0f, ctx.get());
         * ctx->synchronize();
         * 
         * // Without context (automatic sync)
         * fill(tensor, 3.14f);  // Uses default stream, returns after sync
         * @endcode
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            host_value_t<TDataType> host_value,
            ExecutionContext<DeviceType::Cuda>* exec_context = nullptr )
        {
            if (tensor.size() == 0)
                return;

            cudaStream_t stream;
            bool needs_sync = false;

            if (exec_context) {
                // Caller-provided context - borrow stream, let caller control sync
                stream = exec_context->getStream();
            }
            else {
                // No context - use default stream with explicit device setting
                auto device = std::dynamic_pointer_cast<CudaDevice>(tensor.getDevice());
                if (!device) {
                    throw std::runtime_error(
                        "Tensor does not have valid CUDA device for fill operation"
                    );
                }
                
                Cuda::setCurrentDevice(device->getDeviceId());
                stream = nullptr;  // nullptr = default stream (stream 0)
                needs_sync = true;  // Must sync default stream before returning
            }

            // Access raw data through TensorOps helper (which has friend access)
			void* raw_dst = static_cast<ITensor&>(tensor).rawData();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                Cuda::launch_constant_fill_typed<NativeType, int32_t>(
                    raw_dst, tensor.size(), host_value, stream);
            }
            else {
                Cuda::launch_constant_fill_typed<NativeType, float>(
                    raw_dst, tensor.size(), host_value, stream);
            }

            if (needs_sync) {
                // Synchronize default stream to ensure completion
                cudaStreamSynchronize(stream);
            }
        }
    };
}