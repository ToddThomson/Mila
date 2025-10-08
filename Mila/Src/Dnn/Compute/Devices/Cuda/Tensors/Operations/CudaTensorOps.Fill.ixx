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

export module Dnn.TensorOps:Fill.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CudaDeviceMemoryResource;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.DeviceType;
import Compute.CudaTensorDataType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Cuda;

    export template<typename TComputeDeviceTag> struct TensorOps;

    /**
     * @brief CUDA specialization of TensorOps for initialization operations
     *
     * Provides CUDA-specific implementations of tensor fill operations using
     * optimized device kernels for parallel execution on NVIDIA GPUs. Supports
     * all CUDA-compatible tensor data types with automatic type conversion and
     * quantization from host representations.
     *
     * Key features:
     * - Asynchronous kernel execution using CUDA streams via ExecutionContext
     * - Memory-efficient chunked processing for large arrays
     * - Automatic host-to-device type conversion in kernels
     * - Compile-time type dispatch for zero runtime overhead
     * - Support for FP32, FP16, BF16, FP8, and integer types
     */
    export template<>
        struct TensorOps<Compute::CudaComputeDeviceTag>
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<
            TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

        /**
         * @brief Fill tensor with array of host values using CUDA kernels
         *
         * Copies contiguous host values into a CUDA device tensor, performing
         * automatic type conversion and quantization as needed. Uses optional
         * ExecutionContext for explicit stream control.
         *
         * Implementation:
         * - Integer types: Use int32_t host representation with kernel conversion
         * - Float types: Use float host representation with kernel conversion
         * - Chunked processing limits temporary device memory usage
         * - Asynchronous execution via execution context stream
         * - Compile-time type dispatch based on tensor data type
         *
         * @tparam TDataType Abstract tensor data type
         * @param tensor Destination CUDA device tensor to fill
         * @param host_values Span of host values in canonical host representation
         * @param exec_context Optional execution context for stream control
         *
         * @note Host values are automatically converted to device native types
         * @note Uses CUDA stream from execution context for async execution
         * @note For large arrays, uses chunked processing to limit memory usage
         * @note If exec_context is null, creates temporary context
         */
        template<TensorDataType TDataType>
        static void fill(
            Tensor<TDataType, CudaDeviceMemoryResource>& tensor,
            std::span<const host_value_t<TDataType>> host_values,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context = nullptr )
        {
            if (tensor.size() == 0 || host_values.empty())
                return;

            // Get or create execution context
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> cuda_exec_ctx;

            if (exec_context) {
                cuda_exec_ctx = exec_context;
            }
            else {
                // Create temporary execution context for this operation
                int device_id = tensor.getDeviceId();
                if (device_id < 0) {
                    throw std::runtime_error(
                        "Tensor does not have valid device ID for fill operation"
                    );
                }
                cuda_exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( device_id );
            }

            cudaStream_t stream = cuda_exec_ctx->getStream();
            const size_t count = std::min( tensor.size(), host_values.size() );
            void* raw_dst = tensor.rawData();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                Cuda::launch_array_fill_typed<NativeType, int32_t>(
                    raw_dst, host_values.data(), count, stream );
            }
            else {
                Cuda::launch_array_fill_typed<NativeType, float>(
                    raw_dst, host_values.data(), count, stream );
            }

            // Synchronize to ensure fill completes before returning
            cuda_exec_ctx->synchronize();
        }


        /**
         * @brief Fill tensor with scalar host value using CUDA kernels
         *
         * Broadcasts a single host scalar value to all elements of a CUDA device
         * tensor using optimized constant fill kernels. No temporary device memory
         * is required - conversion happens directly in the kernel. Uses optional
         * ExecutionContext for explicit stream control.
         *
         * Implementation:
         * - Integer types: Use int32_t host representation with kernel conversion
         * - Float types: Use float host representation with kernel conversion
         * - Grid-stride loop kernels for scalability across tensor sizes
         * - Asynchronous execution via execution context stream
         * - Compile-time type dispatch based on tensor data type
         *
         * @tparam TDataType Abstract tensor data type
         * @param tensor Destination CUDA device tensor to fill
         * @param host_value Scalar value in canonical host representation
         * @param exec_context Optional execution context for stream control
         *
         * @note Host value is automatically converted to device native type
         * @note Uses CUDA stream from execution context for async execution
         * @note Optimized for constant broadcasts - no temporary memory allocation
         * @note If exec_context is null, creates temporary context
         */
        template<TensorDataType TDataType>
        static void fill(
            Tensor<TDataType, CudaDeviceMemoryResource>& tensor,
            host_value_t<TDataType> host_value,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context = nullptr )
        {
            if (tensor.size() == 0)
                return;

            // Get or create execution context
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> cuda_exec_ctx;

            if (exec_context) {
                cuda_exec_ctx = exec_context;
            }
            else {
                // Create temporary execution context for this operation
                int device_id = tensor.getDeviceId();
                if (device_id < 0) {
                    throw std::runtime_error(
                        "Tensor does not have valid device ID for fill operation"
                    );
                }
                cuda_exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( device_id );
            }

            cudaStream_t stream = cuda_exec_ctx->getStream();
            void* raw_dst = tensor.rawData();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                Cuda::launch_constant_fill_typed<NativeType, int32_t>(
                    raw_dst, tensor.size(), host_value, stream );
            }
            else {
                Cuda::launch_constant_fill_typed<NativeType, float>(
                    raw_dst, tensor.size(), host_value, stream );
            }

            // Synchronize to ensure fill completes before returning
            cuda_exec_ctx->synchronize();
        }
    };
}