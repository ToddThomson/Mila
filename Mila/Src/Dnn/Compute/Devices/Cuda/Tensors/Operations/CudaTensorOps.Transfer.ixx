/**
 * @file CudaTensorOps.Transfer.ixx
 * @brief CUDA tensor transfer operations partition
 *
 * Provides tensor transfer operations using ExecutionContext for stream management.
 * TensorOps work with tensor data directly (via data()) and accept optional
 * ExecutionContext for explicit stream control with zero-overhead borrowing semantics.
 *
 * Implementation strategy:
 * - Raw pointer semantics for ExecutionContext (non-owning borrow)
 * - Automatic fallback to default CUDA stream when no context provided
 * - Stream-based asynchronous execution for pipeline optimization
 * - Automatic type conversion using CUDA kernels
 * - Memory-efficient staging for host-device transfers with conversion
 */

module;
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cstring>
#include "Kernels/Transfer.Copy.h"

export module Compute.CudaTensorOps:Transfer;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.CudaTensorDataType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDevice;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.DeviceType;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CUDA specialization of TensorOps for tensor transfer operations
     *
     * Provides CUDA-specific implementations of tensor transfer operations with
     * automatic optimization based on memory types and optional type conversion.
     * Uses zero-overhead borrowing of ExecutionContext for stream control.
     *
     * Key features:
     * - Automatic transfer direction detection (H2D, D2H, D2D, H2H)
     * - Optional type conversion during transfer using CUDA kernels
     * - Stream-based asynchronous execution
     * - Zero-overhead ExecutionContext borrowing (raw pointer)
     * - Automatic fallback to default stream with explicit sync
     * - Memory-efficient staging for host-device conversions
     */
    export struct TransferOps
    {
        /**
         * @brief Copies tensor data with optional ExecutionContext
         *
         * Transfers data between tensors with automatic optimization based on
         * memory types. Borrows execution context for stream control with zero
         * overhead. Falls back to default stream when no context provided.
         *
         * Transfer directions:
         * - Host to Device (H2D): cudaMemcpyAsync with HostToDevice
         * - Device to Host (D2H): cudaMemcpyAsync with DeviceToHost
         * - Device to Device (D2D): Optimized kernel copy
         * - Host to Host (H2H): std::memcpy
         *
         * Type conversion:
         * - Same type: Direct memory copy (optimal)
         * - Different types: CUDA kernel conversion (D2D) or staged conversion (H2D/D2H)
         *
         * @tparam TSrcDataType Source tensor data type
         * @tparam TSrcMemoryResource Source memory resource type
         * @tparam TDstDataType Destination tensor data type
         * @tparam TDstMemoryResource Destination memory resource type
         * @param src Source tensor
         * @param dst Destination tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control (borrowed, not owned)
         *
         * @throws std::invalid_argument If tensor shapes don't match
         * @throws std::runtime_error If CUDA device is invalid or operations fail
         *
         * @note exec_context must outlive this function call
         * @note When exec_context provided, caller controls synchronization
         * @note When exec_context is null, uses default stream and synchronizes before returning
         * @note For H2D/D2H with type conversion, uses temporary device staging buffers
         *
         * Example:
         * @code
         * // With explicit context (caller manages sync)
         * auto ctx = std::make_unique<CudaExecutionContext>(0);
         * copy(src_tensor, dst_tensor, ctx.get());
         * // ... queue more operations
         * ctx->synchronize();
         *
         * // Without context (automatic sync)
         * copy(src_tensor, dst_tensor);  // Returns after sync completes
         * @endcode
         */
        template<TensorDataType TSrcDataType, typename TSrcMemoryResource,
                 TensorDataType TDstDataType, typename TDstMemoryResource>
            requires isValidTensor<TSrcDataType, TSrcMemoryResource> && isValidTensor<TDstDataType, TDstMemoryResource>
        static void copy(
            const Tensor<TSrcDataType, TSrcMemoryResource>& src,
            Tensor<TDstDataType, TDstMemoryResource>& dst,
            IExecutionContext* exec_context = nullptr )
        {
            if (src.shape() != dst.shape())
            {
                throw std::invalid_argument(
                    "Source and destination tensors must have the same shape for copy"
                );
            }

            if (src.size() == 0)
            {
                return;
            }

            // Determine stream and synchronization requirements
            cudaStream_t stream;
            bool needs_sync = false;
            int device_id = -1;

            if (exec_context)
            {
                auto* cuda_exec_context = cast_context<DeviceType::Cuda>( exec_context );
                
                stream = cuda_exec_context->getStream();
                device_id = cuda_exec_context->getDeviceId();
            }
            else
            {
                if constexpr (!TSrcMemoryResource::is_host_accessible)
                {
                    auto src_device = std::dynamic_pointer_cast<CudaDevice>(src.getDevice());
                    if (src_device)
                    {
                        device_id = src_device->getDeviceId();
                    }
                }

                if (device_id < 0 && !TDstMemoryResource::is_host_accessible)
                {
                    auto dst_device = std::dynamic_pointer_cast<CudaDevice>( dst.getDevice() );
                    if (dst_device)
                    {
                        device_id = dst_device->getDeviceId();
                    }
                }

                if (device_id < 0)
                {
                    throw std::runtime_error(
                        "CUDA transfer operations require at least one CUDA tensor"
                    );
                }

                Cuda::setCurrentDevice( device_id );

                stream = nullptr;  // Default stream
                needs_sync = true;  // Must sync default stream before returning
            }

            // Get data pointers using protected ITensor API
            const void* src_data = static_cast<const ITensor&>(src).rawData();
            void* dst_data = static_cast<ITensor&>(dst).rawData();

            if (!src_data || !dst_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for copy" );
            }

            // Dispatch based on memory accessibility and data types
            constexpr bool src_host = TSrcMemoryResource::is_host_accessible;
            constexpr bool dst_host = TDstMemoryResource::is_host_accessible;
            constexpr bool same_type = (TSrcDataType == TDstDataType);

            if constexpr (!src_host && !dst_host)
            {
                // Device to device
                if constexpr (same_type)
                {
                    copyDeviceToDevice<TSrcDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
                else
                {
                    copyDeviceToDeviceWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
            }
            else if constexpr (src_host && !dst_host)
            {
                // Host to device
                if constexpr (same_type)
                {
                    copyHostToDevice<TSrcDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
                else
                {
                    copyHostToDeviceWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
            }
            else if constexpr (!src_host && dst_host)
            {
                // Device to host
                if constexpr (same_type)
                {
                    copyDeviceToHost<TSrcDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
                else
                {
                    copyDeviceToHostWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), stream, device_id
                    );
                }
            }
            else
            {
                // Host to host
                if constexpr (same_type)
                {
                    copyHostToHost<TSrcDataType>( src_data, dst_data, src.size() );
                }
                else
                {
                    copyHostToHostWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size()
                    );
                }
            }

            if (needs_sync)
            {
                // Synchronize default stream to ensure completion
                cudaStreamSynchronize( stream );
            }
        }

    private:
        // ================================================================
        // Helper Methods
        // ================================================================

        /**
         * @brief Gets raw data pointer from tensor
         *
         * Uses public data() method for host-accessible tensors.
         * For device-only tensors, uses buffer's data() via protected access.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static const void* getDataPointer(
            const Tensor<TDataType, TMemoryResource>& tensor )
        {
            if constexpr (TMemoryResource::is_host_accessible)
            {
                // Use public data() method
                return tensor.data();
            }
            else
            {
                // Device-only memory - access via ITensor interface
                return static_cast<const ITensor&>(tensor).data();
            }
        }

        // ================================================================
        // Same-Type Transfer Operations (No Conversion)
        // ================================================================

        template<TensorDataType TDataType>
        static void copyDeviceToDevice(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            const auto* typed_src = static_cast<const NativeType*>(src_data);
            auto* typed_dst = static_cast<NativeType*>(dst_data);

            Cuda::launch_fast_copy_kernel<NativeType>(
                typed_src, typed_dst, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyHostToDevice(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync(
                dst_data, src_data, bytes,
                cudaMemcpyHostToDevice,
                stream
            );

            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyDeviceToHost(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync(
                dst_data, src_data, bytes,
                cudaMemcpyDeviceToHost,
                stream
            );

            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyHostToHost(
            const void* src_data,
            void* dst_data,
            size_t count )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            std::memcpy( dst_data, src_data, bytes );
        }

        // ================================================================
        // Type Conversion Transfer Operations
        // ================================================================

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyDeviceToDeviceWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            Cuda::launch_convert_copy_kernel<SrcType, DstType>(
                typed_src, typed_dst, count, stream
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToDeviceWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            if constexpr (TSrcDataType == TDstDataType)
            {
                copyHostToDevice<TSrcDataType>( src_data, dst_data, count, stream, device_id );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            constexpr size_t src_element_size = TensorDataTypeTraits<TSrcDataType>::size_in_bytes;
            const size_t src_bytes = count * src_element_size;

            // Allocate temporary device buffer for source data
            SrcType* temp_device_src = nullptr;
            cudaError_t status = cudaMallocAsync(
                reinterpret_cast<void**>(&temp_device_src),
                src_bytes,
                stream
            );
            cudaCheckStatus( status, std::source_location::current() );

            try
            {
                // Copy host source to device temporary
                status = cudaMemcpyAsync(
                    temp_device_src, src_data, src_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                );
                cudaCheckStatus( status, std::source_location::current() );

                // Convert on device
                auto* typed_dst = static_cast<DstType*>(dst_data);
                Cuda::launch_convert_copy_kernel<SrcType, DstType>(
                    temp_device_src, typed_dst, count, stream
                );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                // Free temporary buffer
                cudaFreeAsync( temp_device_src, stream );
            }
            catch (...)
            {
                cudaFreeAsync( temp_device_src, stream );
                throw;
            }
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyDeviceToHostWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count,
            cudaStream_t stream,
            int device_id )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            Cuda::setCurrentDevice( device_id );

            if constexpr (TSrcDataType == TDstDataType)
            {
                copyDeviceToHost<TSrcDataType>( src_data, dst_data, count, stream, device_id );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            constexpr size_t dst_element_size = TensorDataTypeTraits<TDstDataType>::size_in_bytes;
            const size_t dst_bytes = count * dst_element_size;

            // Allocate temporary device buffer for destination data
            DstType* temp_device_dst = nullptr;
            cudaError_t status = cudaMallocAsync(
                reinterpret_cast<void**>(&temp_device_dst),
                dst_bytes,
                stream
            );
            cudaCheckStatus( status, std::source_location::current() );

            try
            {
                // Convert on device
                const auto* typed_src = static_cast<const SrcType*>(src_data);
                Cuda::launch_convert_copy_kernel<SrcType, DstType>(
                    typed_src, temp_device_dst, count, stream
                );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                // Copy converted data to host
                status = cudaMemcpyAsync(
                    dst_data, temp_device_dst, dst_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                );
                cudaCheckStatus( status, std::source_location::current() );

                // Free temporary buffer
                cudaFreeAsync( temp_device_dst, stream );
            }
            catch (...)
            {
                cudaFreeAsync( temp_device_dst, stream );
                throw;
            }
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToHostWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count )
        {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            if constexpr (TSrcDataType == TDstDataType)
            {
                copyHostToHost<TSrcDataType>( src_data, dst_data, count );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            // CPU-based type conversion loop
            for (size_t i = 0; i < count; ++i)
            {
                typed_dst[i] = static_cast<DstType>( typed_src[i] );
            }
        }
    };
}