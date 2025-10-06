/**
 * @file CudaTensorOps.Transfer.ixx
 * @brief CUDA tensor transfer operations partition
 *
 * Provides tensor transfer operations using ExecutionContext for stream management.
 * TensorOps work with tensor data directly (via data()) and accept optional
 * ExecutionContext for explicit stream control.
 */

module;
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cstring>
#include "Kernels/Transfer.Copy.h"

export module Dnn.TensorOps:Transfer.Cuda;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.CudaTensorDataType;
import Compute.DeviceContext;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceContext;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;
import Cuda.Error;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
        struct TensorOps<Compute::CudaComputeDeviceTag>
    {
        // ================================================================
        // High-Level Tensor Copy Operations
        // ================================================================

        /**
         * @brief Copies tensor data with optional ExecutionContext
         *
         * Transfers data between tensors with automatic optimization based on
         * memory types. If no ExecutionContext is provided, creates a temporary
         * one for the operation.
         *
         * @tparam TSrcDataType Source tensor data type
         * @tparam TSrcMemoryResource Source memory resource type
         * @tparam TDstDataType Destination tensor data type
         * @tparam TDstMemoryResource Destination memory resource type
         * @param src Source tensor
         * @param dst Destination tensor (must be pre-allocated with matching shape)
         * @param exec_context Optional execution context for stream control
         */
        template<TensorDataType TSrcDataType, typename TSrcMemoryResource,
            TensorDataType TDstDataType, typename TDstMemoryResource>
            requires isValidTensor<TSrcDataType, TSrcMemoryResource>&&
        isValidTensor<TDstDataType, TDstMemoryResource>
            static void copy(
                const Tensor<TSrcDataType, TSrcMemoryResource>& src,
                Tensor<TDstDataType, TDstMemoryResource>& dst,
                std::shared_ptr<ExecutionContext> exec_context = nullptr )
        {
            // Validate tensor compatibility
            if (src.shape() != dst.shape()) {
                throw std::invalid_argument(
                    "Source and destination tensors must have the same shape for copy"
                );
            }

            if (src.size() == 0) {
                return; // Nothing to copy
            }

            // Get CUDA execution context
            std::shared_ptr<CudaExecutionContext> cuda_exec_ctx;

            if (exec_context) {
                // Use provided execution context
                cuda_exec_ctx = std::dynamic_pointer_cast<CudaExecutionContext>(exec_context);
                if (!cuda_exec_ctx) {
                    throw std::invalid_argument(
                        "CUDA tensor operations require CudaExecutionContext"
                    );
                }
            }
            else {
                // Create temporary execution context for this operation
                // Use src tensor's device if available, otherwise dst's device
                std::string device_name = src.getDeviceName();
                if (device_name.empty() || device_name == "CPU") {
                    device_name = dst.getDeviceName();
                }

                if (device_name.empty() || device_name == "CPU") {
                    throw std::runtime_error(
                        "CUDA transfer operations require at least one CUDA tensor"
                    );
                }

                auto device_ctx = DeviceContext::create( device_name );
                cuda_exec_ctx = std::dynamic_pointer_cast<CudaExecutionContext>(
                    ExecutionContext::create( device_ctx )
                );
            }

            // Get data pointers using public data() method
            const void* src_data = getDataPointer( src );
            void* dst_data = getDataPointer( dst );

            if (!src_data || !dst_data) {
                throw std::runtime_error( "Invalid tensor data pointers for copy" );
            }

            // Dispatch based on memory accessibility and data types
            constexpr bool src_host = TSrcMemoryResource::is_host_accessible;
            constexpr bool dst_host = TDstMemoryResource::is_host_accessible;
            constexpr bool same_type = (TSrcDataType == TDstDataType);

            if constexpr (!src_host && !dst_host) {
                // Device to device
                if constexpr (same_type) {
                    copyDeviceToDevice<TSrcDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
                else {
                    copyDeviceToDeviceWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
            }
            else if constexpr (src_host && !dst_host) {
                // Host to device
                if constexpr (same_type) {
                    copyHostToDevice<TSrcDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
                else {
                    copyHostToDeviceWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
            }
            else if constexpr (!src_host && dst_host) {
                // Device to host
                if constexpr (same_type) {
                    copyDeviceToHost<TSrcDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
                else {
                    copyDeviceToHostWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size(), cuda_exec_ctx
                    );
                }
            }
            else {
                // Host to host
                if constexpr (same_type) {
                    copyHostToHost<TSrcDataType>( src_data, dst_data, src.size() );
                }
                else {
                    copyHostToHostWithConversion<TSrcDataType, TDstDataType>(
                        src_data, dst_data, src.size()
                    );
                }
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
         * For device-only tensors, uses buffer's rawData() via protected access.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static const void* getDataPointer(
            const Tensor<TDataType, TMemoryResource>& tensor )
        {
            if constexpr (TMemoryResource::is_host_accessible) {
                // Use public data() method
                return tensor.data();
            }
            else {
                // Device-only memory - access via ITensor interface
                return static_cast<const ITensor&>(tensor).rawData();
            }
        }

        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void* getDataPointer( Tensor<TDataType, TMemoryResource>& tensor )
        {
            if constexpr (TMemoryResource::is_host_accessible) {
                // Use public data() method
                return tensor.data();
            }
            else {
                // Device-only memory - access via ITensor interface
                return static_cast<ITensor&>(tensor).rawData();
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
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            const auto* typed_src = static_cast<const NativeType*>(src_data);
            auto* typed_dst = static_cast<NativeType*>(dst_data);

            // Use execution context's stream
            launch_fast_copy_kernel<NativeType>(
                typed_src, typed_dst, count, exec_ctx->getStream()
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyHostToDevice(
            const void* src_data,
            void* dst_data,
            size_t count,
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync(
                dst_data, src_data, bytes,
                cudaMemcpyHostToDevice,
                exec_ctx->getStream()
            );

            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyDeviceToHost(
            const void* src_data,
            void* dst_data,
            size_t count,
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync(
                dst_data, src_data, bytes,
                cudaMemcpyDeviceToHost,
                exec_ctx->getStream()
            );

            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TDataType>
        static void copyHostToHost(
            const void* src_data,
            void* dst_data,
            size_t count )
        {
            if (!src_data || !dst_data || count == 0) {
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
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            launch_convert_copy_kernel<SrcType, DstType>(
                typed_src, typed_dst, count, exec_ctx->getStream()
            );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToDeviceWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count,
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            if constexpr (TSrcDataType == TDstDataType) {
                copyHostToDevice<TSrcDataType>( src_data, dst_data, count, exec_ctx );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            constexpr size_t src_element_size = TensorDataTypeTraits<TSrcDataType>::size_in_bytes;
            const size_t src_bytes = count * src_element_size;

            SrcType* temp_device_src = nullptr;
            cudaError_t status = cudaMallocAsync(
                reinterpret_cast<void**>(&temp_device_src),
                src_bytes,
                exec_ctx->getStream()
            );
            cudaCheckStatus( status, std::source_location::current() );

            try {
                status = cudaMemcpyAsync(
                    temp_device_src, src_data, src_bytes,
                    cudaMemcpyHostToDevice,
                    exec_ctx->getStream()
                );
                cudaCheckStatus( status, std::source_location::current() );

                auto* typed_dst = static_cast<DstType*>(dst_data);
                launch_convert_copy_kernel<SrcType, DstType>(
                    temp_device_src, typed_dst, count, exec_ctx->getStream()
                );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                cudaFreeAsync( temp_device_src, exec_ctx->getStream() );
            }
            catch (...) {
                cudaFreeAsync( temp_device_src, exec_ctx->getStream() );
                throw;
            }
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyDeviceToHostWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count,
            std::shared_ptr<CudaExecutionContext> exec_ctx )
        {
            if (!src_data || !dst_data || count == 0 || !exec_ctx) {
                return;
            }

            exec_ctx->getDeviceContext()->makeCurrent();

            if constexpr (TSrcDataType == TDstDataType) {
                copyDeviceToHost<TSrcDataType>( src_data, dst_data, count, exec_ctx );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            constexpr size_t dst_element_size = TensorDataTypeTraits<TDstDataType>::size_in_bytes;
            const size_t dst_bytes = count * dst_element_size;

            DstType* temp_device_dst = nullptr;
            cudaError_t status = cudaMallocAsync(
                reinterpret_cast<void**>(&temp_device_dst),
                dst_bytes,
                exec_ctx->getStream()
            );
            cudaCheckStatus( status, std::source_location::current() );

            try {
                const auto* typed_src = static_cast<const SrcType*>(src_data);
                launch_convert_copy_kernel<SrcType, DstType>(
                    typed_src, temp_device_dst, count, exec_ctx->getStream()
                );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                status = cudaMemcpyAsync(
                    dst_data, temp_device_dst, dst_bytes,
                    cudaMemcpyDeviceToHost,
                    exec_ctx->getStream()
                );
                cudaCheckStatus( status, std::source_location::current() );

                cudaFreeAsync( temp_device_dst, exec_ctx->getStream() );
            }
            catch (...) {
                cudaFreeAsync( temp_device_dst, exec_ctx->getStream() );
                throw;
            }
        }

        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToHostWithConversion(
            const void* src_data,
            void* dst_data,
            size_t count )
        {
            if (!src_data || !dst_data || count == 0) {
                return;
            }

            if constexpr (TSrcDataType == TDstDataType) {
                copyHostToHost<TSrcDataType>( src_data, dst_data, count );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            for (size_t i = 0; i < count; ++i) {
                typed_dst[i] = static_cast<DstType>( typed_src[i] );
            }
        }
    };
}