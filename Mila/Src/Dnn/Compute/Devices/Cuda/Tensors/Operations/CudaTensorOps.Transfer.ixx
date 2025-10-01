/**
 * @file CudaTensorOps.Transfer.ixx
 * @brief CUDA tensor transfer operations partition
 *
 * This partition provides modular tensor transfer operations using the compute backend
 * device context infrastructure. Wraps the existing TensorCopy.cu kernel functionality
 * with a clean, type-safe interface for host-to-device, device-to-host, and device-to-device
 * transfers with optional type conversion.
 */

module;
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cstring>
#include "Kernels/Transfer.Copy.h"

export module Dnn.TensorOps:Transfer.Cuda;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.CudaTensorDataType;
import Compute.CudaDeviceContext;
import Compute.CudaMemoryResource;
import Cuda.Error;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	// Forward declaration of primary template
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
        struct TensorOps<Compute::CudaComputeDeviceTag>
    {
        // ================================================================
        // Same-Type Transfer Operations (No Conversion)
        // ================================================================

        /**
         * @brief Fast same-type device-to-device copy
         *
         * Optimized for cases where source and destination have identical types,
         * using vectorized memory operations for maximum throughput.
         */
        template<TensorDataType TDataType>
        static void copyDeviceToDevice( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::native_type;

            const auto* typed_src = static_cast<const NativeType*>(src_data);
            auto* typed_dst = static_cast<NativeType*>(dst_data);

            // Call host-side wrapper function (implemented in Transfer.Copy.cu)
            launch_fast_copy_kernel<NativeType>(
                typed_src, typed_dst, count, context->getStream() );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        /**
         * @brief Host-to-device copy with same types
         *
         * Transfers data from host memory to device memory using CUDA memcpy
         * with proper stream synchronization.
         */
        template<TensorDataType TDataType>
        static void copyHostToDevice( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync( dst_data, src_data, bytes,
                cudaMemcpyHostToDevice, context->getStream() );

            cudaCheckStatus( status, std::source_location::current() );
        }

        /**
         * @brief Device-to-host copy with same types
         *
         * Transfers data from device memory to host memory using CUDA memcpy
         * with proper stream synchronization.
         */
        template<TensorDataType TDataType>
        static void copyDeviceToHost( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            cudaError_t status = cudaMemcpyAsync( dst_data, src_data, bytes,
                cudaMemcpyDeviceToHost, context->getStream() );
            cudaCheckStatus( status, std::source_location::current() );
        }

        /**
         * @brief Host-to-host copy with same types
         *
         * Transfers data between host memory locations using standard memcpy.
         * Provided for API consistency across transfer operations.
         */
        template<TensorDataType TDataType>
        static void copyHostToHost( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context = nullptr ) {
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

        /**
         * @brief Device-to-device copy with type conversion
         *
         * Transfers data between device memory locations with automatic type
         * conversion using optimized CUDA kernels with proper precision handling.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyDeviceToDeviceWithConversion( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::native_type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::native_type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            // Call host-side wrapper for type conversion kernel
            launch_convert_copy_kernel<SrcType, DstType>(
                typed_src, typed_dst, count, context->getStream() );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }

        /**
         * @brief Host-to-device copy with type conversion
         *
         * Transfers data from host to device memory with type conversion.
         * Uses temporary device memory for conversion when needed.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToDeviceWithConversion( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            if constexpr (TSrcDataType == TDstDataType) {
                // Same types - use direct transfer
                copyHostToDevice<TSrcDataType>( src_data, dst_data, count, context );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::type;

            // Allocate temporary device memory for source data
            constexpr size_t src_element_size = TensorDataTypeTraits<TSrcDataType>::size_in_bytes;
            const size_t src_bytes = count * src_element_size;

            SrcType* temp_device_src = nullptr;
            cudaError_t status = cudaMallocAsync( reinterpret_cast<void**>(&temp_device_src),
                src_bytes, context->getStream() );
            cudaCheckStatus( status, std::source_location::current() );

            try {
                // Transfer host data to temporary device memory
                status = cudaMemcpyAsync( temp_device_src, src_data, src_bytes,
                    cudaMemcpyHostToDevice, context->getStream() );
                cudaCheckStatus( status, std::source_location::current() );

                // Convert on device using wrapper function
                auto* typed_dst = static_cast<DstType*>(dst_data);
                launch_convert_copy_kernel<SrcType, DstType>(
                    temp_device_src, typed_dst, count, context->getStream() );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                // Cleanup temporary memory
                cudaFreeAsync( temp_device_src, context->getStream() );
            }
            catch (...) {
                cudaFreeAsync( temp_device_src, context->getStream() );
                throw;
            }
        }

        /**
         * @brief Device-to-host copy with type conversion
         *
         * Transfers data from device to host memory with type conversion.
         * Uses temporary device memory for conversion when needed.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyDeviceToHostWithConversion( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            if constexpr (TSrcDataType == TDstDataType) {
                // Same types - use direct transfer
                copyDeviceToHost<TSrcDataType>( src_data, dst_data, count, context );
                return;
            }

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::type;

            // Allocate temporary device memory for converted data
            constexpr size_t dst_element_size = TensorDataTypeTraits<TDstDataType>::size_in_bytes;
            const size_t dst_bytes = count * dst_element_size;

            DstType* temp_device_dst = nullptr;
            cudaError_t status = cudaMallocAsync( reinterpret_cast<void**>(&temp_device_dst),
                dst_bytes, context->getStream() );
            cudaCheckStatus( status, std::source_location::current() );

            try {
                // Convert on device using wrapper function
                const auto* typed_src = static_cast<const SrcType*>(src_data);
                launch_convert_copy_kernel<SrcType, DstType>(
                    typed_src, temp_device_dst, count, context->getStream() );

                cudaError_t kernel_status = cudaGetLastError();
                cudaCheckStatus( kernel_status, std::source_location::current() );

                // Transfer converted data to host
                status = cudaMemcpyAsync( dst_data, temp_device_dst, dst_bytes,
                    cudaMemcpyDeviceToHost, context->getStream() );
                cudaCheckStatus( status, std::source_location::current() );

                // Cleanup temporary memory
                cudaFreeAsync( temp_device_dst, context->getStream() );
            }
            catch (...) {
                cudaFreeAsync( temp_device_dst, context->getStream() );
                throw;
            }
        }

        /**
         * @brief Host-to-host copy with type conversion
         *
         * Transfers data between host memory locations with type conversion.
         * Uses CPU-based conversion for host-accessible data.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToHostWithConversion( const void* src_data, void* dst_data,
            size_t count, std::shared_ptr<CudaDeviceContext> context = nullptr ) {
            if (!src_data || !dst_data || count == 0) {
                return;
            }

            if constexpr (TSrcDataType == TDstDataType) {
                copyHostToHost<TSrcDataType>( src_data, dst_data, count, context );
                return;
            }

            // CPU-based type conversion for host data
            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            for (size_t i = 0; i < count; ++i) {
                typed_dst[i] = static_cast<DstType>( typed_src[i] );
            }
        }

        // ================================================================
        // Strided Transfer Operations
        // ================================================================

        /**
         * @brief Device-to-device copy with stride support and type conversion
         *
         * Handles non-contiguous memory layouts using stride information for
         * both source and destination tensors with optional type conversion.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyStridedDeviceToDevice( const void* src_data, void* dst_data,
            size_t count, size_t src_stride, size_t dst_stride,
            std::shared_ptr<CudaDeviceContext> context ) {
            if (!src_data || !dst_data || count == 0 || !context) {
                return;
            }

            context->makeCurrent();

            using SrcType = typename Cuda::TensorDataTypeMap<TSrcDataType>::type;
            using DstType = typename Cuda::TensorDataTypeMap<TDstDataType>::type;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            // Call host-side wrapper for strided copy kernel
            launch_convert_copy_strided_kernel<SrcType, DstType>(
                typed_src, typed_dst, count, src_stride, dst_stride, context->getStream() );

            cudaError_t status = cudaGetLastError();
            cudaCheckStatus( status, std::source_location::current() );
        }
    };
}