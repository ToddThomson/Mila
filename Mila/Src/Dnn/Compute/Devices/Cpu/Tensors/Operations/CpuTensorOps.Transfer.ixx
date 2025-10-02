/**
 * @file CpuTensorOps.Transfer.ixx
 * @brief CPU tensor transfer operations partition
 */

module;
#include <cstring>
#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Dnn.TensorOps:Transfer.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CPU specialization of TensorOps for transfer operations.
     *
     * This specialization provides CPU-specific implementations of tensor
     * transfer operations for the compute device tag `Compute::CpuComputeDeviceTag`.
     * Handles both same-type and type-conversion transfers with host memory optimization.
     */
    export template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
    struct TensorOps<Compute::CpuComputeDeviceTag>
    {
        // ================================================================
        // Same-Type Transfer Operations (No Conversion)
        // ================================================================

        /**
         * @brief Fast same-type host-to-host copy
         *
         * Optimized for cases where source and destination have identical types,
         * using standard library copy operations for maximum efficiency.
         */
        template<TensorDataType TDataType>
        static void copyHostToHost( const void* src_data, void* dst_data, size_t count ) {
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
         * @brief Host-to-host copy with type conversion
         *
         * Transfers data between host memory locations with type conversion.
         * Uses CPU-based element-wise conversion for optimal host performance.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static void copyHostToHostWithConversion( const void* src_data, void* dst_data, size_t count ) {
            if (!src_data || !dst_data || count == 0) {
                return;
            }

            if constexpr (TSrcDataType == TDstDataType) {
                copyHostToHost<TSrcDataType>( src_data, dst_data, count );
                return;
            }

            // CPU-based type conversion for host data
            using SrcType = typename CpuTensorDataTypeTraits::template native_type<TSrcDataType>;
            using DstType = typename CpuTensorDataTypeTraits::template native_type<TDstDataType>;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            // Element-wise conversion with optimal loop
            for (size_t i = 0; i < count; ++i) {
                typed_dst[i] = static_cast<DstType>( typed_src[i] );
            }
        }

        // ================================================================
        // High-Level Tensor Copy Operations
        // ================================================================

        /**
         * @brief Copies tensor data between existing tensors
         *
         * Main entry point for the high-level copy() function. Handles both same-type
         * and type-conversion scenarios between pre-allocated tensors.
         *
         * @tparam TSrcDataType Source tensor data type
         * @tparam TSrcMemoryResource Source memory resource type
         * @tparam TDstDataType Destination tensor data type
         * @tparam TDstMemoryResource Destination memory resource type
         * @param src Source tensor to copy from
         * @param dst Destination tensor to copy to (must be pre-allocated)
         */
        template<TensorDataType TSrcDataType, typename TSrcMemoryResource, 
                 TensorDataType TDstDataType, typename TDstMemoryResource>
            requires isValidTensor<TSrcDataType, TSrcMemoryResource> && isValidTensor<TDstDataType, TDstMemoryResource>
        static void copy( const Tensor<TSrcDataType, TSrcMemoryResource>& src, Tensor<TDstDataType, TDstMemoryResource>& dst ) {

            if (src.shape() != dst.shape()) {
                throw std::invalid_argument( "Source and destination tensors must have the same shape for copy" );
            }

            if (src.size() == 0) {
                return; // Nothing to copy
            }

            // For CPU operations, we primarily handle host-accessible memory
            static_assert(TSrcMemoryResource::is_host_accessible,
                "CPU copy operations require host-accessible source memory");
            static_assert(TDstMemoryResource::is_host_accessible,
                "CPU copy operations require host-accessible destination memory");

            const void* src_data = src.rawData();
            void* dst_data = dst.rawData();

            if (!src_data || !dst_data) {
                throw std::runtime_error( "Invalid tensor data pointers for copy" );
            }

            // Dispatch to appropriate copy method based on data types
            if constexpr (TSrcDataType == TDstDataType) {
                // Same type - use fast memory copy
                copyHostToHost<TSrcDataType>( src_data, dst_data, src.size() );
            }
            else {
                // Different types - use conversion copy
                copyHostToHostWithConversion<TSrcDataType, TDstDataType>(
                    src_data, dst_data, src.size() );
            }
        }
    };
}