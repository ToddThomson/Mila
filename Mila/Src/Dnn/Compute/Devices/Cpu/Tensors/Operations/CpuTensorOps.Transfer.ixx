/**
 * @file CpuTensorOps.Transfer.ixx
 * @brief CPU tensor transfer operations partition
 *
 * Provides CPU-specific implementations of tensor transfer operations for
 * host-accessible memory. All operations execute synchronously with no device
 * synchronization overhead.
 *
 * ExecutionContext handling:
 * - Accepts ExecutionContext parameter for API consistency with device implementations
 * - Parameter is unused for CPU operations (all operations are synchronous)
 * - No stream management needed on CPU
 */

module;
#include <cstring>
#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Compute.CpuTensorOps:Transfer;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;

namespace Mila::Dnn::Compute::Cpu
{
    using namespace Mila::Dnn::Compute;

    namespace Detail
    {
        /**
         * @brief Fast raw-memory copy for host-accessible tensors of same abstract type.
         *
         * Copies `count` logical elements from `src_data` to `dst_data` using
         * a byte-wise memcpy based on the abstract element size for `TDataType`.
         *
         * @tparam TDataType Abstract tensor data type for src and dst
         * @param src_data Pointer to source memory (may be null)
         * @param dst_data Pointer to destination memory (may be null)
         * @param count Number of logical elements to copy
         *
         * @note No operation performed if either pointer is null or count is zero.
         */
        template<TensorDataType TDataType>
        inline void copyHostToHostImpl( const void* src_data, void* dst_data, size_t count ) {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;
            const size_t bytes = count * element_size;

            std::memcpy( dst_data, src_data, bytes );
        }

        /**
         * @brief Element-wise host copy with conversion between abstract data types.
         *
         * Copies `count` logical elements from `src_data` (type `TSrcDataType`) to
         * `dst_data` (type `TDstDataType`) performing a per-element conversion
         * via static_cast to the native CPU types.
         *
         * If the source and destination abstract types are the same, this forwards
         * to `copyHostToHostImpl`.
         *
         * @tparam TSrcDataType Abstract source tensor data type
         * @tparam TDstDataType Abstract destination tensor data type
         * @param src_data Pointer to source memory (may be null)
         * @param dst_data Pointer to destination memory (may be null)
         * @param count Number of logical elements to convert and copy
         *
         * @note No operation performed if either pointer is null or count is zero.
         * @note Loop form chosen to enable compiler auto-vectorization.
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        inline void copyHostToHostWithConversionImpl( const void* src_data, void* dst_data, size_t count ) {
            if (!src_data || !dst_data || count == 0)
            {
                return;
            }

            if constexpr (TSrcDataType == TDstDataType)
            {
                copyHostToHostImpl<TSrcDataType>( src_data, dst_data, count );
                return;
            }

            using SrcType = typename CpuTensorDataTypeTraits::template native_type<TSrcDataType>;
            using DstType = typename CpuTensorDataTypeTraits::template native_type<TDstDataType>;

            const auto* typed_src = static_cast<const SrcType*>(src_data);
            auto* typed_dst = static_cast<DstType*>(dst_data);

            for (size_t i = 0; i < count; ++i)
            {
                typed_dst[i] = static_cast<DstType>( typed_src[i] );
            }
        }
    }

    /**
     * @brief CPU specialization of TensorOps for transfer operations.
     *
     * This specialization provides CPU-specific implementations of tensor
     * transfer operations for the DeviceType::Cpu device type. Handles both
     * same-type and type-conversion transfers with host memory optimization.
     *
     * Key features:
     * - Direct memcpy for same-type transfers (optimal performance)
     * - Element-wise conversion for different-type transfers
     * - Accepts ExecutionContext for API consistency (unused on CPU)
     * - All operations are synchronous (no stream management)
     */
    export struct TransferOps
    {
        /**
         * @brief Copy tensor data between pre-allocated tensors.
         *
         * Validates that `src` and `dst` have identical shapes, ensures non-zero
         * element count, and dispatches to the appropriate same-type or
         * conversion copy implementation.
         *
         * @tparam TSrcDataType Abstract source tensor data type
         * @tparam TSrcMemoryResource Source memory resource type (must be host-accessible)
         * @tparam TDstDataType Abstract destination tensor data type
         * @tparam TDstMemoryResource Destination memory resource type (must be host-accessible)
         * @param src Source tensor (pre-allocated)
         * @param dst Destination tensor (pre-allocated)
         * @param exec_context Optional execution context pointer (ignored for CPU)
         *
         * @throws std::invalid_argument If tensor shapes do not match
         * @throws std::runtime_error If underlying data pointers are invalid
         *
         * @note This function is synchronous and does not use the execution context.
         */
        template<TensorDataType TSrcDataType, typename TSrcMemoryResource,
            TensorDataType TDstDataType, typename TDstMemoryResource>
            requires isValidTensor<TSrcDataType, TSrcMemoryResource>&& isValidTensor<TDstDataType, TDstMemoryResource>
        static void copy(
            const Tensor<TSrcDataType, TSrcMemoryResource>& src,
            Tensor<TDstDataType, TDstMemoryResource>& dst,
            [[maybe_unused]] IExecutionContext* exec_context = nullptr )
        {
            if (src.shape() != dst.shape())
            {
                throw std::invalid_argument( "Source and destination tensors must have the same shape for copy" );
            }

            if (src.size() == 0)
            {
                return; // Nothing to copy
            }

            // For CPU operations, we primarily handle host-accessible memory
            static_assert(TSrcMemoryResource::is_host_accessible,
                "CPU copy operations require host-accessible source memory");
            static_assert(TDstMemoryResource::is_host_accessible,
                "CPU copy operations require host-accessible destination memory");

            const void* src_data = src.data();
            void* dst_data = dst.data();

            if (!src_data || !dst_data)
            {
                throw std::runtime_error( "Invalid tensor data pointers for copy" );
            }

            if constexpr (TSrcDataType == TDstDataType)
            {
                hostCopyImpl<TSrcDataType>( src_data, dst_data, src.size() );
            }
            else
            {
                hostConvertImpl<TSrcDataType, TDstDataType>( src_data, dst_data, src.size() );
            }
        }

    private:
        /**
         * @brief Private wrapper forwarding to Detail::copyHostToHostImpl.
         *
         * Thin forwarding wrapper used to encapsulate the internal implementation.
         *
         * @tparam TDataType Abstract tensor data type
         * @param src_data Pointer to source memory
         * @param dst_data Pointer to destination memory
         * @param count Number of logical elements to copy
         */
        template<TensorDataType TDataType>
        static inline void hostCopyImpl( const void* src_data, void* dst_data, size_t count ) {
            Detail::copyHostToHostImpl<TDataType>( src_data, dst_data, count );
        }

        /**
         * @brief Private wrapper forwarding to Detail::copyHostToHostWithConversionImpl.
         *
         * Thin forwarding wrapper used to encapsulate the internal conversion
         * implementation.
         *
         * @tparam TSrcDataType Abstract source tensor data type
         * @tparam TDstDataType Abstract destination tensor data type
         * @param src_data Pointer to source memory
         * @param dst_data Pointer to destination memory
         * @param count Number of logical elements to convert and copy
         */
        template<TensorDataType TSrcDataType, TensorDataType TDstDataType>
        static inline void hostConvertImpl( const void* src_data, void* dst_data, size_t count ) {
            Detail::copyHostToHostWithConversionImpl<TSrcDataType, TDstDataType>( src_data, dst_data, count );
        }
    };
}