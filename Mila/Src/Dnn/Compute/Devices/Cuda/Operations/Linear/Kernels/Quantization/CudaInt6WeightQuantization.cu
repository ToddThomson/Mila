/**
 * @file CudaInt6WeightQuantization.cu
 * @brief Per-group BF16->INT6 weight quantization kernel: one warp per group of 32.
 *
 * The rule is Int6Packing.ixx's, and every arithmetic step uses an explicitly rounded intrinsic so that no
 * compiler setting (fast math, contraction) can move a code.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>
#include <format>
#include <cstddef>
#include <cstdint>
#include "CudaInt6WeightQuantization.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kGroupSize = 32;
        constexpr int kGroupsPerBlock = 8;
        constexpr unsigned kFullWarp = 0xFFFFFFFFu;

        /**
         * Grid (out_features, ceil(groups / kGroupsPerBlock)); block kGroupsPerBlock warps. Warp w of block
         * (row, y) encodes group y * kGroupsPerBlock + w of that row; lane l holds its element l.
         */
        __global__ void quantize_int6_per_group_kernel(
            const __nv_bfloat16* __restrict__ source,
            uint8_t*             __restrict__ packed,
            __half*              __restrict__ scales,
            int                              in_features )
        {
            const int lane = static_cast<int>( threadIdx.x ) & 31;
            const int group = static_cast<int>( blockIdx.y ) * kGroupsPerBlock + ( static_cast<int>( threadIdx.x ) >> 5 );
            const int groups_per_row = in_features / kGroupSize;

            if ( group >= groups_per_row )
                return;

            const int64_t row = blockIdx.x;
            const int column = group * kGroupSize + lane;
            const float value = __bfloat162float( source[ row * in_features + column ] );

            // Largest magnitude, lowest lane on a tie: the reference keeps the first value it sees.
            float magnitude = fabsf( value );
            int holder = lane;

            for ( int offset = 16; offset > 0; offset >>= 1 )
            {
                const float other_magnitude = __shfl_down_sync( kFullWarp, magnitude, offset );
                const int other_holder = __shfl_down_sync( kFullWarp, holder, offset );

                if ( other_magnitude > magnitude || ( other_magnitude == magnitude && other_holder < holder ) )
                {
                    magnitude = other_magnitude;
                    holder = other_holder;
                }
            }

            magnitude = __shfl_sync( kFullWarp, magnitude, 0 );
            holder = __shfl_sync( kFullWarp, holder, 0 );

            // The reference's extreme starts at +0 and changes only on a strictly larger magnitude, so an
            // all-zero group keeps +0 even when it holds -0 values.
            const float held = __shfl_sync( kFullWarp, value, holder );
            const float extreme = magnitude > 0.0f ? held : 0.0f;

            const float scale = __fdiv_rn( extreme, -32.0f );
            const float inverse = scale != 0.0f ? __fdiv_rn( 1.0f, scale ) : 0.0f;
            const float shifted = __fadd_rn( __fmul_rn( value, inverse ), 32.5f );
            const int code = min( 63, __float2int_rz( shifted ) );

            uint8_t* packed_row = packed + row * ( in_features / 2 + in_features / 4 );

            const int odd_code = __shfl_down_sync( kFullWarp, code, 1 );

            if ( ( lane & 1 ) == 0 )
            {
                packed_row[ column / 2 ] = static_cast<uint8_t>( ( code & 0xF ) | ( ( odd_code & 0xF ) << 4 ) );
            }

            const int high_1 = odd_code >> 4;
            const int high_2 = __shfl_down_sync( kFullWarp, code, 2 ) >> 4;
            const int high_3 = __shfl_down_sync( kFullWarp, code, 3 ) >> 4;

            if ( ( lane & 3 ) == 0 )
            {
                packed_row[ in_features / 2 + column / 4 ] =
                    static_cast<uint8_t>( ( code >> 4 ) | ( high_1 << 2 ) | ( high_2 << 4 ) | ( high_3 << 6 ) );
            }

            if ( lane == 0 )
            {
                scales[ row * groups_per_row + group ] = __float2half_rn( scale );
            }
        }
    }

    void cuda_quantize_int6_per_group(
        const void*  src_bf16,
        void*        dst_packed,
        void*        dst_scales,
        int64_t      out_features,
        int64_t      in_features,
        int          group_size,
        void*        dev_staging,
        size_t       staging_bytes,
        cudaStream_t stream )
    {
        if ( group_size != kGroupSize || in_features % kGroupSize != 0 )
        {
            throw std::runtime_error( std::format(
                "cuda_quantize_int6_per_group: group_size {} with in_features {} is unsupported; "
                "the group must be {} and divide in_features", group_size, in_features, kGroupSize ) );
        }

        const size_t row_bytes = static_cast<size_t>( in_features ) * sizeof( __nv_bfloat16 );

        if ( staging_bytes < row_bytes )
        {
            throw std::runtime_error( std::format(
                "cuda_quantize_int6_per_group: staging buffer of {} bytes cannot hold one row of {} bytes",
                staging_bytes, row_bytes ) );
        }

        const int64_t rows_per_chunk = static_cast<int64_t>( staging_bytes / row_bytes );
        const int64_t groups_per_row = in_features / kGroupSize;
        const int64_t packed_row_bytes = in_features / 2 + in_features / 4;

        for ( int64_t row = 0; row < out_features; row += rows_per_chunk )
        {
            const int64_t rows = std::min( rows_per_chunk, out_features - row );
            const auto* chunk_source = static_cast<const __nv_bfloat16*>( src_bf16 ) + row * in_features;

            cudaError_t error = cudaMemcpyAsync(
                dev_staging, chunk_source, static_cast<size_t>( rows ) * row_bytes, cudaMemcpyHostToDevice, stream );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "cuda_quantize_int6_per_group: source upload failed at row {}: {}",
                    row, cudaGetErrorString( error ) ) );
            }

            const dim3 grid(
                static_cast<unsigned>( rows ),
                static_cast<unsigned>( ( groups_per_row + kGroupsPerBlock - 1 ) / kGroupsPerBlock ) );

            quantize_int6_per_group_kernel<<<grid, kGroupsPerBlock * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>( dev_staging ),
                static_cast<uint8_t*>( dst_packed ) + row * packed_row_bytes,
                static_cast<__half*>( dst_scales ) + row * groups_per_row,
                static_cast<int>( in_features ) );

            error = cudaGetLastError();

            if ( error != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "cuda_quantize_int6_per_group: kernel launch failed: {}", cudaGetErrorString( error ) ) );
            }
        }
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
