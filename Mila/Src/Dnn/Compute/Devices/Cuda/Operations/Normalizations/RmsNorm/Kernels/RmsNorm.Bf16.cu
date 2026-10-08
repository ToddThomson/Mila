/**
 * @file RmsNorm.Bf16.cu
 * @brief BF16 CUDA kernels and host launchers for RMS normalization.
 *
 * All arithmetic is performed in float32; BF16 is used only for I/O.
 * Requires SM >= 8.0 (Ampere) for native BF16 and BF16 atomicAdd support.
 */

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "RmsNorm.cuh"

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    namespace
    {
        constexpr int kValuesPerVector = 8;
        constexpr int kMaximumRowThreads = 1024;

        // Sum of v over the block, returned to every thread.
        __device__ float blockSum( float v )
        {
            __shared__ float warp_sums[ kMaximumRowThreads / WARP_SIZE ];
            __shared__ float total;

            const int lane_id = threadIdx.x % WARP_SIZE;
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

            for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
                v += __shfl_xor_sync( 0xffffffff, v, offset );

            if ( lane_id == 0 )
                warp_sums[ warp_id ] = v;

            __syncthreads();

            if ( warp_id == 0 )
            {
                v = lane_id < num_warps ? warp_sums[ lane_id ] : 0.0f;

                for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
                    v += __shfl_xor_sync( 0xffffffff, v, offset );

                if ( lane_id == 0 )
                    total = v;
            }

            __syncthreads();

            return total;
        }

        bool isVectorAligned( const void* pointer )
        {
            return reinterpret_cast<std::uintptr_t>( pointer ) % sizeof( uint4 ) == 0;
        }
    }

    // One block per contiguous row, 16-byte loads. A decode step normalizes a single row, which the
    // warp-per-row kernel below runs on one warp of the whole GPU; a block per row keeps every row's
    // arithmetic the same whatever the row count, so prefill and decode share one normalization.
    // Each thread writes only the vectors it read, so out may alias inp.
    __global__ void rmsnorm_forward_bf16_row_kernel(
        __nv_bfloat16*                    out,
        __nv_bfloat16* __restrict__       rstd,
        const __nv_bfloat16*              inp,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
        int norm_dim, float epsilon, float weight_offset, int weight_groups )
    {
        const size_t row_offset = static_cast<size_t>( blockIdx.x ) * static_cast<size_t>( norm_dim );
        const size_t weight_offset_elements = static_cast<size_t>( blockIdx.x % weight_groups ) * static_cast<size_t>( norm_dim );
        const uint4* x = reinterpret_cast<const uint4*>( inp + row_offset );
        uint4* o = reinterpret_cast<uint4*>( out + row_offset );
        const uint4* w = weight ? reinterpret_cast<const uint4*>( weight + weight_offset_elements ) : nullptr;
        const uint4* b = bias ? reinterpret_cast<const uint4*>( bias + weight_offset_elements ) : nullptr;
        const int vectors = norm_dim / kValuesPerVector;

        float m2 = 0.0f;

        for ( int v = threadIdx.x; v < vectors; v += blockDim.x )
        {
            const uint4 packed = x[ v ];
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>( &packed );

            #pragma unroll
            for ( int k = 0; k < kValuesPerVector / 2; ++k )
            {
                const float2 value = __bfloat1622float2( pairs[ k ] );
                m2 += value.x * value.x + value.y * value.y;
            }
        }

        m2 = blockSum( m2 );

        const float rstd_val = rsqrtf( m2 / static_cast<float>( norm_dim ) + epsilon );

        if ( threadIdx.x == 0 && rstd != nullptr )
            rstd[ blockIdx.x ] = __float2bfloat16( rstd_val );

        for ( int v = threadIdx.x; v < vectors; v += blockDim.x )
        {
            const uint4 packed = x[ v ];
            const uint4 packed_weight = w ? w[ v ] : uint4{};
            const uint4 packed_bias = b ? b[ v ] : uint4{};
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>( &packed );
            const __nv_bfloat162* weight_pairs = reinterpret_cast<const __nv_bfloat162*>( &packed_weight );
            const __nv_bfloat162* bias_pairs = reinterpret_cast<const __nv_bfloat162*>( &packed_bias );

            uint4 result;
            __nv_bfloat162* result_pairs = reinterpret_cast<__nv_bfloat162*>( &result );

            #pragma unroll
            for ( int k = 0; k < kValuesPerVector / 2; ++k )
            {
                const float2 xv = __bfloat1622float2( pairs[ k ] );
                float2 wv{ 1.0f, 1.0f };
                const float2 bv = b ? __bfloat1622float2( bias_pairs[ k ] ) : float2{ 0.0f, 0.0f };

                if ( w )
                {
                    wv = __bfloat1622float2( weight_pairs[ k ] );
                    wv.x += weight_offset;
                    wv.y += weight_offset;
                }

                result_pairs[ k ] = __floats2bfloat162_rn(
                    xv.x * rstd_val * wv.x + bv.x,
                    xv.y * rstd_val * wv.y + bv.y );
            }

            o[ v ] = result;
        }
    }

    // Each warp processes one normalization slice. Inputs are loaded as BF16
    // and immediately widened to float for all arithmetic. rstd is stored as
    // BF16 to match the typed buffer, with sufficient range for O(1) values.
    __global__ void rmsnorm_forward_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        __nv_bfloat16* __restrict__       rstd,
        const __nv_bfloat16* __restrict__ inp,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
        int num_slices, int norm_dim, int inner_size, float epsilon,
        float weight_offset, int weight_groups )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
            return;

        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const __nv_bfloat16* x = inp
            + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size)
            + inner_idx;
        __nv_bfloat16* o = out
            + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size)
            + inner_idx;
        const int weight_base = (outer_idx % weight_groups) * norm_dim;

        float m2 = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            float val = __bfloat162float( x[ static_cast<size_t>( i ) * static_cast<size_t>( inner_size ) ] );
            m2 += val * val;
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
            m2 += __shfl_down_sync( 0xffffffff, m2, offset );

        m2 = __shfl_sync( 0xffffffff, m2, 0 );

        float rstd_val = rsqrtf( m2 / static_cast<float>(norm_dim) + epsilon );

        if ( lane_id == 0 && rstd != nullptr )
            rstd[ idx ] = __float2bfloat16( rstd_val );

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t stride = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float xv = __bfloat162float( x[ stride ] );
            float w = weight ? (__bfloat162float( weight[ weight_base + i ] ) + weight_offset) : 1.0f;
            float b = bias ? __bfloat162float( bias[ weight_base + i ] ) : 0.0f;
            o[ stride ] = __float2bfloat16( xv * rstd_val * w + b );
        }
    }

    // Each warp processes one normalization slice. Parameter gradients are
    // accumulated via BF16 atomicAdd (SM >= 8.0 required, guaranteed by BF16 support).
    __global__ void rmsnorm_backward_bf16_kernel(
        __nv_bfloat16* __restrict__       dinp,
        __nv_bfloat16* __restrict__       dweight,
        __nv_bfloat16* __restrict__       dbias,
        const __nv_bfloat16* __restrict__ dout,
        const __nv_bfloat16* __restrict__ inp,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ rstd,
        int num_slices, int norm_dim, int inner_size, int weight_groups )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
            return;

        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const __nv_bfloat16* x = inp
            + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size)
            + inner_idx;
        const __nv_bfloat16* dy = dout
            + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size)
            + inner_idx;
        __nv_bfloat16* dx = dinp
            + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size)
            + inner_idx;
        const int weight_base = (outer_idx % weight_groups) * norm_dim;

        float rstd_val = __bfloat162float( rstd[ idx ] );
        float inv_n = 1.0f / static_cast<float>(norm_dim);

        float sum_gx = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t stride = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float x_val = __bfloat162float( x[ stride ] );
            float dy_val = __bfloat162float( dy[ stride ] );
            float w_val = weight ? __bfloat162float( weight[ weight_base + i ] ) : 1.0f;

            float g = dy_val * w_val;
            sum_gx += g * x_val;

            if ( dweight )
                atomicAdd( &dweight[ weight_base + i ], __float2bfloat16( dy_val * (x_val * rstd_val) ) );

            if ( dbias )
                atomicAdd( &dbias[ weight_base + i ], __float2bfloat16( dy_val ) );
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
            sum_gx += __shfl_down_sync( 0xffffffff, sum_gx, offset );

        sum_gx = __shfl_sync( 0xffffffff, sum_gx, 0 );

        float rstd3 = rstd_val * rstd_val * rstd_val;
        float correction = rstd3 * inv_n * sum_gx;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t stride = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float x_val = __bfloat162float( x[ stride ] );
            float dy_val = __bfloat162float( dy[ stride ] );
            float w_val = weight ? __bfloat162float( weight[ weight_base + i ] ) : 1.0f;

            dx[ stride ] = __float2bfloat16( rstd_val * (dy_val * w_val) - x_val * correction );
        }
    }

    // =========================================================================
    // Host launchers

    void cuda_rmsnorm_forward_bf16(
        __nv_bfloat16* Y, __nv_bfloat16* rstd,
        const __nv_bfloat16* X, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        float weight_offset,
        int weight_groups,
        cudaStream_t stream )
    {
        const bool contiguous_rows = inner_size == 1 && norm_dim % kValuesPerVector == 0
            && isVectorAligned( Y ) && isVectorAligned( X ) && isVectorAligned( weight ) && isVectorAligned( bias );

        if ( contiguous_rows )
        {
            const int vectors = norm_dim / kValuesPerVector;
            const int block_size = std::min( kMaximumRowThreads, (vectors + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE );

            if ( outer_size > 0 )
            {
                rmsnorm_forward_bf16_row_kernel<<<outer_size, block_size, 0, stream>>>(
                    Y, rstd, X, weight, bias, norm_dim, epsilon, weight_offset, weight_groups );
            }

            cudaCheck( cudaGetLastError() );

            return;
        }

        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        rmsnorm_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (
            Y, rstd, X, weight, bias, num_slices, norm_dim, inner_size, epsilon, weight_offset, weight_groups);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_rmsnorm_backward_bf16(
        __nv_bfloat16* dX, __nv_bfloat16* dweight, __nv_bfloat16* dbias,
        const __nv_bfloat16* dY, const __nv_bfloat16* X, const __nv_bfloat16* weight,
        const __nv_bfloat16* rstd,
        int outer_size, int inner_size, int norm_dim,
        int weight_groups,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        rmsnorm_backward_bf16_kernel <<< grid_size, block_size, 0, stream >>> (
            dX, dweight, dbias, dY, X, weight, rstd, num_slices, norm_dim, inner_size, weight_groups);

        cudaCheck( cudaGetLastError() );
    }
}