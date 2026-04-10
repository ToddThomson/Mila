/**
 * @file RmsNorm.Bf16.cu
 * @brief BF16 CUDA kernels and host launchers for RMS normalization.
 *
 * All arithmetic is performed in float32; BF16 is used only for I/O.
 * Requires SM >= 8.0 (Ampere) for native BF16 and BF16 atomicAdd support.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "RmsNorm.cuh"

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    // Each warp processes one normalization slice. Inputs are loaded as BF16
    // and immediately widened to float for all arithmetic. rstd is stored as
    // BF16 to match the typed buffer, with sufficient range for O(1) values.
    __global__ void rmsnorm_forward_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        __nv_bfloat16* __restrict__       rstd,
        const __nv_bfloat16* __restrict__ inp,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
        int num_slices, int norm_dim, int inner_size, float epsilon )
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
            float w = weight ? __bfloat162float( weight[ i ] ) : 1.0f;
            float b = bias ? __bfloat162float( bias[ i ] ) : 0.0f;
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
        int num_slices, int norm_dim, int inner_size )
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

        float rstd_val = __bfloat162float( rstd[ idx ] );
        float inv_n = 1.0f / static_cast<float>(norm_dim);

        float sum_gx = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t stride = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float x_val = __bfloat162float( x[ stride ] );
            float dy_val = __bfloat162float( dy[ stride ] );
            float w_val = weight ? __bfloat162float( weight[ i ] ) : 1.0f;

            float g = dy_val * w_val;
            sum_gx += g * x_val;

            if ( dweight )
                atomicAdd( &dweight[ i ], __float2bfloat16( dy_val * (x_val * rstd_val) ) );

            if ( dbias )
                atomicAdd( &dbias[ i ], __float2bfloat16( dy_val ) );
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
            float w_val = weight ? __bfloat162float( weight[ i ] ) : 1.0f;

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
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        rmsnorm_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (
            Y, rstd, X, weight, bias, num_slices, norm_dim, inner_size, epsilon);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_rmsnorm_backward_bf16(
        __nv_bfloat16* dX, __nv_bfloat16* dweight, __nv_bfloat16* dbias,
        const __nv_bfloat16* dY, const __nv_bfloat16* X, const __nv_bfloat16* weight,
        const __nv_bfloat16* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        rmsnorm_backward_bf16_kernel <<< grid_size, block_size, 0, stream >>> (
            dX, dweight, dbias, dY, X, weight, rstd, num_slices, norm_dim, inner_size);

        cudaCheck( cudaGetLastError() );
    }
}