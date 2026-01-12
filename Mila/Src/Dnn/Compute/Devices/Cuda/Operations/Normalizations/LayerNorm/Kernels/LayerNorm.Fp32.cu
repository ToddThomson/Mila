#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "CudaUtils.h"
#include "LayerNorm.cuh"

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    // REVIEW: Consider templating on training mode and bias presence for potential optimizations.

    /**
     * @brief FP32 layer normalization forward kernel with per-warp Welford and correct striding.
     *
     * This implementation uses the same memory layout as the CPU implementation:
     * flattened index for a slice = (outer_index * norm_dim * inner_size) + inner_index,
     * and elements along the normalized dimension are separated by `inner_size`.
     */
    __global__ void layernorm_forward_fp32_kernel(
        float* __restrict__ out,
        float* __restrict__ mean,
        float* __restrict__ rstd,
        const float* __restrict__ inp,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        int num_slices, int norm_dim, int inner_size, float epsilon )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
        {
            return;
        }

        // Decompose slice index into outer and inner to compute correct base pointer with stride
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const float* x = inp + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;
        float* o = out + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;

        float m = 0.0f;
        float m2 = 0.0f;
        int count = 0;

        // Use per-lane accumulation over normalized dimension with stride = inner_size
        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            float val = x[ static_cast<size_t>(i) * static_cast<size_t>(inner_size) ];
            count++;
            float delta = val - m;
            m += delta / count;
            float delta2 = val - m;
            m2 += delta * delta2;
        }

        // Warp-level reduction for mean/variance accumulators
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            float other_m = __shfl_down_sync( 0xffffffff, m, offset );
            float other_m2 = __shfl_down_sync( 0xffffffff, m2, offset );
            int other_count = __shfl_down_sync( 0xffffffff, count, offset );

            if ( lane_id < offset )
            {
                int total = count + other_count;
                if ( total != 0 )
                {
                    float delta = other_m - m;
                    // Combine means and M2 according to parallel Welford
                    m = ( count * m + other_count * other_m ) / total;
                    m2 = m2 + other_m2 + delta * delta * (static_cast<float>(count) * static_cast<float>(other_count)) / static_cast<float>(total);
                    count = total;
                }
            }
        }

        // Broadcast final mean/M2 to lane 0
        m = __shfl_sync( 0xffffffff, m, 0 );
        m2 = __shfl_sync( 0xffffffff, m2, 0 );

        // Use provided epsilon for numerical stability
        float s = rsqrtf( m2 / static_cast<float>( norm_dim ) + epsilon );

#ifndef NDEBUG
        // Lightweight kernel-level sanity check (debug-only).
        // Runs only on lane 0 of each warp and aborts with assert if mean/rstd are NaN/Inf or exceed threshold.
        // This is cheap compared to D2H copies and helps catch upstream kernel/precision errors early.
        constexpr float kLayerNormKernelLimit = 100.0f;
        if ( lane_id == 0 )
        {
            if ( !isfinite( m ) || !isfinite( s ) || fabsf( m ) > kLayerNormKernelLimit || fabsf( s ) > kLayerNormKernelLimit )
            {
                // Print minimal diagnostic to aid locating the offending slice (cheap relative to full copy).
                printf( "LayerNorm FWD anomaly: idx=%d outer=%d inner=%d mean=%f rstd=%f norm_dim=%d inner_size=%d\n",
                        idx, outer_idx, inner_idx, m, s, norm_dim, inner_size );
                assert( false );
            }
        }
#endif

        if ( lane_id == 0 )
        {
            if ( mean != nullptr )
            {
                mean[ idx ] = m;
            }

            if ( rstd != nullptr )
            {
                rstd[ idx ] = s;
            }
        }

        // Write normalized output (strided)
        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>(i) * static_cast<size_t>(inner_size);
            float val = x[ offset ];
            float w = weight ? weight[i] : 1.0f;
            float b = bias ? bias[ i ] : 0.0f;
            float res = s * ( val - m ) * w + b;

        #ifndef NDEBUG
            constexpr float kLayerNormOutputAbsLimit = 50.0f;  // Normalized output should be small
            if ( !isfinite( res ) || fabsf( res ) > kLayerNormOutputAbsLimit )
            {
                printf(
                    "LayerNorm OUTPUT anomaly: idx=%d i=%d val=%f mean=%f rstd=%f weight=%f bias=%f output=%f\n",
                    idx, i, val, m, s, w, b, res
                );
                assert( false );
            }
        #endif

            o[ offset ] = res;
        }
    }

    /**
     * @brief FP32 layer normalization backward kernel (strided accesses).
     *
     * Mirrors CPU backward logic with atomicAdds for parameter gradients.
     */
    __global__ void layernorm_backward_fp32_kernel(
        float* __restrict__ dinp,
        float* __restrict__ dweight,
        float* __restrict__ dbias,
        const float* __restrict__ dout,
        const float* __restrict__ inp,
        const float* __restrict__ weight,
        const float* __restrict__ mean,
        const float* __restrict__ rstd,
        int num_slices, int norm_dim, int inner_size )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
        {
            return;
        }

        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const float* x = inp + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;
        const float* dy = dout + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;
        float* dx = dinp + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;

        float m = mean[ idx ];
        float s = rstd[ idx ];

        float sum_dy_w = 0.0f;
        float sum_dy_w_xhat = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>(i) * static_cast<size_t>(inner_size);
            float x_val = x[ offset ];
            float dy_val = dy[ offset ];
            float w_val = weight ? weight[i] : 1.0f;

            float xhat = ( x_val - m ) * s;
            sum_dy_w += dy_val * w_val;
            sum_dy_w_xhat += dy_val * w_val * xhat;

            if ( dweight )
            {
                atomicAdd( &dweight[ i ], dy_val * xhat );
            }

            if ( dbias )
            {
                atomicAdd( &dbias[ i ], dy_val );
            }
        }

        // Reduce across warp
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            sum_dy_w += __shfl_down_sync( 0xffffffff, sum_dy_w, offset );
            sum_dy_w_xhat += __shfl_down_sync( 0xffffffff, sum_dy_w_xhat, offset );
        }

        sum_dy_w = __shfl_sync( 0xffffffff, sum_dy_w, 0 );
        sum_dy_w_xhat = __shfl_sync( 0xffffffff, sum_dy_w_xhat, 0 );

        float norm_factor = 1.0f / static_cast<float>( norm_dim );

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>(i) * static_cast<size_t>(inner_size);
            float x_val = x[ offset ];
            float dy_val = dy[ offset ];
            float w_val = weight ? weight[i] : 1.0f;

            float xhat = ( x_val - m ) * s;
            float dxhat = dy_val * w_val;

            float out_dx = ( dxhat - norm_factor * sum_dy_w - norm_factor * xhat * sum_dy_w_xhat ) * s;

            // accumulate into dinp (atomic to be conservative across warps/blocks)
            // Use atomicAdd on floats; it's supported on modern GPUs
            atomicAdd( &dx[ offset ], out_dx );
        }
    }

    // ========================================================================
    // Host launchers
    // ========================================================================

    void cuda_layernorm_forward_fp32(
        float* Y, float* mean, float* rstd,
        const float* X, const float* weight, const float* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int num_slices = outer_size * inner_size;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_forward_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            Y, mean, rstd, X, weight, bias, num_slices, norm_dim, inner_size, epsilon );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_layernorm_backward_fp32(
        float* dX, float* dweight, float* dbias,
        const float* dY, const float* X, const float* weight,
        const float* mean, const float* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int num_slices = outer_size * inner_size;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_backward_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            dX, dweight, dbias, dY, X, weight, mean, rstd, num_slices, norm_dim, inner_size );

        cudaCheck( cudaGetLastError() );
    }
}