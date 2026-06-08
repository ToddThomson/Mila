/**
 * @file RmsNorm.Fp32.cu
 * @brief FP32 CUDA kernels and host launchers for RMS normalization.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "CudaUtils.h"
#include "RmsNorm.cuh"

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    // FP32 RMSNorm forward kernel.
    // Layout: [outer_size, norm_dim, inner_size] with strided access pattern.
    // Each warp processes one slice independently.
    // Writes per-slice reciprocal sqrt (rstd) to rstd array for backward pass.
    __global__ void rmsnorm_forward_fp32_kernel(
        float* __restrict__ out,
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

        // Decompose slice index into outer and inner for strided access
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const float* x = inp + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;
        float* o = out + static_cast<size_t>(outer_idx) * static_cast<size_t>(norm_dim) * static_cast<size_t>(inner_size) + inner_idx;

        // Compute mean of squares: m2 = sum_i x_i^2
        float m2 = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            float val = x[ static_cast<size_t>( i ) * static_cast<size_t>( inner_size ) ];
            m2 += val * val;
        }

        // Warp-level reduction
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            m2 += __shfl_down_sync( 0xffffffff, m2, offset );
        }

        // Broadcast final m2 to all lanes
        m2 = __shfl_sync( 0xffffffff, m2, 0 );

        // Compute reciprocal std: rstd = 1/sqrt(m2/N + eps)
        float inv_n = 1.0f / static_cast<float>(norm_dim);
        float rstd_val = rsqrtf( m2 * inv_n + epsilon );

    #ifndef NDEBUG
        constexpr float kRmsNormKernelLimit = 1e6f;
        if ( lane_id == 0 )
        {
            if ( !isfinite( rstd_val ) || fabsf( rstd_val ) > kRmsNormKernelLimit )
            {
                printf( "RMSNorm FWD anomaly: idx=%d outer=%d inner=%d m2=%f rstd=%f norm_dim=%d inner_size=%d\n",
                    idx, outer_idx, inner_idx, m2, rstd_val, norm_dim, inner_size );
                assert( false );
            }
        }
    #endif

        // Store rstd for backward pass
        if ( lane_id == 0 && rstd != nullptr )
        {
            rstd[ idx ] = rstd_val;
        }

        // Write output: y = (x * rstd) * weight + bias
        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float xv = x[ offset ];
            float w = weight ? weight[ i ] : 1.0f;
            float b = bias ? bias[ i ] : 0.0f;
            float xhat = xv * rstd_val;
            float res = xhat * w + b;

        #ifndef NDEBUG
            constexpr float kRmsNormOutputAbsLimit = 1e4f;
            if ( !isfinite( res ) || fabsf( res ) > kRmsNormOutputAbsLimit )
            {
                printf(
                    "RMSNorm OUTPUT anomaly: idx=%d i=%d val=%f m2=%f rstd=%f weight=%f bias=%f output=%f\n",
                    idx, i, xv, m2, rstd_val, w, b, res
                );
                assert( false );
            }
        #endif

            o[ offset ] = res;
        }
    }

    // FP32 RMSNorm backward kernel.
    // Computes dX and accumulates dweight/dbias via atomicAdds.
    // Each warp processes one slice independently.
    __global__ void rmsnorm_backward_fp32_kernel(
        float* __restrict__ dinp,
        float* __restrict__ dweight,
        float* __restrict__ dbias,
        const float* __restrict__ dout,
        const float* __restrict__ inp,
        const float* __restrict__ weight,
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

        float rstd_val = rstd[ idx ];
        float inv_n = 1.0f / static_cast<float>(norm_dim);

        // Compute sum_gx = sum_i (dy_i * weight_i * x_i)
        // Also accumulate parameter gradients
        float sum_gx = 0.0f;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float x_val = x[ offset ];
            float dy_val = dy[ offset ];
            float w_val = weight ? weight[ i ] : 1.0f;

            float g = dy_val * w_val;
            sum_gx += g * x_val;

            // Accumulate parameter gradients
            if ( dweight )
            {
                float xhat = x_val * rstd_val;
                atomicAdd( &dweight[ i ], dy_val * xhat );
            }

            if ( dbias )
            {
                atomicAdd( &dbias[ i ], dy_val );
            }
        }

        // Warp-level reduction for sum_gx
        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            sum_gx += __shfl_down_sync( 0xffffffff, sum_gx, offset );
        }

        // Broadcast final sum to all lanes
        sum_gx = __shfl_sync( 0xffffffff, sum_gx, 0 );

        // Compute input gradient: dx = rstd * g - x * rstd³ * (1/N * sum(g * x))
        float rstd3 = rstd_val * rstd_val * rstd_val;
        float correction = rstd3 * inv_n * sum_gx;

        for ( int i = lane_id; i < norm_dim; i += WARP_SIZE )
        {
            size_t offset = static_cast<size_t>( i ) * static_cast<size_t>( inner_size );
            float x_val = x[ offset ];
            float dy_val = dy[ offset ];
            float w_val = weight ? weight[ i ] : 1.0f;

            float g = dy_val * w_val;
            float out_dx = rstd_val * g - x_val * correction;

            dx[ offset ] = out_dx;
        }
    }

    // ========================================================================
    // Host launchers
    // ========================================================================

    void cuda_rmsnorm_forward_fp32(
        float* Y, float* rstd,
        const float* X, const float* weight, const float* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        rmsnorm_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            Y, rstd, X, weight, bias, num_slices, norm_dim, inner_size, epsilon);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_rmsnorm_backward_fp32(
        float* dX, float* dweight, float* dbias,
        const float* dY, const float* X, const float* weight,
        const float* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int warps_per_block = block_size / WARP_SIZE;
        const int num_slices = outer_size * inner_size;
        const int grid_size = (num_slices + warps_per_block - 1) / warps_per_block;

        // Note: dweight and dbias must be zeroed by caller before this call
        rmsnorm_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            dX, dweight, dbias, dY, X, weight, rstd, num_slices, norm_dim, inner_size);

        cudaCheck( cudaGetLastError() );
    }
}