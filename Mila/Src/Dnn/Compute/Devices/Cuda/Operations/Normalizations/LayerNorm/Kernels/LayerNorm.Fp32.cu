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
     * @brief FP32 layer normalization forward kernel with float4 vectorization.
     *
     * Each warp processes one normalized slice using Welford's online algorithm
     * for numerically stable mean/variance computation.
     *
     * @param out Normalized output tensor
     * @param mean Per-slice mean statistics
     * @param rstd Per-slice reciprocal standard deviation
     * @param inp Input tensor
     * @param weight Scaling parameters
     * @param bias Shift parameters
     * @param num_slices Total number of slices to normalize (outer_size * inner_size)
     * @param norm_dim Size of normalized dimension
     */
    __global__ void layernorm_forward_fp32_kernel(
        float* __restrict__ out,
        float* __restrict__ mean,
        float* __restrict__ rstd,
        const float* __restrict__ inp,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        int num_slices, int norm_dim )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
        {
            return;
        }

        const float* x = inp + idx * norm_dim;
        const int vec_size = norm_dim / 4;

        float m = 0.0f;
        float m2 = 0.0f;
        int count = 0;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 vals = reinterpret_cast<const float4*>( x )[ i ];

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                count++;
                float val = ( &vals.x )[ j ];
                float delta = val - m;
                m += delta / count;
                float delta2 = val - m;
                m2 += delta * delta2;
            }
        }

        for ( int i = vec_size * 4 + lane_id; i < norm_dim; i += WARP_SIZE )
        {
            count++;
            float val = x[ i ];
            float delta = val - m;
            m += delta / count;
            float delta2 = val - m;
            m2 += delta * delta2;
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            float other_m = __shfl_down_sync( 0xffffffff, m, offset );
            float other_m2 = __shfl_down_sync( 0xffffffff, m2, offset );
            int other_count = __shfl_down_sync( 0xffffffff, count, offset );

            if ( lane_id < offset )
            {
                int total = count + other_count;
                float delta = other_m - m;
                m = ( count * m + other_count * other_m ) / total;
                m2 = m2 + other_m2 + delta * delta * count * other_count / total;
                count = total;
            }
        }

        m = __shfl_sync( 0xffffffff, m, 0 );
        m2 = __shfl_sync( 0xffffffff, m2, 0 );

        float s = rsqrtf( m2 / norm_dim + 1e-5f );

        if ( lane_id == 0 )
        {
            if ( mean != nullptr )
            {
                __stcs( mean + idx, m );
            }

            if ( rstd != nullptr )
            {
                __stcs( rstd + idx, s );
            }
        }

        float* o = out + idx * norm_dim;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 vals = reinterpret_cast<const float4*>( x )[ i ];
            float4 w = reinterpret_cast<const float4*>( weight )[ i ];

            float4 result;
            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                (&result.x)[ j ] = s * ((&vals.x)[ j ] - m) * (&w.x)[ j ];
            }

            if ( bias != nullptr )
            {
                float4 b = reinterpret_cast<const float4*>( bias )[ i ];
                #pragma unroll
                for ( int j = 0; j < 4; j++ )
                {
                    (&result.x)[ j ] += (&b.x)[ j ];
                }
            }

            reinterpret_cast<float4*>( o )[ i ] = result;
        }

        for ( int c = vec_size * 4 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float n = s * (__ldcs( x + c ) - m) * weight[ c ];

            if ( bias != nullptr )
            {
                n += bias[ c ];
            }

            __stcs( o + c, n );
        }
    }

    /**
     * @brief FP32 layer normalization backward kernel.
     *
     * Computes input gradient and accumulates parameter gradients using
     * cached forward-pass statistics (mean/rstd).
     *
     * @param dinp Input gradient
     * @param dweight Weight gradient accumulator (uses atomics across slices)
     * @param dbias Bias gradient accumulator (uses atomics across slices)
     * @param dout Output gradient
     * @param inp Original forward pass input
     * @param weight Forward pass weight parameters
     * @param mean Forward pass mean statistics
     * @param rstd Forward pass reciprocal standard deviation
     * @param num_slices Total number of slices (outer_size * inner_size)
     * @param norm_dim Size of normalized dimension
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
        int num_slices, int norm_dim )
    {
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        int idx = blockIdx.x * num_warps + warp_id;

        if ( idx >= num_slices )
        {
            return;
        }

        const float* x = inp + idx * norm_dim;
        const float* dy = dout + idx * norm_dim;
        float* dx = dinp + idx * norm_dim;

        float m = mean[ idx ];
        float s = rstd[ idx ];

        const int vec_size = norm_dim / 4;

        float sum_dy_w = 0.0f;
        float sum_dy_w_xhat = 0.0f;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 x_vec = reinterpret_cast<const float4*>( x )[ i ];
            float4 dy_vec = reinterpret_cast<const float4*>( dy )[ i ];
            float4 w_vec = reinterpret_cast<const float4*>( weight )[ i ];

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                float x_val = ( &x_vec.x )[ j ];
                float dy_val = ( &dy_vec.x )[ j ];
                float w_val = ( &w_vec.x )[ j ];

                float xhat = ( x_val - m ) * s;
                sum_dy_w += dy_val * w_val;
                sum_dy_w_xhat += dy_val * w_val * xhat;

                int c = i * 4 + j;
                atomicAdd( &dweight[ c ], dy_val * xhat );

                if ( dbias != nullptr )
                {
                    atomicAdd( &dbias[ c ], dy_val );
                }
            }
        }

        for ( int c = vec_size * 4 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float x_val = x[ c ];
            float dy_val = dy[ c ];
            float w_val = weight[ c ];

            float xhat = ( x_val - m ) * s;
            sum_dy_w += dy_val * w_val;
            sum_dy_w_xhat += dy_val * w_val * xhat;

            atomicAdd( &dweight[ c ], dy_val * xhat );

            if ( dbias != nullptr )
            {
                atomicAdd( &dbias[ c ], dy_val );
            }
        }

        for ( int offset = WARP_SIZE / 2; offset > 0; offset /= 2 )
        {
            sum_dy_w += __shfl_down_sync( 0xffffffff, sum_dy_w, offset );
            sum_dy_w_xhat += __shfl_down_sync( 0xffffffff, sum_dy_w_xhat, offset );
        }

        sum_dy_w = __shfl_sync( 0xffffffff, sum_dy_w, 0 );
        sum_dy_w_xhat = __shfl_sync( 0xffffffff, sum_dy_w_xhat, 0 );

        float norm_factor = 1.0f / norm_dim;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 x_vec = reinterpret_cast<const float4*>( x )[ i ];
            float4 dy_vec = reinterpret_cast<const float4*>( dy )[ i ];
            float4 w_vec = reinterpret_cast<const float4*>( weight )[ i ];

            float4 dx_vec;

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                float x_val = ( &x_vec.x )[ j ];
                float dy_val = ( &dy_vec.x )[ j ];
                float w_val = ( &w_vec.x )[ j ];

                float xhat = ( x_val - m ) * s;
                float dxhat = dy_val * w_val;

                ( &dx_vec.x )[ j ] = ( dxhat - norm_factor * sum_dy_w - norm_factor * xhat * sum_dy_w_xhat ) * s;
            }

            reinterpret_cast<float4*>( dx )[ i ] = dx_vec;
        }

        for ( int c = vec_size * 4 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float x_val = x[ c ];
            float dy_val = dy[ c ];
            float w_val = weight[ c ];

            float xhat = ( x_val - m ) * s;
            float dxhat = dy_val * w_val;

            dx[ c ] = ( dxhat - norm_factor * sum_dy_w - norm_factor * xhat * sum_dy_w_xhat ) * s;
        }
    }

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
            Y, mean, rstd, X, weight, bias, num_slices, norm_dim );

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
            dX, dweight, dbias, dY, X, weight, mean, rstd, num_slices, norm_dim );

        cudaCheck( cudaGetLastError() );
    }
}