//#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "CudaUtils.h"
#include "LayerNorm.cuh"

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    /**
     * @brief FP16 layer normalization forward kernel with vectorized loads.
     *
     * Accumulates statistics in FP32 for numerical stability. Processes 8 half
     * values per iteration using float4 loads (16 bytes).
     *
     * @param out Normalized output (FP16)
     * @param mean Per-slice mean statistics (FP32)
     * @param rstd Per-slice reciprocal standard deviation (FP32)
     * @param inp Input tensor (FP16)
     * @param weight Scaling parameters (FP16)
     * @param bias Shift parameters (FP16)
     * @param num_slices Total number of slices to normalize (outer_size * inner_size)
     * @param norm_dim Size of normalized dimension
     */
    __global__ void layernorm_forward_fp16_kernel(
        half* __restrict__ out,
        float* __restrict__ mean,
        float* __restrict__ rstd,
        const half* __restrict__ inp,
        const half* __restrict__ weight,
        const half* __restrict__ bias,
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

        const half* x = inp + idx * norm_dim;
        const int vec_size = norm_dim / 8;

        float m = 0.0f;
        float m2 = 0.0f;
        int count = 0;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 raw = reinterpret_cast<const float4*>( x )[ i ];
            half2* h2_ptr = reinterpret_cast<half2*>( &raw );

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                half2 h2 = h2_ptr[ j ];
                float val1 = __half2float( h2.x );
                float val2 = __half2float( h2.y );

                count++;
                float delta = val1 - m;
                m += delta / count;
                float delta2 = val1 - m;
                m2 += delta * delta2;

                count++;
                delta = val2 - m;
                m += delta / count;
                delta2 = val2 - m;
                m2 += delta * delta2;
            }
        }

        for ( int i = vec_size * 8 + lane_id; i < norm_dim; i += WARP_SIZE )
        {
            count++;
            float val = __half2float( x[ i ] );
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
                mean[ idx ] = m;
            }

            if ( rstd != nullptr )
            {
                rstd[ idx ] = s;
            }
        }

        half* o = out + idx * norm_dim;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 raw = reinterpret_cast<const float4*>( x )[ i ];
            float4 w_raw = reinterpret_cast<const float4*>( weight )[ i ];
            float4 b_raw = reinterpret_cast<const float4*>( bias )[ i ];

            half2* h2_ptr = reinterpret_cast<half2*>( &raw );
            half2* w_h2_ptr = reinterpret_cast<half2*>( &w_raw );
            half2* b_h2_ptr = reinterpret_cast<half2*>( &b_raw );

            float4 result;
            half2* result_h2 = reinterpret_cast<half2*>( &result );

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                half2 h2 = h2_ptr[ j ];
                half2 w_h2 = w_h2_ptr[ j ];
                half2 b_h2 = b_h2_ptr[ j ];

                float val1 = __half2float( h2.x );
                float val2 = __half2float( h2.y );
                float w1 = __half2float( w_h2.x );
                float w2 = __half2float( w_h2.y );
                float b1 = __half2float( b_h2.x );
                float b2 = __half2float( b_h2.y );

                float norm1 = s * ( val1 - m ) * w1 + b1;
                float norm2 = s * ( val2 - m ) * w2 + b2;

                result_h2[ j ] = __halves2half2( __float2half( norm1 ), __float2half( norm2 ) );
            }

            reinterpret_cast<float4*>( o )[ i ] = result;
        }

        for ( int c = vec_size * 8 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float val = __half2float( x[ c ] );
            float w = __half2float( weight[ c ] );
            float b = __half2float( bias[ c ] );
            float normalized = s * ( val - m ) * w + b;
            o[ c ] = __float2half( normalized );
        }
    }

    /**
     * @brief FP16 layer normalization backward kernel.
     *
     * Accumulates parameter gradients in FP32 for numerical stability.
     *
     * @param dinp Input gradient (FP16)
     * @param dweight Weight gradient accumulator (FP32, uses atomics)
     * @param dbias Bias gradient accumulator (FP32, uses atomics)
     * @param dout Output gradient (FP16)
     * @param inp Original forward pass input (FP16)
     * @param weight Forward pass weight parameters (FP16)
     * @param mean Forward pass mean statistics (FP32)
     * @param rstd Forward pass reciprocal standard deviation (FP32)
     * @param num_slices Total number of slices (outer_size * inner_size)
     * @param norm_dim Size of normalized dimension
     */
    __global__ void layernorm_backward_fp16_kernel(
        half* __restrict__ dinp,
        float* __restrict__ dweight,
        float* __restrict__ dbias,
        const half* __restrict__ dout,
        const half* __restrict__ inp,
        const half* __restrict__ weight,
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

        const half* x = inp + idx * norm_dim;
        const half* dy = dout + idx * norm_dim;
        half* dx = dinp + idx * norm_dim;

        float m = mean[ idx ];
        float s = rstd[ idx ];

        const int vec_size = norm_dim / 8;

        float sum_dy_w = 0.0f;
        float sum_dy_w_xhat = 0.0f;

        for ( int i = lane_id; i < vec_size; i += WARP_SIZE )
        {
            float4 x_raw = reinterpret_cast<const float4*>( x )[ i ];
            float4 dy_raw = reinterpret_cast<const float4*>( dy )[ i ];
            float4 w_raw = reinterpret_cast<const float4*>( weight )[ i ];

            half2* x_h2 = reinterpret_cast<half2*>( &x_raw );
            half2* dy_h2 = reinterpret_cast<half2*>( &dy_raw );
            half2* w_h2 = reinterpret_cast<half2*>( &w_raw );

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                float x_val1 = __half2float( x_h2[ j ].x );
                float x_val2 = __half2float( x_h2[ j ].y );
                float dy_val1 = __half2float( dy_h2[ j ].x );
                float dy_val2 = __half2float( dy_h2[ j ].y );
                float w_val1 = __half2float( w_h2[ j ].x );
                float w_val2 = __half2float( w_h2[ j ].y );

                float xhat1 = ( x_val1 - m ) * s;
                float xhat2 = ( x_val2 - m ) * s;

                sum_dy_w += dy_val1 * w_val1 + dy_val2 * w_val2;
                sum_dy_w_xhat += dy_val1 * w_val1 * xhat1 + dy_val2 * w_val2 * xhat2;

                int c1 = i * 8 + j * 2;
                int c2 = c1 + 1;
                atomicAdd( &dweight[ c1 ], dy_val1 * xhat1 );
                atomicAdd( &dweight[ c2 ], dy_val2 * xhat2 );
                atomicAdd( &dbias[ c1 ], dy_val1 );
                atomicAdd( &dbias[ c2 ], dy_val2 );
            }
        }

        for ( int c = vec_size * 8 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float x_val = __half2float( x[ c ] );
            float dy_val = __half2float( dy[ c ] );
            float w_val = __half2float( weight[ c ] );

            float xhat = ( x_val - m ) * s;
            sum_dy_w += dy_val * w_val;
            sum_dy_w_xhat += dy_val * w_val * xhat;

            atomicAdd( &dweight[ c ], dy_val * xhat );
            atomicAdd( &dbias[ c ], dy_val );
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
            float4 x_raw = reinterpret_cast<const float4*>( x )[ i ];
            float4 dy_raw = reinterpret_cast<const float4*>( dy )[ i ];
            float4 w_raw = reinterpret_cast<const float4*>( weight )[ i ];

            half2* x_h2 = reinterpret_cast<half2*>( &x_raw );
            half2* dy_h2 = reinterpret_cast<half2*>( &dy_raw );
            half2* w_h2 = reinterpret_cast<half2*>( &w_raw );

            float4 dx_raw;
            half2* dx_h2 = reinterpret_cast<half2*>( &dx_raw );

            #pragma unroll
            for ( int j = 0; j < 4; j++ )
            {
                float x_val1 = __half2float( x_h2[ j ].x );
                float x_val2 = __half2float( x_h2[ j ].y );
                float dy_val1 = __half2float( dy_h2[ j ].x );
                float dy_val2 = __half2float( dy_h2[ j ].y );
                float w_val1 = __half2float( w_h2[ j ].x );
                float w_val2 = __half2float( w_h2[ j ].y );

                float xhat1 = ( x_val1 - m ) * s;
                float xhat2 = ( x_val2 - m ) * s;
                float dxhat1 = dy_val1 * w_val1;
                float dxhat2 = dy_val2 * w_val2;

                float dx_val1 = ( dxhat1 - norm_factor * sum_dy_w - norm_factor * xhat1 * sum_dy_w_xhat ) * s;
                float dx_val2 = ( dxhat2 - norm_factor * sum_dy_w - norm_factor * xhat2 * sum_dy_w_xhat ) * s;

                dx_h2[ j ] = __halves2half2( __float2half( dx_val1 ), __float2half( dx_val2 ) );
            }

            reinterpret_cast<float4*>( dx )[ i ] = dx_raw;
        }

        for ( int c = vec_size * 8 + lane_id; c < norm_dim; c += WARP_SIZE )
        {
            float x_val = __half2float( x[ c ] );
            float dy_val = __half2float( dy[ c ] );
            float w_val = __half2float( weight[ c ] );

            float xhat = ( x_val - m ) * s;
            float dxhat = dy_val * w_val;

            float dx_val = ( dxhat - norm_factor * sum_dy_w - norm_factor * xhat * sum_dy_w_xhat ) * s;
            dx[ c ] = __float2half( dx_val );
        }
    }

    void cuda_layernorm_forward_fp16(
        half* Y, half* mean, half* rstd,
        const half* X, const half* weight, const half* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int num_slices = outer_size * inner_size;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_forward_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            Y, reinterpret_cast<float*>( mean ), reinterpret_cast<float*>( rstd ),
            X, weight, bias, num_slices, norm_dim );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_layernorm_backward_fp16(
        half* dX, half* dweight, half* dbias,
        const half* dY, const half* X, const half* weight,
        const half* mean, const half* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int num_slices = outer_size * inner_size;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_backward_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            dX, reinterpret_cast<float*>( dweight ), reinterpret_cast<float*>( dbias ),
            dY, X, weight,
            reinterpret_cast<const float*>( mean ), reinterpret_cast<const float*>( rstd ),
            num_slices, norm_dim );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_layernorm_forward_fp16_contig(
        half* Y, half* mean, half* rstd,
        const half* X, const half* weight, const half* bias,
        int num_slices, int norm_dim,
        float epsilon,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (
            Y, reinterpret_cast<float*>(mean), reinterpret_cast<float*>(rstd),
            X, weight, bias, num_slices, norm_dim);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_layernorm_backward_fp16_contig(
        half* dX, half* dweight, half* dbias,
        const half* dY, const half* X, const half* weight,
        const half* mean, const half* rstd,
        int num_slices, int norm_dim,
        cudaStream_t stream )
    {
        const int block_size = 512;
        const int grid_size = ceil_div( num_slices * WARP_SIZE, block_size );

        layernorm_backward_fp16_kernel << <grid_size, block_size, 0, stream >> > (
            dX, reinterpret_cast<float*>(dweight), reinterpret_cast<float*>(dbias),
            dY, X, weight,
            reinterpret_cast<const float*>(mean), reinterpret_cast<const float*>(rstd),
            num_slices, norm_dim);

        cudaCheck( cudaGetLastError() );
    }
}