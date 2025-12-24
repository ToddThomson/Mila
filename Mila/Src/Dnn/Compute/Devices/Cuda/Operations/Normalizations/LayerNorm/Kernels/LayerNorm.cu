//#include <cassert>
//#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
#include "CudaUtils.h"

// TJT: The cooperative groups headers above cause issues with our build system currently,
// so the code that depends on them has been commented out for now.

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    //namespace cg = cooperative_groups;

    //__global__ void layernorm_forward_fp32_kernel( float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
    //    const float* __restrict__ inp, const float* __restrict__ weight,
    //    const float* __restrict__ bias, int N, int C ) {
    //    int lane_id = threadIdx.x % WARP_SIZE;
    //    int warp_id = threadIdx.x / WARP_SIZE;
    //    int num_warps = blockDim.x / WARP_SIZE;

    //    int idx = blockIdx.x * num_warps + warp_id;
    //    if ( idx >= N ) {
    //        return;
    //    } // guard

    //    // the row of input that this group of threads is responsible for
    //    const float* x = inp + idx * C;

    //    // mean
    //    float sum = 0.0f;
    //    for ( int i = lane_id; i < C; i += WARP_SIZE ) {
    //        sum += (float)x[ i ];
    //    }
    //    sum = warpReduceSum( sum );
    //    float m = sum / C;
    //    if ( lane_id == 0 && mean != nullptr ) {
    //        __stcs( mean + idx, m );
    //    }

    //    // rstd
    //    sum = 0.0f;
    //    for ( int i = lane_id; i < C; i += WARP_SIZE ) {
    //        float diff = (float)x[ i ] - m;
    //        sum += diff * diff;
    //    }
    //    sum = warpReduceSum( sum );
    //    float s = rsqrtf( sum / C + 1e-5f );
    //    if ( lane_id == 0 && rstd != nullptr ) {
    //        __stcs( rstd + idx, s );
    //    }

    //    // final normalization and scaling by weight/bias
    //    float* o = out + idx * C;
    //    for ( int c = lane_id; c < C; c += WARP_SIZE ) {
    //        // load and store using the .cs "streaming" hint to the compiler,
    //        // indicating that this data will not be reused soon, and can be streamed through the caches
    //        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
    //        float n = s * ((float)__ldcs( x + c ) - m);
    //        __stcs( o + c, (float)(n * (float)weight[ c ] + (float)bias[ c ]) );
    //    }
    //}

    //__global__ void layernorm_forward_fp32_kernel( float* __restrict__ Y, float* __restrict__ mean, float* __restrict__ rstd,
    //    const float* __restrict__ X, const float* __restrict__ weight,  const float* __restrict__ bias, int N, int C, float epsilon ) 
    //{
    //    cg::thread_block block = cg::this_thread_block();
    //    cg::thread_block_tile<32> warp = cg::tiled_partition<32>( block );
    //    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    //    
    //    if ( idx >= N ) {
    //        return;
    //    }

    //    // the row of input that this group of threads is responsible for
    //    const float* x = X + idx * C;

    //    // mean
    //    float sum = 0.0f;
    //    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
    //        sum += x[ i ];
    //    }
    //    
    //    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    //    float m = sum / C;
    //    
    //    if ( warp.thread_rank() == 0 && mean != nullptr ) {
    //        __stcs( mean + idx, m );
    //    }

    //    // rstd
    //    sum = 0.0f;
    //    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
    //        float diff = x[ i ] - m;
    //        sum += diff * diff;
    //    }
    //    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    //    float s = rsqrtf( sum / C + epsilon );
    //    if ( warp.thread_rank() == 0 && rstd != nullptr ) {
    //        __stcs( rstd + idx, s );
    //    }

    //    // final normalization and scaling by weight/bias
    //    float* y = Y + idx * C;
    //    for ( int c = warp.thread_rank(); c < C; c += warp.size() ) {
    //        // load and store using the .cs "streaming" hint to the compiler,
    //        // indicating that this data will not be reused soon, and can be streamed through the caches
    //        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
    //        float n = s * (__ldcs( x + c ) - m);
    //        float result = n * weight[ c ];
    //        if ( bias != nullptr ) {
				//result += bias[ c ];
    //        }
    //        __stcs( y + c, result );
    //    }
    //}

    //__global__ void layernorm_forward_fp16_kernel( half* __restrict__ Y, half* __restrict__ mean, half* __restrict__ rstd,
    //    const half* __restrict__ X, const half* __restrict__ weight, const half* __restrict__ bias, int N, int C, float epsilon ) 
    //{
    //    cg::thread_block block = cg::this_thread_block();
    //    cg::thread_block_tile<32> warp = cg::tiled_partition<32>( block );
    //    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    //    
    //    if ( idx >= N ) {
    //        return;
    //    }

    //    // The row of input that this group of threads is responsible for
    //    const half* x = X + idx * C;

    //    // 
    //    // Mean calculation - Accumulate in float for better precision
    //    float sum = 0.0f;
    //    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
    //        sum += __half2float( x[ i ] );
    //    }
    //    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    //    float m = sum / C;
    //    if ( warp.thread_rank() == 0 && mean != nullptr ) {
    //        __stcs( mean + idx, __float2half( m ) );
    //    }

    //    // rstd calculation
    //    sum = 0.0f;
    //    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
    //        float val = __half2float( x[ i ] );
    //        float diff = val - m;
    //        sum += diff * diff;
    //    }
    //    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    //    float s = rsqrtf( sum / C + epsilon );
    //    if ( warp.thread_rank() == 0 && rstd != nullptr ) {
    //        __stcs( rstd + idx, __float2half( s ) );
    //    }

    //    // Final normalization and scaling by weight/bias
    //    half* y = Y + idx * C;
    //    for ( int c = warp.thread_rank(); c < C; c += warp.size() ) {
    //        // Load and store using streaming hints for better cache utilization
    //        float xval = __half2float( __ldcs( x + c ) );
    //        float normalized = s * (xval - m);
    //        float wval = __half2float( weight[ c ] );
    //        float result = normalized * wval;
    //        if ( bias != nullptr ) {
    //            float bval = __half2float( bias[ c ] );
    //            result += bval;
    //        }
    //        __stcs( y + c, __float2half( result ) );
    //    }
    //}

    void cuda_layernorm_forward_fp32(
        float* Y, float* mean, float* rstd,
        const float* X, const float* weight, const float* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream ) {
        const int block_size = 512;
        const int N = B * T;
        const int grid_size = ceil_div( N * 32, block_size );

        /*layernorm_forward_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            Y, mean, rstd, X, weight, bias, N, C, epsilon);

        cudaCheck( cudaGetLastError() );*/
    }

    void cuda_layernorm_forward_fp16(
        half* Y, half* mean, half* rstd,
        const half* X, const half* weight, const half* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream ) {
        const int block_size = 512;
        const int N = B * T;
        const int grid_size = ceil_div( N * 32, block_size );

        //layernorm_forward_fp16_kernel <<<grid_size, block_size, 0, stream >>> (
        //    Y, mean, rstd, X, weight, bias, N, C, epsilon);

        //cudaCheck( cudaGetLastError() );
    }
}