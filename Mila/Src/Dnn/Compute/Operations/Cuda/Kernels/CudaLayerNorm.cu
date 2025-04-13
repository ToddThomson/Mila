#include <cassert>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Cuda.Utils.h"

namespace cg = cooperative_groups;

__global__ void layernorm_forward_kernel3( float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
    const float* __restrict__ inp, const float* __restrict__ weight,
    const float* __restrict__ bias, int N, int C ) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>( block );
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if ( idx >= N ) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
        sum += x[ i ];
    }
    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    float m = sum / C;
    if ( warp.thread_rank() == 0 && mean != nullptr ) {
        __stcs( mean + idx, m );
    }

    // rstd
    sum = 0.0f;
    for ( int i = warp.thread_rank(); i < C; i += warp.size() ) {
        float diff = x[ i ] - m;
        sum += diff * diff;
    }
    sum = cg::reduce( warp, sum, cg::plus<float>{} );
    float s = rsqrtf( sum / C + 1e-5f );
    if ( warp.thread_rank() == 0 && rstd != nullptr ) {
        __stcs( rstd + idx, s );
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for ( int c = warp.thread_rank(); c < C; c += warp.size() ) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs( x + c ) - m);
        __stcs( o + c, n * weight[ c ] + bias[ c ] );
    }
}

void cuda_layernorm_forward( 
    float* out, 
    float* mean, float* rstd,
    const float* inp, 
    const float* weight, const float* bias,
    int B, int T, int C,
    cudaStream_t stream ) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = ceil_div( N * 32, block_size );

    layernorm_forward_kernel3<<<grid_size, block_size >>>(out, mean, rstd, inp, weight, bias, N, C);
    
    cudaCheck( cudaGetLastError() );
}