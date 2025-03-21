#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include "Cuda.Utils.h"

__device__ inline float4 add_float4( const float4& a, const float4& b ) {
    return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
__global__ void encoder_forward_kernel3( float4* out,
    const int* inp, const float4* wte, const float4* wpe,
    int B, int T, int C ) {
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if ( idx < N ) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[ b * T + t ];
        out[ b * T * C4 + t * C4 + c4 ] = add_float4( wte[ ix * C4 + c4 ], wpe[ t * C4 + c4 ] );
    }
}

void cuda_encoder_forward( 
    float* out,
    const int* inp,
    const float* wte, const float* wpe,
    int B, int T, int C ) {
    assert( C % 4 == 0 );
    const int block_size = 512;
    const int N = B * T * C;
    const int grid_size = ceil_div( N / 4, block_size );
    
    encoder_forward_kernel3 << <grid_size, block_size >> > ((float4*)out, inp, (float4*)wte, (float4*)wpe, B, T, C);
    
    cudaCheck( cudaGetLastError() );
}
