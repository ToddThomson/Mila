#define _USE_MATH_DEFINES
#include <math.h>
#include "Cuda.Utils.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_forward_kernel( float* out, const float* inp, int N ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N ) {
        float xi = inp[ i ];
        float cube = 0.044715f * xi * xi * xi;
        out[ i ] = 0.5f * xi * (1.0f + tanhf( GELU_SCALING_FACTOR * (xi + cube) ));
    }
}

__global__ void gelu_backward_kernel( float* dinp, const float* inp, const float* dout, const int N ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N ) {
        float x = inp[ i ];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf( tanh_arg );
        float coshf_out = coshf( tanh_arg );
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[ i ] = local_grad * dout[ i ];
    }
}

void cuda_gelu_forward( float* out, const float* inp, int N, cudaStream_t stream ) {
    const int block_size = 128;
    const int grid_size = ceil_div( N, block_size );
    
    gelu_forward_kernel <<<grid_size, block_size>>> (out, inp, N);
    
    cudaCheck( cudaGetLastError() );
}

void cuda_gelu_backward( float* dinp, const float* inp, const float* dout, const int N ) {
    const int block_size = 128;
    const int grid_size = ceil_div( N, block_size );
    gelu_backward_kernel << <grid_size, block_size >> > (dinp, inp, dout, N);
    cudaCheck( cudaGetLastError() );
}
