#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cuda.Utils.h"

__global__ void residual_forward_kernel( float* out, const float* input_1, const float* input_2, int N ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < N ) {
        out[ idx ] = __ldcs( &input_1[ idx ] ) + __ldcs( &input_2[ idx ] );
    }
}

void cuda_residual_forward( float* out, const float* inp1, const float* inp2, int N, cudaStream_t stream ) {
    const int block_size = 256;
    const int grid_size = ceil_div( N, block_size );

    residual_forward_kernel <<<grid_size, block_size >>> (out, inp1, inp2, N);

    cudaCheck( cudaGetLastError() );
}