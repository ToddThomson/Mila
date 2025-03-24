#include <cassert>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cuda.Utils.h"

__global__ void residual_forward_kernel(
    float* out, 
    const float* inp1, const float* inp2, 
    int N ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < N ) {
        out[ idx ] = __ldcs( &inp1[ idx ] ) + __ldcs( &inp2[ idx ] );
    }
}

void cuda_residual_forward(
    float* out, 
    const float* inp1, const float* inp2, 
    int N ) {
    const int block_size = 256;
    const int grid_size = ceil_div( N, block_size );
    
    residual_forward_kernel << <grid_size, block_size >> > (out, inp1, inp2, N);
    
    cudaCheck( cudaGetLastError() );
}