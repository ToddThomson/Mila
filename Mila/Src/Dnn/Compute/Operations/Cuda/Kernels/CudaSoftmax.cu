#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cuda.Utils.h"

__global__ void softmax_forward_kernel1( float* out, const float* inp, int N, int C ) {
    // input is (N, C)
    // output is (N, C), each row of input will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N ) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for ( int j = 0; j < C; j++ ) {
            if ( inp_row[ j ] > maxval ) {
                maxval = inp_row[ j ];
            }
        }
        double sum = 0.0;
        for ( int j = 0; j < C; j++ ) {
            out_row[ j ] = expf( inp_row[ j ] - maxval );
            sum += out_row[ j ];
        }
        for ( int j = 0; j < C; j++ ) {
            out_row[ j ] /= (float)sum;
        }
    }
}
void cuda_softmax_forward( float* output, const float* input, int N, int C ) {
    const int block_size = 512;
    const int grid_size = ceil_div( N, block_size );
    
    softmax_forward_kernel1 <<<grid_size, block_size>>> (output, input, N, C);
    
    cudaCheck( cudaGetLastError() );
}