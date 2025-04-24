#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA kernel for softmax forward pass with FP32 precision
     *
     * Computes softmax function on each row of the input tensor.
     *
     * @param out Output tensor (N, C)
     * @param inp Input tensor (N, C)
     * @param N Number of rows
     * @param C Number of columns (features)
     */
    __global__ void softmax_forward_fp32_kernel( float* out, const float* inp, int N, int C ) {
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

    /**
     * @brief CUDA kernel for softmax forward pass with FP16 precision
     *
     * Computes softmax function on each row of the input tensor using half precision.
     *
     * @param out Output tensor (N, C) in half precision
     * @param inp Input tensor (N, C) in half precision
     * @param N Number of rows
     * @param C Number of columns (features)
     */
    __global__ void softmax_forward_fp16_kernel( half* out, const half* inp, int N, int C ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < N ) {
            const half* inp_row = inp + i * C;
            half* out_row = out + i * C;

            // Find max value for numerical stability
            float maxval = -INFINITY;
            for ( int j = 0; j < C; j++ ) {
                float val = __half2float( inp_row[ j ] );
                if ( val > maxval ) {
                    maxval = val;
                }
            }

            // Compute exp and sum
            float sum = 0.0f;
            for ( int j = 0; j < C; j++ ) {
                float val = expf( __half2float( inp_row[ j ] ) - maxval );
                out_row[ j ] = __float2half( val );
                sum += val;
            }

            // Normalize
            for ( int j = 0; j < C; j++ ) {
                out_row[ j ] = __float2half( __half2float( out_row[ j ] ) / sum );
            }
        }
    }

    /**
     * @brief Host function to launch softmax forward pass with full precision (FP32)
     *
     * Computes softmax function on each row of the input tensor.
     *
     * @param output Output tensor (N, C)
     * @param input Input tensor (N, C)
     * @param N Number of rows
     * @param C Number of columns (features)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_softmax_forward_fp32(
        float* output,
        const float* input,
        int N,
        int C,
        cudaStream_t stream ) {

        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (output, input, N, C);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host function to launch softmax forward pass with half precision (FP16)
     *
     * Computes softmax function on each row of the input tensor using FP16
     * for better performance on compatible hardware.
     *
     * @param output Output tensor (N, C) in half precision
     * @param input Input tensor (N, C) in half precision
     * @param N Number of rows
     * @param C Number of columns (features)
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_softmax_forward_fp16(
        half* output,
        const half* input,
        int N,
        int C,
        cudaStream_t stream ) {

        const int block_size = 512;
        const int grid_size = ceil_div( N, block_size );

        softmax_forward_fp16_kernel << <grid_size, block_size, 0, stream >> > (output, input, N, C);
        cudaCheck( cudaGetLastError() );
    }
}
