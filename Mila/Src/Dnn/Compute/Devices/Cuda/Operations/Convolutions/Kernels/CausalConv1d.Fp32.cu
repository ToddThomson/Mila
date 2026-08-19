/**
 * @file CausalConv1d.Fp32.cu
 * @brief FP32 kernels and host launchers for the depthwise causal 1-D convolution.
 */

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Convolution
{
    // Upper bound on retained rows (K-1), so the state shift can stage in registers.
    constexpr int kMaxStateRows = 7;

    __global__ void causal_conv1d_forward_fp32_kernel(
        float* __restrict__       out,
        const float* __restrict__ x,
        const float* __restrict__ state,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        int B, int T, int C, int K )
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = B * T * C;

        if ( idx >= total )
            return;

        const int c = idx % C;
        const int t = (idx / C) % T;
        const int b = idx / (C * T);
        const int state_rows = K - 1;

        float accumulator = bias ? bias[ c ] : 0.0f;

        for ( int i = 0; i < K; ++i )
        {
            const int source_t = t - state_rows + i;
            float value = 0.0f;

            if ( source_t >= 0 )
            {
                value = x[ (b * T + source_t) * C + c ];
            }
            else if ( state )
            {
                value = state[ (b * state_rows + (state_rows + source_t)) * C + c ];
            }

            accumulator += weight[ c * K + i ] * value;
        }

        out[ idx ] = accumulator;
    }

    __global__ void causal_conv1d_update_state_fp32_kernel(
        float* __restrict__       state,
        const float* __restrict__ x,
        int B, int T, int C, int K )
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = B * C;

        if ( idx >= total )
            return;

        const int c = idx % C;
        const int b = idx / C;
        const int state_rows = K - 1;

        float staged[ kMaxStateRows ];

        for ( int j = 0; j < state_rows; ++j )
        {
            const int source_t = T - state_rows + j;

            staged[ j ] = (source_t >= 0)
                ? x[ (b * T + source_t) * C + c ]
                : state[ (b * state_rows + (state_rows + source_t)) * C + c ];
        }

        for ( int j = 0; j < state_rows; ++j )
        {
            state[ (b * state_rows + j) * C + c ] = staged[ j ];
        }
    }

    // =========================================================================
    // Host launchers

    void cuda_causal_conv1d_forward_fp32(
        float* out,
        const float* x,
        const float* state,
        const float* weight,
        const float* bias,
        int B, int T, int C, int K,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;
        const int total = B * T * C;
        const int grid_size = ceil_div( total, block_size );

        causal_conv1d_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            out, x, state, weight, bias, B, T, C, K );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_causal_conv1d_update_state_fp32(
        float* state,
        const float* x,
        int B, int T, int C, int K,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;
        const int total = B * C;
        const int grid_size = ceil_div( total, block_size );

        causal_conv1d_update_state_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            state, x, B, T, C, K );

        cudaCheck( cudaGetLastError() );
    }
}
