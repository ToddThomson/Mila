/**
 * @file CausalConv1d.Bf16.cu
 * @brief BF16 kernels and host launchers for the depthwise causal 1-D convolution.
 *
 * Accumulation is in float. The taps are few (K = 4 on Qwen 3.8) but the channel count
 * is not, and BF16's 8-bit mantissa makes a running sum of four products the wrong place
 * to save two bytes. Requires SM >= 8.0 for native BF16.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Convolution
{
    constexpr int kMaxStateRowsBf16 = 7;

    __global__ void causal_conv1d_forward_bf16_kernel(
        __nv_bfloat16* __restrict__       out,
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ state,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
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

        float accumulator = bias ? __bfloat162float( bias[ c ] ) : 0.0f;

        for ( int i = 0; i < K; ++i )
        {
            const int source_t = t - state_rows + i;
            float value = 0.0f;

            if ( source_t >= 0 )
            {
                value = __bfloat162float( x[ (b * T + source_t) * C + c ] );
            }
            else if ( state )
            {
                value = __bfloat162float(
                    state[ (b * state_rows + (state_rows + source_t)) * C + c ] );
            }

            accumulator += __bfloat162float( weight[ c * K + i ] ) * value;
        }

        out[ idx ] = __float2bfloat16( accumulator );
    }

    __global__ void causal_conv1d_update_state_bf16_kernel(
        __nv_bfloat16* __restrict__       state,
        const __nv_bfloat16* __restrict__ x,
        int B, int T, int C, int K )
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = B * C;

        if ( idx >= total )
            return;

        const int c = idx % C;
        const int b = idx / C;
        const int state_rows = K - 1;

        __nv_bfloat16 staged[ kMaxStateRowsBf16 ];

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

    void cuda_causal_conv1d_forward_bf16(
        __nv_bfloat16* out,
        const __nv_bfloat16* x,
        const __nv_bfloat16* state,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        int B, int T, int C, int K,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;
        const int total = B * T * C;
        const int grid_size = ceil_div( total, block_size );

        causal_conv1d_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (
            out, x, state, weight, bias, B, T, C, K );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_causal_conv1d_update_state_bf16(
        __nv_bfloat16* state,
        const __nv_bfloat16* x,
        int B, int T, int C, int K,
        cudaStream_t stream )
    {
        constexpr int block_size = 256;
        const int total = B * C;
        const int grid_size = ceil_div( total, block_size );

        causal_conv1d_update_state_bf16_kernel << <grid_size, block_size, 0, stream >> > (
            state, x, B, T, C, K );

        cudaCheck( cudaGetLastError() );
    }
}
