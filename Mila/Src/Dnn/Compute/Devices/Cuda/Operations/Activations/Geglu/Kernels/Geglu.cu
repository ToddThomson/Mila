/**
 * @file Geglu.cu
 * @brief CUDA GeGLU forward kernels (FP32 + BF16), scalar, FP32 arithmetic.
 *
 * out[token, col] = GeluTanh(gate[token, col]) * up[token, col]. The GeluTanh math
 * comes from the shared ElementwiseActivation functor library (single source of
 * truth) so it stays bit-consistent with the Activation component.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Geglu.cuh"
// Shared GeluTanh math source (single source of truth). File-relative include at the
// same depth the Elementwise kernel uses; resolves identically in-tree and installed.
#include "../../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

namespace Mila::Dnn::Compute::Cuda::Geglu
{
    using Mila::Dnn::Activations::GeluTanh;

    // Thread i owns output element i. token = i / half_width, col = i % half_width;
    // the gate half precedes the up half within each 2*half_width token row.
    __global__ void geglu_forward_fp32_kernel(
        float* __restrict__ Y, const float* __restrict__ X,
        int N, int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i >= N ) return;

        int token = i / half_width;
        int col = i % half_width;
        int gate_idx = token * 2 * half_width + col;
        int up_idx = gate_idx + half_width;

        GeluTanh gelu;
        Y[ i ] = gelu.fwd( X[ gate_idx ] ) * X[ up_idx ];
    }

    __global__ void geglu_forward_bf16_kernel(
        __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ X,
        int N, int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i >= N ) return;

        int token = i / half_width;
        int col = i % half_width;
        int gate_idx = token * 2 * half_width + col;
        int up_idx = gate_idx + half_width;

        // Promote to FP32 for the gate math (BF16's 7 mantissa bits are insufficient
        // for tanh/exp), narrow only at the store. Matches the SwiGLU kernel strategy.
        GeluTanh gelu;
        float gate = __bfloat162float( X[ gate_idx ] );
        float up = __bfloat162float( X[ up_idx ] );
        Y[ i ] = __float2bfloat16( gelu.fwd( gate ) * up );
    }

    void cuda_geglu_forward_fp32(
        float* Y, const float* X,
        int N, int half_width,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int grid_size = (N + block_size - 1) / block_size;

        geglu_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N, half_width);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_geglu_forward_bf16(
        __nv_bfloat16* Y, const __nv_bfloat16* X,
        int N, int half_width,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int grid_size = (N + block_size - 1) / block_size;

        geglu_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (Y, X, N, half_width);

        cudaCheck( cudaGetLastError() );
    }
}
