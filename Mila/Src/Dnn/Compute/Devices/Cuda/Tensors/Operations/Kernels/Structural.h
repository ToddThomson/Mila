/**
 * @file Structural.h
 * @brief Host-callable launcher declarations for CUDA structural tensor operations.
 */

#pragma once

#include <cuda_runtime.h>

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Vectorized 3-way last-dimension split, float32.
     *
     * Splits input [B, T, D0+D1+D2] into three output tensors
     * [B, T, D0], [B, T, D1], [B, T, D2] along the last dimension.
     *
     * @param src    Input device buffer  [B * T * (D0+D1+D2) floats].
     * @param out0   Output device buffer [B * T * D0 floats].
     * @param out1   Output device buffer [B * T * D1 floats].
     * @param out2   Output device buffer [B * T * D2 floats].
     * @param B      Batch size.
     * @param T      Sequence length.
     * @param D0     Last-dim size of output 0. Must be a multiple of 4.
     * @param D1     Last-dim size of output 1. Must be a multiple of 4.
     * @param D2     Last-dim size of output 2. Must be a multiple of 4.
     * @param stream CUDA stream for kernel scheduling.
     */
    void cuda_split3_fp32(
        const float* __restrict__ src,
        float* __restrict__ out_a,
        float* __restrict__ out_b,
        float* __restrict__ out_c,
        int src_rows,
        int dim_a, int dim_b, int dim_c,
        cudaStream_t stream );

}
