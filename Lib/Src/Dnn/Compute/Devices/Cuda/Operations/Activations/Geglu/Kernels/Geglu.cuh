/**
 * @file Geglu.cuh
 * @brief CUDA GeGLU (GELU-gated linear unit) forward kernel declarations.
 *
 * GeGLU(gate, up) = GeluTanh(gate) * up, with the gate/up halves laid out
 * contiguously per token (identical to the SwiGLU layout):
 *   X: [ gate_0 ... gate_(hw-1) | up_0 ... up_(hw-1) ]  per token row
 *   Y: [ y_0    ... y_(hw-1)    ]                        per token row
 * where hw = half_width = last input dimension / 2, and N = T * half_width.
 *
 * Forward only: Gemma is inference-only, so the GeGLU backward (training) is not
 * implemented here. The optimized SiLU SwiGLU kernels are a separate, untouched
 * unit; this is a deliberately simple scalar kernel (the FFN is GEMM-bound, not
 * activation-bound) that shares the single GeluTanh math source.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Geglu
{
    void cuda_geglu_forward_fp32(
        float* Y, const float* X,
        int N, int half_width,
        cudaStream_t stream );

    void cuda_geglu_forward_bf16(
        __nv_bfloat16* Y, const __nv_bfloat16* X,
        int N, int half_width,
        cudaStream_t stream );
}
