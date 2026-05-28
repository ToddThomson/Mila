/**
 * @file CudaW4A16Gemm.Wmma.cuh
 * @brief WMMA-accelerated FP4 E2M1 x BF16 W4A16 GEMM for SM80+.
 *
 * Declares cuda_fp4a16_gemm_wmma(), a drop-in replacement for cuda_fp4a16_gemm()
 * that uses nvcuda::wmma m16n16k16 BF16 tensor core MMA. Dequantization of packed
 * FP4 E2M1 nibbles happens in shared memory before each WMMA B fragment load.
 * Requires SM >= 8.0 (Ampere/Ada) for BF16 WMMA support.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief WMMA-accelerated per-group FP4 E2M1 W4A16 GEMM.
     *
     * Computes output[m, n] = sum_k( activations[m, k] * dequant(W[n, k]) ) + bias[n]
     *
     * where dequant(W[n, k]) = fp4_e2m1_lut[nibble(W[n,k])] * scales[n, k/group_size].
     *
     * Signature is identical to cuda_fp4a16_gemm — drop-in at the call site.
     * SM gate: caller must only invoke when SM major >= 8.
     *
     * Layout (all row-major):
     *   activations    : [outer_size x in_features]              BF16
     *   weights_packed : [out_features x in_features/2]          uint8  (2 FP4 E2M1 per byte)
     *   scales         : [out_features x in_features/group_size] float32
     *   bias           : [out_features]                          BF16 (optional, nullptr ok)
     *   output         : [outer_size x out_features]             BF16
     *
     * @param output         Device BF16 output [outer_size x out_features].
     * @param activations    Device BF16 activations [outer_size x in_features].
     * @param weights_packed Device uint8 packed FP4 E2M1 weights [out_features x in_features/2].
     * @param scales         Device float32 per-group scales [out_features x in_features/group_size].
     * @param bias           Device BF16 bias [out_features], or nullptr.
     * @param outer_size     M — number of input/output rows.
     * @param in_features    K — inner dimension (must be divisible by group_size).
     * @param out_features   N — number of output channels.
     * @param group_size     Quantization group size along K (64 or 128).
     * @param stream         CUDA stream.
     */
    void cuda_fp4a16_gemm_wmma(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const uint8_t*       weights_packed,
        const float*         scales,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        int                  group_size,
        cudaStream_t         stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
