/**
 * @file CudaW4A16Gemm.cuh
 * @brief Fused W4A16 GEMM: packed INT4 weights x BF16 activations -> BF16 output.
 *
 * Declares cuda_w4a16_gemm(), a tiled GEMM kernel that loads packed INT4 weights and
 * dequantizes them per-group inline in shared memory using per-group scales and zero
 * points. No intermediate FP16/BF16 weight buffer is required.
 *
 * Weight storage format (GPTQ-compatible):
 *   weights_packed : [N, K/2]            uint8  — 2 INT4 values per byte
 *                                                  low  nibble = W[n, 2i  ]
 *                                                  high nibble = W[n, 2i+1]
 *   scales         : [N, K/group_size]   float32 — one scale per group
 *   zero_points    : [N, K/(group_size*2)] uint8  — 2 packed INT4 zeros per byte
 *                                                  (nullptr -> symmetric, zero = 8)
 *
 * Dequantization per element:
 *   zero  = nibble from zero_points (or 8.0f if nullptr)
 *   W_f32 = (float(nibble) - zero) * scale[n, k / group_size]
 *
 * Memory bandwidth (M = outer_size, N = out_features, K = in_features):
 *   Weight reads: N * K * 0.5 bytes (INT4) vs N * K * 2 bytes (BF16)  =  4x reduction
 *   vs W8A16 fused: 2x further reduction over FP8
 *
 * Intended for SM >= 8.0 (Ampere/Ada) where BF16 compute is supported.
 * Requires K to be divisible by group_size and group_size to be a multiple of the
 * kernel tile size (16). Currently supports group_size = 64 and 128.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Fused per-group W4A16 GEMM.
     *
     * Computes output[m, n] = sum_k( activations[m, k] * dequant(W[n, k]) ) + bias[n]
     *
     * where dequant(W[n, k]) = (float(nibble[n,k]) - zero[n, k/g]) * scales[n, k/g]
     * and g = group_size.
     *
     * Layout (all row-major):
     *   activations    : [outer_size x in_features]         BF16
     *   weights_packed : [out_features x in_features/2]     uint8  (2 INT4 per byte)
     *   scales         : [out_features x in_features/group_size]  float32
     *   zero_points    : [out_features x in_features/(group_size*2)]  uint8  (2 INT4 per byte)
     *                    Pass nullptr for symmetric quantization (zero = 8).
     *   bias           : [out_features]                     BF16   (optional, may be nullptr)
     *   output         : [outer_size x out_features]        BF16
     *
     * Requirements:
     *   SM >= 8.0 for BF16 compute.
     *   in_features divisible by group_size.
     *   group_size must be 64 or 128.
     *   out_features and in_features need not be tile-size multiples.
     *
     * @param output         Device BF16 output [outer_size x out_features].
     * @param activations    Device BF16 activations [outer_size x in_features].
     * @param weights_packed Device uint8 packed INT4 weights [out_features x in_features/2].
     * @param scales         Device float32 per-group scales [out_features x in_features/group_size].
     * @param zero_points    Device uint8 packed INT4 zero points, or nullptr for symmetric.
     * @param bias           Device BF16 bias [out_features], or nullptr.
     * @param outer_size     M — number of input/output rows.
     * @param in_features    K — inner dimension (must be divisible by group_size).
     * @param out_features   N — number of output channels.
     * @param group_size     Quantization group size along K (64 or 128).
     * @param stream         CUDA stream.
     */
    void cuda_w4a16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const uint8_t*       weights_packed,
        const float*         scales,
        const uint8_t*       zero_points,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        int                  group_size,
        cudaStream_t         stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
