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
#include <cuda_fp8.h>
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

    /**
     * @brief Fused per-group FP4_E2M1 W4A16 GEMM.
     *
     * Identical tiled GEMM structure to cuda_w4a16_gemm but uses FP4 E2M1 nibble
     * decode instead of INT4 arithmetic during the tile load. No zero-points —
     * sign is encoded directly in the nibble (bit 3).
     *
     * Dequantization per element:
     *   nibble  = low nibble if k even, high nibble if k odd
     *   W_f32   = fp4_e2m1_lut[nibble] * scales[n, k/group_size]
     *
     * FP4 E2M1 representable values (nibble → float):
     *   0→0.0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
     *   8→−0.0, 9→−0.5, 10→−1.0, 11→−1.5, 12→−2.0, 13→−3.0, 14→−4.0, 15→−6.0
     *
     * Layout (all row-major):
     *   activations    : [outer_size x in_features]              BF16
     *   weights_packed : [out_features x in_features/2]          uint8  (2 FP4 per byte)
     *   scales         : [out_features x in_features/group_size] float32
     *   bias           : [out_features]                          BF16 (optional, nullptr ok)
     *   output         : [outer_size x out_features]             BF16
     *
     * @param output         Device BF16 output [outer_size x out_features].
     * @param activations    Device BF16 activations [outer_size x in_features].
     * @param weights_packed Device uint8 packed FP4 weights [out_features x in_features/2].
     * @param scales         Device float32 per-group scales [out_features x in_features/group_size].
     * @param bias           Device BF16 bias [out_features], or nullptr.
     * @param outer_size     M — number of input/output rows.
     * @param in_features    K — inner dimension (must be divisible by group_size).
     * @param out_features   N — number of output channels.
     * @param group_size     Quantization group size along K (64 or 128).
     * @param stream         CUDA stream.
     */
    void cuda_fp4a16_gemm(
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

    /**
     * @brief Per-group FP4 E2M1 weight matrix -> BF16 dequantization.
     *
     * Expands the packed FP4 weight tensor into a caller-owned BF16 staging buffer
     * for the 2-phase prefill path (dequantize -> standard BF16 cuBLASLt GEMM),
     * mirroring cuda_fp8_dequantize_to_bf16. BF16 rounding matches the fused GEMM
     * kernels' weight treatment.
     *
     * @param output         Device BF16 staging buffer [out_features x in_features].
     * @param weights_packed Device uint8 packed FP4 weights [out_features x in_features/2].
     * @param scales         Device float32 per-group scales [out_features x in_features/group_size].
     * @param out_features   N — number of output channels (grid dimension).
     * @param in_features    K — inner dimension (must be divisible by group_size).
     * @param group_size     Quantization group size along K (64 or 128).
     * @param stream         CUDA stream.
     */
    void cuda_fp4_dequantize_to_bf16(
        __nv_bfloat16* output,
        const uint8_t* weights_packed,
        const float*   scales,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream );

    /**
     * @brief Compute the static per-tensor FP8 weight scale from the FP4 group scales.
     *
     * Derived entirely from the already-stored FP4 per-group scales -- no access to
     * the packed weights is needed. Because scale[n,g] = group_absmax / 6 and the FP4
     * decode magnitude peaks at 6, the global weight absmax equals 6 * max(scale), so
     *   weight_fp8_scale = (6 / 448) * max(scale)   (448 = E4M3 max magnitude).
     *
     * This is the sB divisor applied when upcasting FP4 -> FP8 and the value bound to
     * the cuBLASLt A_SCALE_POINTER for the weight operand. Computed once at load.
     *
     * @param weight_fp8_scale_out Device float scalar, written with the per-tensor scale.
     * @param fp4_group_scales     Device float32 FP4 per-group scales [num_scales].
     * @param num_scales           out_features * (in_features / group_size).
     * @param stream               CUDA stream.
     */
    void cuda_compute_fp8_weight_scale(
        float*       weight_fp8_scale_out,
        const float* fp4_group_scales,
        int64_t      num_scales,
        cudaStream_t stream );

    /**
     * @brief Per-group FP4 E2M1 weight matrix -> FP8_E4M3 transient upcast (prefill).
     *
     * Sibling of cuda_fp4_dequantize_to_bf16 that targets the W4A8-FP8 GEMM: expands
     * the packed FP4 weights into a caller-owned FP8_E4M3 staging buffer, dividing by
     * the per-tensor weight scale so the values land in E4M3 range. Half the staging
     * traffic of the BF16 path (1 byte/elem vs 2). The FP4 per-group scale is folded
     * in; the residual per-tensor scale is recovered by cuBLASLt's A_SCALE at GEMM time:
     *   fp8[n,k] = E4M3( fp4_decode(nibble) * scale[n, k/group_size] / weight_fp8_scale ).
     *
     * @param output           Device FP8_E4M3 staging buffer [out_features x in_features].
     * @param weights_packed   Device uint8 packed FP4 weights [out_features x in_features/2].
     * @param scales           Device float32 per-group scales [out_features x in_features/group_size].
     * @param weight_fp8_scale Device float scalar (from cuda_compute_fp8_weight_scale).
     * @param out_features     N -- number of output channels (grid dimension).
     * @param in_features      K -- inner dimension (must be divisible by group_size).
     * @param group_size       Quantization group size along K (64 or 128).
     * @param stream           CUDA stream.
     */
    void cuda_fp4_dequantize_to_fp8(
        __nv_fp8_e4m3* output,
        const uint8_t* weights_packed,
        const float*   scales,
        const float*   weight_fp8_scale,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
