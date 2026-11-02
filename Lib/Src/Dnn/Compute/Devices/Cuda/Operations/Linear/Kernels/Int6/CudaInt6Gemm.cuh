/**
 * @file CudaInt6Gemm.cuh
 * @brief Batched INT6 forward: BF16 activations against six-bit codes on the BF16 tensor cores, each group's
 * scale applied once in FP32.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /// Elements that share one weight scale.
    inline constexpr int kInt6GemmGroupSize = 32;

    /// The GEMM's K tile: in_features must be a whole number of them.
    inline constexpr int kInt6GemmInFeaturesMultiple = 64;

    /**
     * @brief output[m, n] = bf16( bias[n] + sum over groups g of
     *        weight_scale[n, g] * sum over the group of input[m, k] * ( code[n, k] - 32 ) ).
     *
     * A code widens to BF16 exactly and a product of it with a BF16 activation is exact in FP32, so each group's
     * dot product rounds only in its FP32 sum; the scale applies once per group, and the result rounds once to
     * BF16. Nothing is staged and the activations are not quantized.
     *
     * @param output        Device BF16 [rows x out_features].
     * @param input         Device BF16 [rows x in_features].
     * @param weight_codes  Device packed INT6 [out_features x 3 * in_features / 4], layout of Int6Packing.ixx.
     * @param weight_scales Device IEEE half [out_features x in_features / 32]; may be negative.
     * @param bias          Device BF16 [out_features], or nullptr.
     * @param in_features   Must be a multiple of kInt6GemmInFeaturesMultiple.
     * @param out_features  Must be even.
     *
     * @throws std::invalid_argument on an unsupported shape.
     */
    void cuda_int6_bf16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* input,
        const uint8_t*       weight_codes,
        const __half*        weight_scales,
        const __nv_bfloat16* bias,
        int                  rows,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
