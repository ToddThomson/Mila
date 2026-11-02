/**
 * @file CudaInt4Gemm.cuh
 * @brief Q4_0 prefill: BF16 activations quantized to INT8 per 32-element block, multiplied against the packed
 * INT4 codes on the INT8 tensor cores.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /// Elements that share one activation scale, and one weight scale: one Q4_0 block.
    inline constexpr int kInt4GemmBlockSize = 32;

    /// The GEMM's shallower K tile: in_features must be a whole number of them. A width 128 divides runs 128 deep.
    inline constexpr int kInt4GemmInFeaturesMultiple = 64;

    /// Offset of the FP32 block scales in the prefill's scratch, after the INT8 codes rounded up to 16 bytes.
    constexpr std::size_t int4GemmScaleOffset( std::size_t rows, std::size_t in_features ) noexcept
    {
        return ( rows * in_features + 15u ) & ~static_cast<std::size_t>( 15u );
    }

    /// Bytes of scratch the prefill needs for rows x in_features activations: the INT8 codes, then the scales.
    constexpr std::size_t int4GemmScratchBytes( std::size_t rows, std::size_t in_features ) noexcept
    {
        return int4GemmScaleOffset( rows, in_features ) + rows * ( in_features / kInt4GemmBlockSize ) * sizeof( float );
    }

    /**
     * @brief Quantize BF16 activations to INT8, one FP32 scale per 32-element block.
     *
     * For a block with largest magnitude a: scale = a / 127 and code = round-half-even( x * ( 127 / a ) ), all
     * in FP32; an all-zero block has scale 0 and zero codes. Codes lie in [-127, 127].
     *
     * @param codes       Device INT8 [rows x in_features].
     * @param scales      Device FP32 [rows x in_features / 32].
     * @param input       Device BF16 [rows x in_features].
     * @param in_features Must be a multiple of 32.
     */
    void cuda_quantize_bf16_to_int8_per_block(
        int8_t*              codes,
        float*               scales,
        const __nv_bfloat16* input,
        int                  rows,
        int                  in_features,
        cudaStream_t         stream );

    /**
     * @brief output[m, n] = bf16( bias[n] + sum over blocks b of
     *        activation_scale[m, b] * weight_scale[n, b] * sum over the block of code_a * ( code_w - 8 ) ).
     *
     * Each block's dot product is exact in INT32; the scaling and the sum over blocks run in FP32, and the result
     * rounds once to BF16.
     *
     * @param output            Device BF16 [rows x out_features].
     * @param activation_codes  Device INT8 [rows x in_features], from cuda_quantize_bf16_to_int8_per_block.
     * @param activation_scales Device FP32 [rows x in_features / 32].
     * @param weight_codes      Device packed INT4 [out_features x in_features / 2], layout of Int4Packing.ixx.
     * @param weight_scales     Device IEEE half [out_features x in_features / 32]; may be negative.
     * @param bias              Device BF16 [out_features], or nullptr.
     * @param in_features       Must be a multiple of kInt4GemmInFeaturesMultiple.
     * @param out_features      Must be even.
     *
     * @throws std::invalid_argument on an unsupported shape.
     */
    void cuda_int4_int8_gemm(
        __nv_bfloat16*       output,
        const int8_t*        activation_codes,
        const float*         activation_scales,
        const uint8_t*       weight_codes,
        const __half*        weight_scales,
        const __nv_bfloat16* bias,
        int                  rows,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
