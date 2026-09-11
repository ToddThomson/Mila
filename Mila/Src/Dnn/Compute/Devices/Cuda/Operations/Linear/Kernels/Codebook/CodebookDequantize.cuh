/**
 * @file CodebookDequantize.cuh
 * @brief Expands packed codebook weights into a BF16 staging buffer for the prefill GEMM.
 * The packed layout consumed here is normative in Experimental/Quantization/CodebookPacking.ixx.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * staging[oc][c] = codebook[code(oc, c)] * scale(oc, c / group_size), as BF16.
     *
     * This is the two-phase prefill path: one full BF16 expansion of the weight matrix
     * per forward, then a standard cuBLASLt GEMM. It is deliberately the cheap way to
     * reach a working prefill at 3B scale and does NOT scale to Qwen3.8 -- one
     * 5120x17408 FFN matrix expands to 170 MiB, which is why Specifications/Qwen3.8.md
     * section 5 requires the fused tile-load GEMM before the 27B chassis runs.
     *
     * Requirements match the GEMV: C % 16 == 0 and C % group_size == 0, group_size in
     * {32, 64, 128}, codebook a DEVICE pointer of 4 (2-bit) or 8 (3-bit) float entries,
     * scales IEEE half row-major [oc][C / group_size].
     */
    void launch_codebook2_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const __half* scales, const float* codebook,
        int C, int OC, int group_size );

    void launch_codebook3_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const __half* scales, const float* codebook,
        int C, int OC, int group_size );
}
