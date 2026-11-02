/**
 * @file CudaDecodeRows.cuh
 * @brief Decode's arithmetic for a few rows at once: every weight format as one tensor-core product whose work per
 * weight does not grow with the row count (Gemma4Mtp.md 4.7).
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /// The most rows one call takes: the rows are the n = 8 operand of mma.m16n8k16.
    inline constexpr int kDecodeRowsMaximum = 8;

    enum class DecodeRowsFormat
    {
        Bf16,           ///< BF16 weights [OC, C]
        Fp8PerChannel,  ///< FP8 E4M3 weights [OC, C], FP32 scale per output channel
        Fp4,            ///< FP4 E2M1 nibbles [OC, C / 2], low nibble the even column; FP32 scale per group
        Int4,           ///< INT4 codes [OC, C / 2] (Int4Packing.ixx); IEEE half scale per group of 32
        Int6,           ///< INT6 rows of [3C / 4] bytes, low nibbles then high two bits (Int6Packing.ixx); half scale per 32
        Codebook2,      ///< 2-bit codes [OC, C / 4] into a 4-entry FP32 table; half scale per group
        Codebook3       ///< Codebook2's plane plus a 1-bit plane [OC, C / 8], an 8-entry table; half scale per group
    };

    /// The weight tensors of one Linear, by format. Members a format does not use are null.
    struct DecodeRowsWeights
    {
        DecodeRowsFormat format;
        const void* codes;                  ///< the primary weight tensor
        const std::uint8_t* high_plane;     ///< Codebook3 only
        const void* scales;                 ///< half or float by format; null for Bf16
        const float* codebook;              ///< codebook formats only
        int group_size;                     ///< per-group formats; ignored otherwise
    };

    /**
     * @brief y[r, oc] = sum_c x[r, c] * W[oc, c] + bias[oc] for r < rows, rows <= kDecodeRowsMaximum.
     *
     * Each weight is read once for all rows and widened to BF16 in registers -- exactly, for every format but the
     * codebooks, whose FP32 entries are carried as a BF16 pair -- and the group or channel scale is applied to the FP32
     * accumulator, as the decode matvecs apply it. The sums run in another order than the matvecs', so a row equals a
     * one-row decode to FP32 rounding, not bit for bit.
     *
     * @throws std::invalid_argument unless 1 <= rows <= kDecodeRowsMaximum and C is a multiple of 32 (and of the
     *         format's group size).
     */
    void cuda_decode_rows_bf16(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const DecodeRowsWeights& weights,
        const __nv_bfloat16* bias,
        int rows,
        int C,
        int OC,
        cudaStream_t stream );
}
