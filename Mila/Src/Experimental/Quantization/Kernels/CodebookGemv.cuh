/**
 * @file CodebookGemv.cuh
 * @brief Decode-path GEMV launchers for the codebook weight policies (W2A16 / W3A16).
 * The packed layout consumed here is normative in Experimental/Quantization/CodebookPacking.ixx.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Experimental::Quantization::Kernels
{
    /**
     * y[oc] = sum_c( x[c] * codebook[code(oc, c)] * scale(oc, c / group_size) ) + bias[oc]
     *
     * Requirements:
     *   - C % 16 == 0 and C % group_size == 0 (uint32/uint16 plane loads, one scale per chunk)
     *   - group_size in {32, 64, 128}
     *   - codebook is a DEVICE pointer: 4 float entries (2-bit), 8 entries (3-bit)
     *   - scales are IEEE half, row-major [oc][C / group_size]
     *   - bias may be nullptr
     */
    void launch_codebook2_matvec_decode(
        cudaStream_t stream,
        __nv_bfloat16* y, const __nv_bfloat16* x,
        const std::uint8_t* codes_packed, const __half* scales, const float* codebook,
        const __nv_bfloat16* bias, int C, int OC, int group_size );

    void launch_codebook3_matvec_decode(
        cudaStream_t stream,
        __nv_bfloat16* y, const __nv_bfloat16* x,
        const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const __half* scales, const float* codebook,
        const __nv_bfloat16* bias, int C, int OC, int group_size );

    // Host-convenience entries for tests and bring-up: plain host pointers in, one
    // synchronous device round trip, first cudaError_t as int (0 == success).
    // Activations, bias and the returned y pass through BF16 exactly as the launch
    // path sees them.
    int run_codebook2_matvec_decode_host(
        const float* x, const std::uint8_t* codes_packed, const std::uint16_t* scale_bits,
        const float* codebook, const float* bias, float* y, int C, int OC, int group_size );

    int run_codebook3_matvec_decode_host(
        const float* x, const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const std::uint16_t* scale_bits, const float* codebook, const float* bias, float* y,
        int C, int OC, int group_size );

    // 1 when a CUDA device is present, else 0. Lets tests skip cleanly.
    int codebook_gemv_device_available();
}
