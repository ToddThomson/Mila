/**
 * @file CudaFp4WeightQuantization.cuh
 * @brief Host-side per-group BF16->FP4_E2M1 weight quantization for CudaLinearOp.
 *
 * Declares cuda_quantize_fp4_per_group(), which performs per-group absmax
 * quantization and uploads the resulting packed FP4 weight tensor and per-group
 * FP32 scale tensor to device. Uses void* / int parameters so the header is
 * safe to include from both NVCC-compiled and cl.exe-compiled TUs.
 *
 * FP4 E2M1 format (4 bits: S EE M):
 *   Positive representable values (nibbles 0..7): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
 *   Negative values (nibbles 8..15): sign-magnitude mirror of the above.
 *   Max representable magnitude: 6.0f
 *
 * Packing (same as INT4):
 *   weights_packed[n, k/2]: low nibble = W[n, 2i], high nibble = W[n, 2i+1]
 *
 * Scale granularity: one float32 per group of kGroupSize input channels.
 *   scale[n, g] = max(|W[n, g*kGroupSize .. (g+1)*kGroupSize)|) / 6.0f
 */

#pragma once
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Per-group BF16->FP4_E2M1 quantization with device upload.
     *
     * For each output channel and each group of kGroupSize input channels:
     *   scale[n, g] = max(|W[n, g*gs..(g+1)*gs)|) / 6.0f
     *   packed[n, k/2] = fp4_e2m1(W[n,k]/scale[n,k/gs]) nibble-packed
     *
     * @param src_bf16      Host pointer to BF16 weight blob [out_features * in_features].
     * @param dst_packed    Device pointer to packed uint8 output [out_features * in_features/2].
     * @param dst_scales    Device pointer to float32 scales [out_features * in_features/group_size].
     * @param out_features  Number of output channels (rows).
     * @param in_features   Number of input channels (columns). Must be divisible by group_size.
     * @param group_size    Quantization group size (64 or 128).
     *
     * @throws std::runtime_error if any CUDA call fails or group_size is unsupported.
     */
    void cuda_quantize_fp4_per_group(
        const void* src_bf16,
        void*       dst_packed,
        float*      dst_scales,
        int64_t     out_features,
        int64_t     in_features,
        int         group_size );

} // namespace Mila::Dnn::Compute::Cuda::Linear
