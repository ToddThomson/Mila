/**
 * @file CudaFp8WeightQuantization.cuh
 * @brief Host-side per-channel BF16→FP8_E4M3 weight quantization for CudaLinearOp.
 *
 * Declares the free function that performs per-channel absmax quantization on
 * the host, then uploads the resulting FP8_E4M3 weight tensor and FP32 scale
 * tensor to device. This interface uses plain C types (void*, int64_t) so the
 * header is safe to include from both NVCC-compiled and cl.exe-compiled TUs.
 */

#pragma once
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Per-channel BF16→FP8_E4M3 quantization with device upload.
     *
     * Reads the host BF16 source blob row by row (each row is one output channel).
     * For each row computes:
     *   scale[o] = max(|W[o,:]|) / 448.0f     (448 = max representable FP8_E4M3)
     *   W_fp8[o, i] = FP8_E4M3( W_bf16[o, i] / scale[o] )
     *
     * Uploads the resulting FP8 weight matrix and float32 scale vector to the
     * device tensors via cudaMemcpy (host-to-device). The source blob is never
     * retained on device.
     *
     * @param src_bf16      Host pointer to BF16 weight blob [out_features * in_features].
     * @param dst_fp8       Device pointer to FP8_E4M3 output tensor [out_features * in_features].
     * @param dst_scales    Device pointer to float32 scale vector [out_features].
     * @param out_features  Number of output channels (rows).
     * @param in_features   Number of input channels (columns).
     *
     * @throws std::runtime_error if either cudaMemcpy call fails.
     */
    void cuda_quantize_fp8_per_channel(
        const void* src_bf16,
        void*       dst_fp8,
        float*      dst_scales,
        int64_t     out_features,
        int64_t     in_features );

} // namespace Mila::Dnn::Compute::Cuda::Linear
