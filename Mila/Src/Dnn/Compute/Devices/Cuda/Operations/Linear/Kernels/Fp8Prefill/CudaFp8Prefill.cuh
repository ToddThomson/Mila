#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Apply per-channel (per output feature) scales to a prefill output matrix.
     *
     * Computes output[t, out] *= scales[out] for all t in [0, outer_size).
     * Used as a post-GEMM correction pass on the native FP8 cuBLASLt path, where
     * the weight matrix was quantized with per-channel scales but cuBLASLt only
     * accepts a single per-tensor scale at execute time (set to 1.0f).
     *
     * @param output      Device BF16 tensor [outer_size, out_features], modified in-place.
     * @param scales      Device float tensor [out_features], one scale per output channel.
     * @param outer_size  Number of tokens (batch * sequence_length for prefill).
     * @param out_features Number of output features (output channels).
     * @param stream      CUDA stream.
     */
    void cuda_fp8_apply_per_channel_scales(
        __nv_bfloat16* output,
        const float* scales,
        int outer_size,
        int out_features,
        cudaStream_t stream );

    /**
     * @brief Dequantize an FP8-E4M3 weight matrix to BF16 using per-channel scales.
     *
     * Computes output[out, k] = float(input[out, k]) * scales[out] for all channels.
     * Used on the BF16 fallback prefill path when a native FP8 cuBLASLt plan is
     * unavailable, producing a temporary BF16 weight copy that feeds the standard
     * BF16 cuBLASLt GEMM plan.
     *
     * One CUDA block per output channel; threads stride over in_features.
     *
     * @param output      Device BF16 tensor [out_features, in_features], written.
     * @param input       Device FP8-E4M3 tensor [out_features, in_features], read-only.
     * @param scales      Device float tensor [out_features], per-channel scales.
     * @param out_features Number of output channels (rows of weight matrix).
     * @param in_features  Number of input features (columns of weight matrix).
     * @param stream       CUDA stream.
     */
    void cuda_fp8_dequantize_to_bf16(
        __nv_bfloat16* output,
        const __nv_fp8_e4m3* input,
        const float* scales,
        int out_features,
        int in_features,
        cudaStream_t stream );

    /**
     * @brief Quantize a BF16 activation matrix to FP8_E4M3 with dynamic per-token scales.
     *
     * Prefill activation path for the W4A8-FP8 GEMM. One block per token (row):
     * row absmax reduction -> scales_out[t] = max(absmax, epsilon) / 448.0f
     * (448 = E4M3 max magnitude), then elementwise quantize of that row in the
     * same launch: fp8_out[t, k] = E4M3( input[t, k] / scales_out[t] ).
     *
     * Per-token (not per-tensor) scaling is the numerics fix for the W4A8-FP8 path:
     * a single per-tensor scale lets one outlier token crush every other token's
     * resolution, and the resulting error compounds across transformer layers into
     * incoherent generation even though a per-layer 5e-2 oracle passes.
     *
     * cuBLASLt on Ada accepts only per-tensor scale pointers, so these per-token
     * scales are NOT bound to the GEMM descriptor (B_SCALE stays a constant 1.0f).
     * They are applied exactly after the GEMM by cuda_fp8_apply_per_token_scales:
     *   Y[t, n] = scales_out[t] * ( sB * sum_k X8[t, k] * W8[n, k] ).
     *
     * @param fp8_out     Device FP8_E4M3 tensor [outer_size * in_features], written.
     * @param scales_out  Device float tensor [outer_size], one scale per token, written.
     * @param input       Device BF16 activations [outer_size * in_features], read-only.
     * @param outer_size  Number of tokens (batch * sequence_length for prefill).
     * @param in_features Inner dimension K.
     * @param stream      CUDA stream.
     */
    void cuda_quantize_bf16_to_fp8_per_token(
        __nv_fp8_e4m3* fp8_out,
        float*         scales_out,
        const __nv_bfloat16* input,
        int            outer_size,
        int            in_features,
        cudaStream_t   stream );

    /**
     * @brief Apply per-token scales (and optional bias) to a prefill output matrix.
     *
     * Computes output[t, out] = output[t, out] * scales[t] + bias[out] for all tokens.
     * Post-GEMM epilogue for the W4A8-FP8 prefill path: the GEMM runs with a unit
     * activation scale (cuBLASLt Ada FP8 accepts only per-tensor scale pointers), and
     * the true per-token activation scales are applied here. The factorization is
     * exact -- each output row is a single linear function of its token's scale.
     * Folds the bias addition so no separate cuda_add_bias pass is needed.
     *
     * @param output      Device BF16 tensor [outer_size, out_features], modified in-place.
     * @param scales      Device float tensor [outer_size], one scale per token.
     * @param bias        Device BF16 tensor [out_features], or nullptr for no bias.
     * @param outer_size  Number of tokens (rows of the output matrix).
     * @param out_features Number of output features (columns of the output matrix).
     * @param stream      CUDA stream.
     */
    void cuda_fp8_apply_per_token_scales(
        __nv_bfloat16* output,
        const float* scales,
        const __nv_bfloat16* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream );

    /**
     * @brief Add a bias vector to every row of a BF16 output matrix.
     *
     * Computes output[t, out] += bias[out] for all t in [0, outer_size).
     * Used as a post-GEMM bias addition on the FP8 BF16-fallback prefill path,
     * where the cuBLASLt plan is built with has_bias=false to avoid the epilogue
     * constraints that cause INVALID_VALUE on Ada for multi-row GEMMs.
     *
     * One thread per output feature; iterates over outer_size tokens.
     * Grid: ceil(out_features / kBlockSize) blocks of kBlockSize threads.
     *
     * @param output      Device BF16 tensor [outer_size, out_features], modified in-place.
     * @param bias        Device BF16 tensor [out_features], read-only.
     * @param outer_size  Number of tokens (rows of the output matrix).
     * @param out_features Number of output features (columns of the output matrix).
     * @param stream      CUDA stream.
     */
    void cuda_add_bias(
        __nv_bfloat16* output,
        const __nv_bfloat16* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream );

    /**
     * @brief FP32 overload of cuda_add_bias.
     *
     * The non-quantized FP32 prefill path also builds its cuBLASLt plan with
     * has_bias=false and adds bias post-GEMM: cuBLASLt's heuristic returns
     * CUBLAS_STATUS_NOT_SUPPORTED for CUBLAS_COMPUTE_32F combined with
     * CUBLASLT_EPILOGUE_BIAS on this configuration, so the epilogue cannot be used.
     *
     * @param output      Device FP32 tensor [outer_size, out_features], modified in-place.
     * @param bias        Device FP32 tensor [out_features], read-only.
     * @param outer_size  Number of tokens (rows of the output matrix).
     * @param out_features Number of output features (columns of the output matrix).
     * @param stream      CUDA stream.
     */
    void cuda_add_bias(
        float* output,
        const float* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream );
}
