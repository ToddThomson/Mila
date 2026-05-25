/**
 * @file CudaW8A16Gemm.cuh
 * @brief Fused W8A16 GEMM: FP8_E4M3 weights x BF16 activations -> BF16 output.
 *
 * Declares cuda_w8a16_gemm(), a tiled GEMM kernel that loads FP8 weights from global
 * memory and dequantizes them per-channel inline in shared memory. Replaces the
 * 2-phase approach (cuda_fp8_dequantize_to_bf16 + cuBLASLt BF16 GEMM), eliminating
 * the BF16 staging buffer and the extra write+read of the full weight matrix.
 *
 * Memory traffic comparison (M = outer_size, N = out_features, K = in_features):
 *   2-phase: N*K*1 (FP8 read) + N*K*2 (BF16 write) + N*K*2 (BF16 read)
 *          + M*K*2 (activation) + M*N*2 (output) = 5*N*K + 2*M*K + 2*M*N bytes
 *   Fused:   N*K*1 (FP8 read) + M*K*2 (activation) + M*N*2 (output)
 *          = N*K + 2*M*K + 2*M*N bytes
 *
 * Intended as the primary batch compute path for the FP8 quantized linear op on
 * SM >= 8.0 (Ampere/Ada) hardware. For outer_size == 1 the matvec decode kernel
 * remains the correct path.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Fused per-channel W8A16 GEMM.
     *
     * Computes output[m, n] = sum_k( activations[m, k] * (float)weights[n, k] * scales[n] )
     *                       + bias[n]
     *
     * using a shared-memory tiled GEMM with inline FP8->float dequantization and per-channel
     * scale application. The scale multiply occurs in registers during tile load — no additional
     * global memory traffic beyond the FP8 weight read and the float32 scale vector.
     *
     * Layout (all row-major):
     *   activations : [outer_size x in_features]   BF16
     *   weights     : [out_features x in_features]  FP8_E4M3
     *   scales      : [out_features]                float32  (per output channel)
     *   bias        : [out_features]                BF16     (optional, may be nullptr)
     *   output      : [outer_size x out_features]  BF16
     *
     * Requirements:
     *   SM >= 8.0 for BF16 compute support.
     *   Dimensions need not be tile-size multiples; boundary threads are masked.
     *
     * @param output        Device BF16 output tensor [outer_size x out_features].
     * @param activations   Device BF16 activation tensor [outer_size x in_features].
     * @param weights       Device FP8_E4M3 weight tensor [out_features x in_features].
     * @param scales        Device float32 per-channel scale vector [out_features].
     * @param bias          Device BF16 bias vector [out_features], or nullptr.
     * @param outer_size    M — number of input/output rows.
     * @param in_features   K — inner dimension.
     * @param out_features  N — number of output channels.
     * @param stream        CUDA stream.
     */
    void cuda_w8a16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const __nv_fp8_e4m3* weights,
        const float*         scales,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
