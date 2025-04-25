/**
 * @file CudaOps.h
 * @brief CUDA kernel function declarations for neural network operations.
 *
 * This header file declares CUDA kernel functions that implement various neural network operations
 * optimized for execution on NVIDIA GPUs. The operations include:
 * - Encoder operations for embedding input tokens
 * - GELU activation functions
 * - Layer normalization
 * - Matrix multiplication
 * - Multi-head attention
 * - Softmax activation
 * - Residual connections
 * - Attention mechanisms
 *
 * Each operation provides implementations for both single-precision (fp32) and half-precision (fp16)
 * floating point data types to support different performance and accuracy requirements.
 * These functions serve as the computational backend for the neural network operations
 * in the Mila deep learning framework.
 */
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute
{
    // Attention functions
    void cuda_mha_forward_fp32(
        float* out,
        float* qkvr, float* att,
        const float* inp,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    void cuda_mha_forward_fp16(
        half* out,
        half* qkvr, half* att,
        const half* inp,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    // Encoder functions
    void cuda_encoder_forward_fp32(
        float* out, const int* inp,
        const float* wte, const float* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    void cuda_encoder_forward_fp16(
        half* out, const int* inp,
        const half* wte, const half* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    // GELU functions
    void cuda_gelu_forward_fp32(
        float* out,
        const float* inp,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp32(
        float* dinp,
        const float* inp,
        const float* dout,
        const int N,
        cudaStream_t stream );

    void cuda_gelu_forward_fp16(
        half* out,
        const half* inp,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp16(
        half* dinp,
        const half* inp,
        const half* dout,
        const int N,
        cudaStream_t stream );

	// LayerNorm functions
    void cuda_layernorm_forward_fp32(
        float* out,
        float* mean, float* rstd,
        const float* inp,
        const float* weight, const float* bias,
		int B, int T, int C, float epsilon,
        cudaStream_t stream );

    void cuda_layernorm_forward_fp16(
        half* out,
        half* mean, half* rstd,
        const half* inp,
        const half* weight, const half* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream );
    
	// Matmul functions
    void cuda_matmul_forward_fp32(
        float* out,
        const float* inp,
        const float* weight, const float* bias,
        int B, int T, int C, int OC,
        cudaStream_t stream );

	void cuda_matmul_forward_fp16(
		half* out, const half* inp,
		const half* weight, const half* bias,
		int B, int T, int C, int OC,
		cudaStream_t stream );

    // Softmax functions
    void cuda_softmax_forward_fp32(
        float* output,
        const float* input,
        int N, int C,
        cudaStream_t stream );

    void cuda_softmax_forward_fp16(
        half* output,
        const half* input,
        int N, int C,
        cudaStream_t stream );

    // Residual functions
    void cuda_residual_forward_fp32(
        float* out,
        const float* inp1, const float* inp2,
        int N,
        cudaStream_t stream );

    void cuda_residual_forward_fp16(
        half* out,
        const half* inp1, const half* inp2,
        int N,
        cudaStream_t stream );
}