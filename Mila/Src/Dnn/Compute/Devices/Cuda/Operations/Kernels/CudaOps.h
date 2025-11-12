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
 * - Fused Softmax Cross Entropy operations
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
    // Reduction kernels
    void cuda_reduce_sum_batch_fp32(
        float* dBias,
        const float* dY,
        int outer_size,
        int out_features,
        cudaStream_t stream );

    /* TODO: void cuda_reduce_sum_batch_fp16(
        half* dBias,
        const half* dY,
        int batch_size,
        int out_features,
        cudaStream_t stream );

    void cuda_reduce_sum_batch_bfp16(
        __nv_bfloat16* dBias,
        const __nv_bfloat16* dY,
        int batch_size,
        int out_features,
        cudaStream_t stream );*/
    
    // Attention functions
    void cuda_mha_forward_fp32(
        float* Y,
        float* qkvr, float* att,
        const float* X,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    void cuda_mha_forward_fp16(
        half* Y,
        half* qkvr, half* att,
        const half* X,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    // Encoder functions
    void cuda_encoder_forward_fp32(
        float* Y, const int* X,
        const float* wte, const float* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    void cuda_encoder_forward_fp16(
        half* Y, const int* X,
        const half* wte, const half* wpe,
        int B, int T, int C,
        cudaStream_t stream );

    // GELU functions
    void cuda_gelu_forward_fp32(
        float* Y,
        const float* X,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp32(
        float* dX,
        const float* X,
        const float* dY,
        const int N,
        cudaStream_t stream );

    void cuda_gelu_forward_fp16(
        half* Y,
        const half* X,
        int N,
        cudaStream_t stream );

    void cuda_gelu_backward_fp16(
        half* dX,
        const half* X,
        const half* dY,
        const int N,
        cudaStream_t stream );

    // LayerNorm functions
    void cuda_layernorm_forward_fp32(
        float* Y,
        float* mean, float* rstd,
        const float* X,
        const float* weight, const float* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream );

    void cuda_layernorm_forward_fp16(
        half* Y,
        half* mean, half* rstd,
        const half* X,
        const half* weight, const half* bias,
        int B, int T, int C, float epsilon,
        cudaStream_t stream );

    // Matmul functions
    void cuda_matmul_forward_fp32(
        float* Y,
        const float* X,
        const float* weight, const float* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream );

    void cuda_matmul_forward_fp16(
        half* Y, const half* X,
        const half* weight, const half* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream );

    // Softmax functions
    template <typename TPrecision>
    void cuda_softmax_forward(
        TPrecision* Y,
        const TPrecision* X,
        int N,
        int C,
        cudaStream_t stream );

    template <typename TPrecision>
    void cuda_softmax_forward_general(
        TPrecision* Y,
        const TPrecision* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    // SoftmaxCrossEntropy functions
    template <typename TPrecision>
    void cuda_softmax_crossentropy_forward(
        TPrecision* Y_loss,
        TPrecision* Y,
        const TPrecision* X,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    template <typename TPrecision>
    void cuda_softmax_crossentropy_backward(
        TPrecision* dX,
        const TPrecision* dY_loss,
        const TPrecision* Y,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    // Residual functions
    void cuda_residual_forward_fp32(
        float* Y,
        const float* X1, const float* X2,
        int N,
        cudaStream_t stream );

    /**
     * @brief Host function to launch residual backward kernel with full precision (FP32)
     *
     * Propagates gradients through the residual connection. Both inputs receive
     * the same gradient since the forward pass is simple addition.
     *
     * Formula: dX1 += dY, dX2 += dY
     *
     * @param dX1 Gradient tensor for first input (accumulated)
     * @param dX2 Gradient tensor for second input (accumulated)
     * @param dY Gradient from downstream layers
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_backward_fp32(
        float* dX1,
        float* dX2,
        const float* dY,
        int N,
        cudaStream_t stream );

    void cuda_residual_forward_fp16(
        half* Y,
        const half* X1, const half* X2,
        int N,
        cudaStream_t stream );

    /**
     * @brief Host function to launch residual backward kernel with half precision (FP16)
     *
     * Propagates gradients through the residual connection using half-precision arithmetic.
     * Both inputs receive the same gradient.
     *
     * Formula: dX1 += dY, dX2 += dY
     *
     * @param dX1 Gradient tensor for first input in half precision (accumulated)
     * @param dX2 Gradient tensor for second input in half precision (accumulated)
     * @param dY Gradient from downstream layers in half precision
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_backward_fp16(
        half* dX1,
        half* dX2,
        const half* dY,
        int N,
        cudaStream_t stream );
}