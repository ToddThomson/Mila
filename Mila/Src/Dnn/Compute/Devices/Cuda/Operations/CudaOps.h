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
}