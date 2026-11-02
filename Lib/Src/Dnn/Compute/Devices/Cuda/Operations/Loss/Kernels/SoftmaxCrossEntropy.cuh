// Minimal header for SoftmaxCrossEntropy host launchers (FP32/FP16).

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace Mila::Dnn::Compute::Cuda::SoftmaxCrossEntropy
{
    // Threshold for switching between warp and block algorithms
    constexpr int WARP_VOCAB_THRESHOLD = 1024;
    inline constexpr int WARP_SIZE = 32;

    void cuda_softmax_crossentropy_forward_fp32(
        float* losses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );
    
    void cuda_softmax_crossentropy_backward_fp32(
        float* dlogits,
        const float* dlosses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream );

    __global__ void softmax_crossentropy_forward_fp32_warp_kernel(
        float* losses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size );

    __global__ void softmax_crossentropy_forward_fp32_block_kernel(
        float* losses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size );

    __global__ void softmax_crossentropy_backward_fp32_warp_kernel(
        float* dlogits,
        const float* dlosses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size );

    __global__ void softmax_crossentropy_backward_fp32_block_kernel(
        float* dlogits,
        const float* dlosses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size );
}