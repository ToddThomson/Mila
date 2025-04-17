#pragma once
#include <cuda_runtime.h>

void cuda_encoder_forward(
    float* out, const int* inp,
    const float* wte, const float* wpe,
    int B, int T, int C,
    cudaStream_t stream );

void cuda_gelu_forward(
    float* out, const float* inp,
    int N,
    cudaStream_t stream );

void cuda_layernorm_forward(
    float* out,
    float* mean, float* rstd,
    const float* inp,
    const float* weight, const float* bias,
    int B, int T, int C,
    cudaStream_t  stream);

void cuda_matmul_forward(
    float* out,
    const float* inp,
    const float* weight, const float* bias,
    int B, int T, int C, int OC,
    cudaStream_t stream );

void cuda_residual_forward(
    float* out,
    const float* inp1, const float* inp2,
    int N,
    cudaStream_t stream );

void cuda_softmax_forward(
    float* out,
    const float* inp,
    int N, int C,
    cudaStream_t stream );
