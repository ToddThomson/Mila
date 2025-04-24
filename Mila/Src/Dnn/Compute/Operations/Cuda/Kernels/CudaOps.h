#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute
{
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
    
	// Matmul functions
    void cuda_matmul_forward_fp32(
        float* out,
        const float* inp,
        const float* weight, const float* bias,
        int B, int T, int C, int OC,
        cudaStream_t stream );

	void cuda_matmul_forward_fp16(
		half* out,
		const half* inp,
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

void cuda_layernorm_forward(
    float* out,
    float* mean, float* rstd,
    const float* inp,
    const float* weight, const float* bias,
    int B, int T, int C,
    cudaStream_t  stream );

void cuda_attention_forward(
    float* out,
    float* qkvr, float* att,
    const float* inp,
    int B, int T, int C, int NH,
    cudaStream_t stream );
