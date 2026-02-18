module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/CudaAttention.cuh"

export module Compute.CudaAttentionOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Attention
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for attention non-matmul operations.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_mha_kernels;

        template <>
        struct cuda_mha_kernels<float>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                float* q, float* k, float* v,
                const float* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp32( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void permute_qkv_padded(
                float* q, float* k, float* v,
                const float* inp,
                int B, int input_T, int output_T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_padded_fp32( q, k, v, inp, B, input_T, output_T, NH, HS, stream );
            }

            static inline void permute_qkv_decode(
                float* q, float* k, float* v,
                const float* inp,
                int B, int position, int cache_T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_decode_fp32( q, k, v, inp, B, position, cache_T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const float* vaccum, float* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp32( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp32( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_padded_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int max_T, int actual_T,
                cudaStream_t stream )
            {
                cuda_softmax_padded_forward_fp32( att, scale, preatt, B, NH, max_T, actual_T, stream );
            }

            static inline void softmax_decode_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int max_len, int actual_len,
                cudaStream_t stream )
            {
                cuda_softmax_decode_forward_fp32( att, scale, preatt, B, NH, max_len, actual_len, stream );
            }

            static inline void softmax_backward(
                float* dpreatt, const float* datt, const float* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp32( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                float* dinp,
                const float* dq, const float* dk, const float* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_backward_fp32( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                float* dvaccum, const float* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp32( dvaccum, dout, B, T, NH, HS, stream );
            }
        };

        template <>
        struct cuda_mha_kernels<half>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                half* q, half* k, half* v,
                const half* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp16( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void permute_qkv_padded(
                half* q, half* k, half* v,
                const half* inp,
                int B, int input_T, int output_T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_padded_fp16( q, k, v, inp, B, input_T, output_T, NH, HS, stream );
            }

            static inline void permute_qkv_decode(
                half* q, half* k, half* v,
                const half* inp,
                int B, int position, int cache_T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_decode_fp16( q, k, v, inp, B, position, cache_T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const half* vaccum, half* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp16( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp16( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_padded_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int max_T, int actual_T,
                cudaStream_t stream )
            {
                cuda_softmax_padded_forward_fp16( att, scale, preatt, B, NH, max_T, actual_T, stream );
            }

            static inline void softmax_decode_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int max_len, int actual_len,
                cudaStream_t stream )
            {
                cuda_softmax_decode_forward_fp16( att, scale, preatt, B, NH, max_len, actual_len, stream );
            }

            static inline void softmax_backward(
                half* dpreatt, const half* datt, const half* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp16( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                half* dinp,
                const half* dq, const half* dk, const half* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_backward_fp16( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                half* dvaccum, const half* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp16( dvaccum, dout, B, T, NH, HS, stream );
            }
        };
    }
}