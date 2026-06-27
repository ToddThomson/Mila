#pragma once

// Host launchers for device-side token sampling. Phase A implements the greedy
// (argmax) reduction only; stochastic top-k/top-p land in later phases. See
// Specifications/TokenSampling.md.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace Mila::Dnn::Compute::Cuda::Sampling
{
    // Argmax over `vocab` logits; writes the winning index as int32 to token_out[0].
    // Ties resolve to the lowest index, matching the host std::max_element baseline.
    void cuda_sample_argmax_fp32( const float* logits, int32_t* token_out, int vocab, cudaStream_t stream );
    void cuda_sample_argmax_bf16( const __nv_bfloat16* logits, int32_t* token_out, int vocab, cudaStream_t stream );

    // Header-visible templated dispatcher.
    template <typename TNative>
    inline void cuda_sample_argmax( const TNative* logits, int32_t* token_out, int vocab, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_argmax: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_argmax_fp32( logits, token_out, vocab, stream );
        else
            cuda_sample_argmax_bf16( logits, token_out, vocab, stream );
    }

    // Stochastic multinomial: applies the optional logit softcap (softcap*tanh(x/softcap),
    // skipped when softcap <= 0) and temperature, takes the FP32 softmax, optionally truncates
    // to top-k (0 disables) and/or top-p (>= 1 disables) via threshold binary search, then
    // inverse-CDF samples against the host-drawn uniform `r`. `scratch` is a caller-owned device
    // buffer of `vocab` floats (working store). top_k == 0 && top_p == 1 is the full multinomial.
    void cuda_sample_stochastic_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );
    void cuda_sample_stochastic_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_sample_stochastic(
        const TNative* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_stochastic: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_stochastic_fp32( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
        else
            cuda_sample_stochastic_bf16( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }
}
