#pragma once

// Host launchers for the device-side next-token log-probability: one block per row of logits, each row
// reduced to the log-probability it gives the token that actually followed it.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace Mila::Dnn::Compute::Cuda::LogLikelihood
{
    // log_probabilities[ r ] = log softmax( softcap( logits[ r, : ] ) )[ tokens[ r + 1 ] ] for r < rows, where
    // `tokens` points at the token row 0 predicts from, so tokens[ r + 1 ] is row r's target. softcap <= 0 means
    // none. The row maximum is subtracted before exponentiating and the exponentials are summed in double; the
    // result is written as FP32 into `log_probabilities`, which may be host memory the device can address
    // (pinned). A target outside [0, vocab) writes NaN.
    void cuda_next_token_log_probability_fp32(
        const float* logits, const int32_t* tokens, float* log_probabilities,
        int rows, int vocab, float softcap, cudaStream_t stream );
    void cuda_next_token_log_probability_bf16(
        const __nv_bfloat16* logits, const int32_t* tokens, float* log_probabilities,
        int rows, int vocab, float softcap, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_next_token_log_probability(
        const TNative* logits, const int32_t* tokens, float* log_probabilities,
        int rows, int vocab, float softcap, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_next_token_log_probability: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_next_token_log_probability_fp32( logits, tokens, log_probabilities, rows, vocab, softcap, stream );
        else
            cuda_next_token_log_probability_bf16( logits, tokens, log_probabilities, rows, vocab, softcap, stream );
    }
}
