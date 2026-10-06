#pragma once

// Host launchers for device-side token sampling. Greedy argmax plus the stochastic
// multinomial with optional top-k / top-p truncation. The stochastic path is a
// multi-block kernel pipeline (histogram threshold refinement + chunked inverse-CDF);
// the original single-block kernel is retained as the parity reference oracle.
// See Specifications/TokenSampling.md.

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

    // The most rows one call samples: a verify's K + 1 (Gemma4Mtp.md 4.8).
    inline constexpr int kMaxSampleRows = 8;

    // One host-drawn uniform per row, passed by value so a call carries every row's draw.
    struct SampleRowUniforms
    {
        float value[ kMaxSampleRows ];
    };

    // Argmax of each of `rows` contiguous rows of `vocab` logits into token_out[row], one block per row.
    void cuda_sample_rows_argmax_fp32( const float* logits, int32_t* token_out, int rows, int vocab, cudaStream_t stream );
    void cuda_sample_rows_argmax_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, int rows, int vocab, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_sample_rows_argmax(
        const TNative* logits, int32_t* token_out, int rows, int vocab, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_rows_argmax: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_rows_argmax_fp32( logits, token_out, rows, vocab, stream );
        else
            cuda_sample_rows_argmax_bf16( logits, token_out, rows, vocab, stream );
    }

    // Stochastic pipeline scratch geometry. The op allocates one FP32 and one INT32
    // device tensor of these element counts and passes their raw pointers through;
    // the layout (header slots, reduction partials, histogram bins, chunk partials)
    // is private to Sampling.cu.
    inline constexpr int kStochasticBlock = 256;
    inline constexpr int kStochasticMaxChunks = 1024;
    inline constexpr int kStochasticBins = 4096;
    inline constexpr int kStochasticHeaderFloats = 16;
    inline constexpr int kStochasticFloatScratchElements =
        kStochasticHeaderFloats + 2 * kStochasticMaxChunks + kStochasticBins + kStochasticMaxChunks;
    inline constexpr int kStochasticIndexScratchElements = 16 + kStochasticBins;

    // Stochastic multinomial: applies the optional logit softcap (softcap*tanh(x/softcap),
    // skipped when softcap <= 0) and temperature, takes the FP32 softmax, optionally truncates
    // to top-k (0 disables) and/or top-p (>= 1 disables) via histogram threshold refinement,
    // then inverse-CDF samples against the host-drawn uniform `r` in token-index order.
    // `scratch` is a caller-owned device buffer of `vocab` floats (working store);
    // `reduction_scratch` / `index_scratch` are caller-owned device buffers sized by the
    // kStochastic*ScratchElements constants above. top_k == 0 && top_p == 1 is the full
    // multinomial. The whole pipeline is enqueued on `stream` with no host round-trip.
    void cuda_sample_stochastic_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );
    void cuda_sample_stochastic_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_sample_stochastic(
        const TNative* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_stochastic: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_stochastic_fp32( logits, token_out, scratch, reduction_scratch, index_scratch,
                vocab, softcap, temperature, top_k, top_p, r, stream );
        else
            cuda_sample_stochastic_bf16( logits, token_out, scratch, reduction_scratch, index_scratch,
                vocab, softcap, temperature, top_k, top_p, r, stream );
    }

    // The stochastic pipeline over `rows` contiguous rows of `vocab` logits at once, each launch serving every row:
    // row r is sampled against uniforms.value[r] into token_out[r], exactly as the one-row call samples that row.
    // `scratch`, `reduction_scratch` and `index_scratch` hold `rows` times the one-row element counts.
    void cuda_sample_rows_stochastic_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch, int rows,
        int vocab, float softcap, float temperature, int top_k, float top_p,
        const SampleRowUniforms& uniforms, cudaStream_t stream );
    void cuda_sample_rows_stochastic_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch, int rows,
        int vocab, float softcap, float temperature, int top_k, float top_p,
        const SampleRowUniforms& uniforms, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_sample_rows_stochastic(
        const TNative* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch, int rows,
        int vocab, float softcap, float temperature, int top_k, float top_p,
        const SampleRowUniforms& uniforms, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_rows_stochastic: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_rows_stochastic_fp32( logits, token_out, scratch, reduction_scratch, index_scratch, rows,
                vocab, softcap, temperature, top_k, top_p, uniforms, stream );
        else
            cuda_sample_rows_stochastic_bf16( logits, token_out, scratch, reduction_scratch, index_scratch, rows,
                vocab, softcap, temperature, top_k, top_p, uniforms, stream );
    }

    // Speculative acceptance over a verified round (Gemma4Mtp.md 4.8). `tokens` holds the round's `rows` tokens --
    // the known one, then the drafts -- and `chosen` the token sampled at each row. A draft is kept while the row
    // before it chose it. Writes result[0] = kept count m, result[1..m] = the kept drafts, result[m + 1] = the token
    // chosen at row m, and that next token into tokens[0] for the next round.
    void cuda_speculative_accept(
        int32_t* tokens, const int32_t* chosen, int rows, int32_t* result, cudaStream_t stream );

    // Reference oracle: the original single-block stochastic kernel (bisection thresholds
    // + thread-0 serial inverse-CDF). Retained for new-vs-reference parity tests only —
    // ~11 ms/token at a 262k vocab, never on the production path. Same semantics as the
    // pipeline above up to float reduction order at truncation/CDF boundaries.
    void cuda_sample_stochastic_reference_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );
    void cuda_sample_stochastic_reference_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream );

    template <typename TNative>
    inline void cuda_sample_stochastic_reference(
        const TNative* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        static_assert( std::is_same_v<TNative, float> || std::is_same_v<TNative, __nv_bfloat16>,
                       "cuda_sample_stochastic_reference: unsupported precision" );

        if constexpr ( std::is_same_v<TNative, float> )
            cuda_sample_stochastic_reference_fp32( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
        else
            cuda_sample_stochastic_reference_bf16( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }
}
