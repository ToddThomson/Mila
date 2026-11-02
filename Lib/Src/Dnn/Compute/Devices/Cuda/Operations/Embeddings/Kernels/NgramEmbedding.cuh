#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Embedding
{
    inline constexpr int kMaxNgramSize = 8;
    inline constexpr int kMaxNgramHeads = 32;

    /**
     * @brief The n-gram hash's constants, passed by value as a kernel argument.
     *
     * They are configuration that never changes between calls, so a recorded decode step replays them unchanged,
     * and they occupy no device memory.
     */
    struct NgramHashConstants
    {
        std::int64_t multipliers[ kMaxNgramSize ];
        std::int64_t head_vocab_sizes[ kMaxNgramHeads ];
        std::int64_t head_offsets[ kMaxNgramHeads ];
        std::int64_t eos_token_id;
        int ngram_size;
        int heads_per_ngram;
        int heads;
        int history_length;
    };

    /**
     * @brief Table rows for every head of every position, and the id history advanced past the chunk.
     *
     * For each position, with the previous ngram_size - 1 ids as history:
     *
     *   shift_s  = the id s positions back, or eos when an EOS lies among the s ids before this one
     *   mixed_k  = (shift_0 * m_0) XOR ... XOR (shift_{k-1} * m_{k-1})       for order k = 2 .. ngram_size
     *   id       = floor_mod( mixed_k, P_j ) + offset_j                       for each head j of order k
     *
     * The products wrap in uint64 and are reinterpreted as int64, and the modulus is a floor mod. One block per
     * batch row: the block reads the history before any thread writes it, so the update is in place.
     *
     * @param token_ids [B, T]; @param history [B, history_length], read then overwritten; @param ngram_ids [B, T, heads].
     */
    void cuda_ngram_hash( const std::int32_t* token_ids, std::int32_t* history, std::int32_t* ngram_ids,
        int batch, int length, const NgramHashConstants& constants, cudaStream_t stream );

    /**
     * @brief out[ i, c ] = table[ ngram_ids[ i ], c ], copied without conversion.
     *
     * An id outside the table writes NaN for that row: a kernel cannot throw, and a valid configuration never
     * produces one.
     */
    void cuda_ngram_gather( float* output, const float* table, const std::int32_t* ngram_ids,
        int count, int row_width, int table_rows, cudaStream_t stream );
    void cuda_ngram_gather( __nv_bfloat16* output, const __nv_bfloat16* table, const std::int32_t* ngram_ids,
        int count, int row_width, int table_rows, cudaStream_t stream );
}
