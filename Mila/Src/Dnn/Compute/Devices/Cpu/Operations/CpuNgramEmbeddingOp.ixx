/**
 * @file CpuNgramEmbeddingOp.ixx
 * @brief CPU implementation of the hashed n-gram embedding: the hash, its id history, and the table gather (FP32).
 *
 * The reference behind NgramEmbedding<Cpu>, and the one the CUDA kernels are checked against.
 */

module;
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

export module Compute.CpuNgramEmbeddingOp;

import Dnn.Components.NgramEmbeddingConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Hashes each token's n-grams to table rows, and gathers them.
     *
     * For each position, with history the previous ngram_size - 1 ids:
     *
     *   shift_s  = the id s positions back, or eos when an EOS lies among the s ids before this one
     *              (an EOS belongs to the segment it ends, so it is never itself a reason to cut)
     *   mixed_k  = (shift_0 * m_0) XOR ... XOR (shift_{k-1} * m_{k-1})       for order k = 2 .. ngram_size
     *   id       = floor_mod( mixed_k, P_j ) + offset_j                       for each head j of order k
     *
     * The products wrap as two's-complement int64, which is what the reference's int64 tensors do: they are
     * computed in uint64 and reinterpreted. The modulus is a floor mod -- C++ % keeps a negative dividend's sign,
     * torch.remainder does not -- and an id that is off by one row is a quietly worse model, not a crash.
     */
    export class CpuNgramEmbeddingOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ConfigType = NgramEmbeddingConfig;

        CpuNgramEmbeddingOp( IExecutionContext* context, const NgramEmbeddingConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuNgramEmbeddingOp requires a CPU execution context" );
            }

            config_.validate();
        }

        void setParameters( ITensor* table, ITensor* /*bias*/ ) override
        {
            table_ = table ? static_cast<const float*>( table->rawData() ) : nullptr;
        }

        /**
         * @brief Table rows for every head of every position, then the history advanced past this chunk.
         *
         * @param token_ids [B, T] INT32.
         * @param history   [B, ngram_size - 1] INT32: the ids before this chunk, eos where there were none.
         * @param ngram_ids [B, T, heads] INT32.
         */
        void hash( const ITensor& token_ids, ITensor& history, ITensor& ngram_ids ) const
        {
            const auto& shape = token_ids.shape();

            if ( shape.size() != 2 )
            {
                throw std::runtime_error( "CpuNgramEmbeddingOp::hash - token ids must be [B, T]" );
            }

            const int64_t batch = static_cast<int64_t>( shape[ 0 ] );
            const int64_t length = static_cast<int64_t>( shape[ 1 ] );
            const int64_t history_length = static_cast<int64_t>( config_.getHistoryLength() );
            const int64_t heads = static_cast<int64_t>( config_.getHeads() );
            const int64_t ngram_size = static_cast<int64_t>( config_.getNgramSize() );
            const int64_t heads_per_ngram = static_cast<int64_t>( config_.getHeadsPerNgram() );
            const int64_t eos = config_.getEosTokenId();

            const auto& multipliers = config_.getMultipliers();
            const auto& sizes = config_.getHeadVocabSizes();
            const auto& offsets = config_.getHeadOffsets();

            const auto* ids = static_cast<const int32_t*>( token_ids.rawData() );
            auto* state = static_cast<int32_t*>( history.rawData() );
            auto* out = static_cast<int32_t*>( ngram_ids.rawData() );

            std::vector<int64_t> sequence( static_cast<size_t>( history_length + length ) );
            std::vector<int64_t> shifted( static_cast<size_t>( ngram_size ) );

            for ( int64_t b = 0; b < batch; ++b )
            {
                for ( int64_t j = 0; j < history_length; ++j )
                {
                    sequence[ static_cast<size_t>( j ) ] = state[ b * history_length + j ];
                }

                for ( int64_t t = 0; t < length; ++t )
                {
                    sequence[ static_cast<size_t>( history_length + t ) ] = ids[ b * length + t ];
                }

                for ( int64_t t = 0; t < length; ++t )
                {
                    const int64_t p = history_length + t;

                    for ( int64_t s = 0; s < ngram_size; ++s )
                    {
                        shifted[ static_cast<size_t>( s ) ] = shiftedId( sequence, p, s, eos );
                    }

                    for ( int64_t order = 2; order <= ngram_size; ++order )
                    {
                        uint64_t mixed = static_cast<uint64_t>( shifted[ 0 ] ) * static_cast<uint64_t>( multipliers[ 0 ] );

                        for ( int64_t s = 1; s < order; ++s )
                        {
                            mixed ^= static_cast<uint64_t>( shifted[ static_cast<size_t>( s ) ] )
                                * static_cast<uint64_t>( multipliers[ static_cast<size_t>( s ) ] );
                        }

                        const int64_t signed_mixed = static_cast<int64_t>( mixed );

                        for ( int64_t head = 0; head < heads_per_ngram; ++head )
                        {
                            const size_t global_head = static_cast<size_t>( ( order - 2 ) * heads_per_ngram + head );
                            const int64_t prime = sizes[ global_head ];
                            int64_t row = signed_mixed % prime;

                            if ( row < 0 )
                            {
                                row += prime;
                            }

                            out[ ( b * length + t ) * heads + static_cast<int64_t>( global_head ) ] =
                                static_cast<int32_t>( row + offsets[ global_head ] );
                        }
                    }
                }

                for ( int64_t j = 0; j < history_length; ++j )
                {
                    state[ b * history_length + j ] = static_cast<int32_t>( sequence[ static_cast<size_t>( length + j ) ] );
                }
            }
        }

        /// out[ b, t, head * row_width + c ] = table[ ngram_ids[ b, t, head ], c ].
        void gather( const ITensor& ngram_ids, ITensor& output ) const
        {
            if ( !table_ )
            {
                throw std::runtime_error( "CpuNgramEmbeddingOp::gather - table not set" );
            }

            const int64_t row_width = static_cast<int64_t>( config_.getRowWidth() );
            const int64_t rows = static_cast<int64_t>( config_.getTableRows() );
            const int64_t count = static_cast<int64_t>( ngram_ids.size() );
            const auto* ids = static_cast<const int32_t*>( ngram_ids.rawData() );
            auto* out = static_cast<float*>( output.rawData() );

            if ( static_cast<int64_t>( output.size() ) != count * row_width )
            {
                throw std::runtime_error( "CpuNgramEmbeddingOp::gather - output is not [..., heads * row_width]" );
            }

            for ( int64_t i = 0; i < count; ++i )
            {
                const int64_t row = ids[ i ];

                if ( row < 0 || row >= rows )
                {
                    throw std::runtime_error( "CpuNgramEmbeddingOp::gather - n-gram id outside the table" );
                }

                std::memcpy( out + i * row_width, table_ + row * row_width, static_cast<size_t>( row_width ) * sizeof( float ) );
            }
        }

        void build( const BuildContext& build_context ) override
        {
            Operation<DeviceType::Cpu, TensorDataType::FP32>::build( build_context );
        }

        OperationType getOperationType() const override
        {
            return OperationType::NgramEmbeddingOp;
        }

        std::string getName() const override
        {
            return "Cpu::NgramEmbeddingOp";
        }

    private:
        // The id s positions before p, or eos when one of the s ids before p is an EOS.
        static int64_t shiftedId( const std::vector<int64_t>& sequence, int64_t p, int64_t s, int64_t eos )
        {
            for ( int64_t back = 1; back <= s; ++back )
            {
                if ( sequence[ static_cast<size_t>( p - back ) ] == eos )
                {
                    return eos;
                }
            }

            return sequence[ static_cast<size_t>( p - s ) ];
        }

        IExecutionContext* context_;
        NgramEmbeddingConfig config_;

        const float* table_{ nullptr };
    };
}
