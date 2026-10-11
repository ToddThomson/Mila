/**
 * @file CudaNgramEmbeddingOp.ixx
 * @brief CUDA implementation of the hashed n-gram embedding: the hash with its id history, and the gather.
 *
 * The hash and the gather are separate launches so that where the table lives decides only the gather.
 */

module;
#include "Kernels/NgramEmbedding.cuh"
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

export module Compute.CudaNgramEmbeddingOp;

import Dnn.Components.NgramEmbeddingConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Embedding
{
    /**
     * @brief Hashes each token's n-grams to table rows, advances the id history, and gathers the rows.
     *
     * The hash reads its token ids and history from device memory and passes its constants by value, so a recorded
     * decode step replays unchanged.
     */
    export template<TensorDataType TPrecision>
    class CudaNgramEmbeddingOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = NgramEmbeddingConfig;

        CudaNgramEmbeddingOp( IExecutionContext* context, const NgramEmbeddingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaNgramEmbeddingOp" ) ), config_( config )
        {
            config_.validate();

            constants_ = packConstants( config_ );
        }

        void setParameters( ITensor* table, ITensor* /*bias*/ ) override
        {
            table_ = table;
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
                throw std::runtime_error( "CudaNgramEmbeddingOp::hash - token ids must be [B, T]" );
            }

            cuda_ngram_hash(
                static_cast<const std::int32_t*>( token_ids.rawData() ),
                static_cast<std::int32_t*>( history.rawData() ),
                static_cast<std::int32_t*>( ngram_ids.rawData() ),
                narrowToKernelIndex( shape[ 0 ] ), narrowToKernelIndex( shape[ 1 ] ),
                constants_, context_->getStream() );
        }

        /// out[ b, t, head * row_width + c ] = table[ ngram_ids[ b, t, head ], c ].
        void gather( const ITensor& ngram_ids, ITensor& output ) const
        {
            if ( !table_ )
            {
                throw std::runtime_error( "CudaNgramEmbeddingOp::gather - table not set" );
            }

            const dim_t row_width = config_.getRowWidth();

            if ( output.size() != ngram_ids.size() * row_width )
            {
                throw std::runtime_error( "CudaNgramEmbeddingOp::gather - output is not [..., heads * row_width]" );
            }

            cuda_ngram_gather(
                static_cast<NativeType*>( output.rawData() ),
                static_cast<const NativeType*>( table_->rawData() ),
                static_cast<const std::int32_t*>( ngram_ids.rawData() ),
                narrowToKernelIndex( ngram_ids.size() ), narrowToKernelIndex( row_width ),
                narrowToKernelIndex( config_.getTableRows() ), context_->getStream() );
        }

        void build( const BuildContext& build_context ) override
        {
            Operation<DeviceType::Cuda, TPrecision>::build( build_context );
        }

        OperationType getOperationType() const override
        {
            return OperationType::NgramEmbeddingOp;
        }

        std::string getName() const override
        {
            return "Cuda::NgramEmbeddingOp";
        }

    private:
        static NgramHashConstants packConstants( const NgramEmbeddingConfig& config )
        {
            const auto& multipliers = config.getMultipliers();
            const auto& sizes = config.getHeadVocabSizes();
            const auto& offsets = config.getHeadOffsets();

            if ( config.getNgramSize() > kMaxNgramSize || config.getHeads() > kMaxNgramHeads )
            {
                throw std::invalid_argument( std::format(
                    "CudaNgramEmbeddingOp: n-gram size {} with {} heads exceeds the CUDA hash's limits of {} and {}",
                    config.getNgramSize(), config.getHeads(), kMaxNgramSize, kMaxNgramHeads ) );
            }

            NgramHashConstants constants{};

            for ( std::size_t i = 0; i < multipliers.size(); ++i )
            {
                constants.multipliers[ i ] = multipliers[ i ];
            }

            for ( std::size_t i = 0; i < sizes.size(); ++i )
            {
                constants.head_vocab_sizes[ i ] = sizes[ i ];
                constants.head_offsets[ i ] = offsets[ i ];
            }

            constants.eos_token_id = config.getEosTokenId();
            constants.ngram_size = static_cast<int>( config.getNgramSize() );
            constants.heads_per_ngram = static_cast<int>( config.getHeadsPerNgram() );
            constants.heads = static_cast<int>( config.getHeads() );
            constants.history_length = static_cast<int>( config.getHistoryLength() );

            return constants;
        }

        CudaExecutionContext* context_{ nullptr };
        NgramEmbeddingConfig config_;
        NgramHashConstants constants_{};

        ITensor* table_{ nullptr };
    };
}
