/**
 * @file CpuSamplingOp.ixx
 * @brief CPU token sampling operation (FP32).
 *
 * Phase A: greedy (argmax) sampling on the host. See Specifications/TokenSampling.md.
 */

module;
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <functional>
#include <format>
#include <span>

export module Compute.CpuSamplingOp;

import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU token sampler op: maps a logits row to a single int32 token id.
     *
     * Reads the final `vocab_size` FP32 logits and, for the greedy branch, returns
     * the argmax (lowest index on ties, matching the host baseline).
     */
    export class CpuSamplingOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        CpuSamplingOp( IExecutionContext* context, const SamplingConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuSamplingOp requires a CPU execution context" );
            }

            config_.validate();
        }

        void forward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            const int64_t vocab = config_.getVocabularySize();
            const dim_t offset = logits.size() - vocab;

            const float* row = static_cast<const float*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

            out[ 0 ] = sampleRow( row, params, r );
        }

        /**
         * @brief Enqueue-contract mirror of the CUDA op for the decode-ahead loop.
         *
         * CPU execution is synchronous, so this computes immediately and stashes the
         * token; awaitToken() just returns it. Keeps the pipelined generation loop
         * device-agnostic.
         */
        void enqueueForward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            forward( logits, token_out, params, r );
            pending_token_ = static_cast<const int32_t*>( token_out.rawData() )[ 0 ];
        }

        int32_t awaitToken() const
        {
            return pending_token_;
        }

        /// Mirror of the CUDA op's device-only step: computed now, and the readback slot left alone.
        void enqueueForwardOnDevice(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            forward( logits, token_out, params, r );
        }

        /// Mirror of the CUDA op's rows step: row r of the first @p rows rows is drawn against `uniforms[r]`.
        void enqueueRowsOnDevice(
            const ITensor& logits,
            ITensor& tokens_out,
            dim_t rows,
            const SamplingParams& params,
            std::span<const float> uniforms ) const
        {
            const int64_t vocab = config_.getVocabularySize();

            if ( rows < 1 || rows > config_.getMaximumRows() || static_cast<dim_t>( uniforms.size() ) < rows
                || logits.size() < rows * vocab || tokens_out.size() < rows )
            {
                throw std::invalid_argument( std::format(
                    "CpuSamplingOp::enqueueRowsOnDevice: {} rows, where this sampler was built for 1 to {}, need {} "
                    "uniforms, {} logits and {} token slots; given {}, {} and {}",
                    rows, config_.getMaximumRows(), rows, rows * vocab, rows,
                    uniforms.size(), logits.size(), tokens_out.size() ) );
            }

            const float* first_row = static_cast<const float*>( logits.rawData() );
            int32_t* out = static_cast<int32_t*>( tokens_out.rawData() );

            for ( dim_t row = 0; row < rows; ++row )
                out[ row ] = sampleRow( first_row + row * vocab, params, uniforms[ static_cast<size_t>( row ) ] );
        }

        /// Mirror of the CUDA op's acceptance walk: writes `[m, d_1 .. d_m, next]` and the next token into slot 0.
        void enqueueAcceptOnDevice(
            ITensor& tokens,
            const ITensor& chosen,
            dim_t rows,
            ITensor& result ) const
        {
            if ( rows < 1 || rows > config_.getMaximumRows() || tokens.size() < rows || chosen.size() < rows
                || result.size() < rows + 1 )
            {
                throw std::invalid_argument( std::format(
                    "CpuSamplingOp::enqueueAcceptOnDevice: {} rows, where this sampler was built for 1 to {}, need {} "
                    "tokens, {} chosen and {} result slots; given {}, {} and {}",
                    rows, config_.getMaximumRows(), rows, rows, rows + 1, tokens.size(), chosen.size(), result.size() ) );
            }

            int32_t* round = static_cast<int32_t*>( tokens.rawData() );
            const int32_t* picked = static_cast<const int32_t*>( chosen.rawData() );
            int32_t* out = static_cast<int32_t*>( result.rawData() );

            dim_t accepted = 0;

            while ( accepted < rows - 1 && round[ accepted + 1 ] == picked[ accepted ] )
                ++accepted;

            const int32_t next = picked[ accepted ];

            out[ 0 ] = static_cast<int32_t>( accepted );

            for ( dim_t i = 1; i <= accepted; ++i )
                out[ i ] = round[ i ];

            out[ accepted + 1 ] = next;
            round[ 0 ] = next;
        }

        OperationType getOperationType() const override
        {
            return OperationType::SamplingOp;
        }

        std::string getName() const override
        {
            return "Cpu::SamplingOp";
        }

    private:

        int32_t sampleRow( const float* row, const SamplingParams& params, float r ) const
        {
            const int64_t vocab = config_.getVocabularySize();

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                int32_t best_idx = 0;
                float best = row[ 0 ];

                for ( int64_t i = 1; i < vocab; ++i )
                {
                    if ( row[ i ] > best )
                    {
                        best = row[ i ];
                        best_idx = static_cast<int32_t>( i );
                    }
                }

                return best_idx;
            }

            const float softcap = config_.getFinalLogitSoftcap();
            const float temperature = params.temperature;

            const auto scaled = [&]( int64_t i ) -> float
            {
                float x = row[ i ];

                if ( softcap > 0.0f )
                    x = softcap * std::tanh( x / softcap );

                return x / temperature;
            };

            float max_val = -std::numeric_limits<float>::infinity();

            for ( int64_t i = 0; i < vocab; ++i )
                max_val = std::max( max_val, scaled( i ) );

            std::vector<float> probs( static_cast<size_t>( vocab ) );
            double sum = 0.0;

            for ( int64_t i = 0; i < vocab; ++i )
            {
                const float e = static_cast<float>( std::exp( static_cast<double>( scaled( i ) - max_val ) ) );
                probs[ i ] = e;
                sum += e;
            }

            // Top-k: keep the k largest probabilities, zero the rest. Exact k-th value via
            // nth_element (the device path approximates this threshold by binary search).
            if ( params.top_k > 0 && params.top_k < static_cast<int>( vocab ) )
            {
                std::vector<float> ordered( probs );
                std::nth_element( ordered.begin(), ordered.begin() + ( vocab - params.top_k ), ordered.end() );
                const float threshold = ordered[ vocab - params.top_k ];

                sum = 0.0;
                for ( int64_t i = 0; i < vocab; ++i )
                {
                    if ( probs[ i ] < threshold ) probs[ i ] = 0.0f;
                    else sum += probs[ i ];
                }
            }

            // Top-p: keep the smallest set of highest-probability tokens whose mass reaches top_p.
            if ( params.top_p < 1.0f )
            {
                std::vector<float> ordered( probs );
                std::sort( ordered.begin(), ordered.end(), std::greater<float>() );

                const double target = static_cast<double>( params.top_p ) * sum;
                double cum = 0.0;
                float threshold = 0.0f;

                for ( float v : ordered )
                {
                    cum += v;
                    threshold = v;

                    if ( cum >= target ) break;
                }

                sum = 0.0;
                for ( int64_t i = 0; i < vocab; ++i )
                {
                    if ( probs[ i ] < threshold ) probs[ i ] = 0.0f;
                    else sum += probs[ i ];
                }
            }

            const double sample_target = static_cast<double>( r ) * sum;
            double cumulative = 0.0;
            int32_t result = static_cast<int32_t>( vocab - 1 );

            for ( int64_t i = 0; i < vocab; ++i )
            {
                cumulative += probs[ i ];

                if ( probs[ i ] > 0.0f && cumulative >= sample_target )
                {
                    result = static_cast<int32_t>( i );
                    break;
                }
            }

            return result;
        }

        IExecutionContext* context_;
        SamplingConfig config_;
        mutable int32_t pending_token_{ 0 };
    };
}
