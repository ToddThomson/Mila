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

export module Compute.CpuSamplingOp;

import Dnn.Samplers.SamplingConfig;
import Dnn.GenerateParams;
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
            const int64_t offset = static_cast<int64_t>( logits.size() ) - vocab;

            const float* row = static_cast<const float*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

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

                out[ 0 ] = best_idx;
                return;
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

            out[ 0 ] = result;
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
        IExecutionContext* context_;
        SamplingConfig config_;
    };
}
