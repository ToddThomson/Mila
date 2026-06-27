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
            [[maybe_unused]] float r ) const
        {
            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (!greedy)
            {
                throw std::runtime_error(
                    "CpuSamplingOp: stochastic sampling not implemented (Phase B). "
                    "Set temperature <= 0 or top_k == 1 for greedy decode." );
            }

            const int64_t vocab = config_.getVocabularySize();
            const int64_t offset = static_cast<int64_t>( logits.size() ) - vocab;

            const float* row = static_cast<const float*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

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
