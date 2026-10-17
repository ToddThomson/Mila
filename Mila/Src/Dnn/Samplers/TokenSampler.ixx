/**
 * @file TokenSampler.ixx
 * @brief Device-agnostic token sampler facade.
 *
 * Model-owned orchestrator tool (sibling of the Optimizer facade). Resolves the
 * device SamplingOp through OperationTraits, owns the host RNG, and performs the
 * 4-byte device->host token readback. See Specifications/TokenSampling.md.
 */

export module Dnn.Samplers.TokenSampler;

import std;
import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Compute.SamplerBase;
import Compute.OperationTraits;
import Compute.OperationType;
import Dnn.Tensor;
import Dnn.TensorOps;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The standard token sampler: temperature / top-k / top-p multinomial with a presence penalty.
     *
     * Dispatches to the device SamplingOp resolved by OperationTraits (CudaSamplingOp).
     * Shares the model's ExecutionContext so the op runs on the decode
     * stream and writes the token in place. The host-drawn uniform is generated here so
     * the op stays pure and deterministic.
     *
     * Owns the reply's produced tokens, the state the presence penalty reads: beginReply() starts a
     * reply, and every sample drawn under a penalty marks its token on the device.
     *
     * @tparam TDeviceType Device the logits reside on.
     * @tparam TPrecision  Logits precision (FP32 or BF16).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class TokenSampler : public Sampler<TDeviceType, TPrecision>
    {
    public:
        using Base = Sampler<TDeviceType, TPrecision>;
        using TokenTensor = typename Base::TokenTensor;
        using SamplingOpType = typename OperationTraits<OperationType::SamplingOp, TDeviceType, TPrecision>::type;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ProducedTensor = Tensor<TensorDataType::UINT8, MR>;

        TokenSampler( IExecutionContext* context, const SamplingConfig& config )
            : context_( context ),
              config_( config ),
              host_token_( Device::Cpu(), shape_t{ 1, 1 } ),
              rng_( std::random_device{}() )
        {
            if (!context_)
            {
                throw std::invalid_argument( "TokenSampler: ExecutionContext cannot be null" );
            }

            config_.validate();
            op_ = std::make_shared<SamplingOpType>( context_, config_ );
        }

        /**
         * @brief Start a reply sampled at @p params: with a presence penalty, no token counts as produced yet.
         *
         * Enqueued on the model's stream, so it needs no wait. A reply sampled with the penalty off tracks nothing,
         * and a sample drawn under a penalty after one is refused.
         */
        void beginReply( const SamplingParams& params )
        {
            tracking_ = params.presence_penalty != 0.0f;

            if ( !tracking_ )
                return;

            if ( !produced_ )
            {
                produced_ = std::make_unique<ProducedTensor>(
                    context_->getDeviceId(), shape_t{ config_.getVocabularySize() } );
            }

            op_->enqueueClearProduced( *produced_ );
        }

        /// The reply's produced tokens, for the speculative sampler's rows and acceptance; null when nothing is tracked.
        ITensor* producedTokens() noexcept
        {
            return tracking_ ? produced_.get() : nullptr;
        }

        std::int32_t sample(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params ) override
        {
            const float r = drawUniform();

            op_->forward( logits, token_out, params, r, producedFor( params ) );

            // The op runs on the context's stream, so the readback copies on it and waits for it: the
            // host token is valid before the caller's stop-check / on_token callback.
            copy( token_out, host_token_, context_ );
            context_->synchronize();

            return host_token_.data()[ 0 ];
        }

        /**
         * @brief Enqueue one sampling step without waiting for the token readback.
         *
         * Decode-ahead half of the pipelined generation loop: the op samples on the
         * model's stream (ordered after the forward pass that produced @p logits) and
         * writes the token into @p token_out in place, so the next decode step can be
         * enqueued before the host knows the token id. The host uniform is drawn here
         * at enqueue time -- one draw per sampled token, same RNG sequence as sample().
         * awaitToken() must be called before the next enqueueSample().
         */
        void enqueueSample(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params )
        {
            const float r = drawUniform();

            op_->enqueueForward( logits, token_out, params, r, producedFor( params ) );
        }

        /**
         * @brief Enqueue one sampling step whose token stays on the device: no readback, no host wait.
         *
         * For a chain of passes that each read the token the one before chose -- a draft model's steps, the rows of
         * a verify -- with the host reading the tokens once at the end of the chain. Draws one host uniform, as
         * enqueueSample() does, and may be called any number of times between enqueueSample() and awaitToken().
         *
         * @param logits    Device logits; the last `vocab_size` elements are the row sampled.
         * @param token_out Device INT32 element the token is written to, such as one slot of a token sequence.
         * @param params    Per-call sampling parameters; greedy draws nothing from the uniform.
         */
        void enqueueSampleOnDevice(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params )
        {
            const float r = drawUniform();

            op_->enqueueForwardOnDevice( logits, token_out, params, r, producedFor( params ) );
        }

        /**
         * @brief Block until the last enqueueSample()'s token id is host-visible and return it.
         */
        std::int32_t awaitToken()
        {
            return op_->awaitToken();
        }

        /**
         * @brief Draw the next uniform in [0, 1) from the model's sampling stream.
         *
         * Every sampling step draws one, greedy included, so a seeded run's stream advances the same way whichever
         * sampler consumes it; the speculative sampler's rows take theirs from here (Gemma4Mtp.md 4.8).
         */
        float drawUniform()
        {
            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );

            return dist( rng_ );
        }

        /**
         * @brief Reseed the host RNG for reproducible sampling.
         *
         * Called once per generation run (not per token) when the caller supplies a seed.
         */
        void reseed( std::uint64_t seed )
        {
            rng_.seed( static_cast<std::mt19937::result_type>( seed ) );
        }

    private:

        ITensor* producedFor( const SamplingParams& params ) const
        {
            if ( params.presence_penalty == 0.0f )
                return nullptr;

            if ( !tracking_ )
            {
                throw std::logic_error(
                    "TokenSampler: a sample with a presence penalty outside a reply begun with one" );
            }

            return produced_.get();
        }

        IExecutionContext* context_;
        SamplingConfig config_;
        Tensor<TensorDataType::INT32, CpuMemoryResource> host_token_;
        std::shared_ptr<SamplingOpType> op_;
        std::mt19937 rng_;

        // One flag per vocabulary token the current reply has produced; allocated by the first reply with a penalty.
        std::unique_ptr<ProducedTensor> produced_;
        bool tracking_{ false };
    };
}
