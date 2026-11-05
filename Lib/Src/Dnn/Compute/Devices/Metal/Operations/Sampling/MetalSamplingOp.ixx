/**
 * @file MetalSamplingOp.ixx
 * @brief Token sampling on Metal.
 */

export module Compute.MetalSamplingOp;

import std;
import Compute.Device;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.ITensor;
import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorOps;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Sampling
{
    /**
     * @brief Draws one token from the last row of logits, greedy or by temperature, top-k and top-p.
     *
     * Metal has no event for one small readback, so `awaitToken` synchronizes the context, which also finishes anything encoded after
     * the sample. The decode-ahead pipeline therefore runs one step at a time here, with the same tokens. Several rows at once are Phase 6's,
     * with the speculative sampler that uses them.
     */
    export template<TensorDataType TPrecision>
    class MetalSamplingOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        MetalSamplingOp( IExecutionContext* context, const SamplingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalSamplingOp" ) ),
              config_( validated( config ) ),
              scratch_( context->getDeviceId(), shape_t{ config.getVocabularySize() } )
        {
            uploadSuppressedTokens();
        }

        /**
         * @brief Encode one sample into `token_out`.
         *
         * @param logits   Metal logits; the last `vocab_size` elements are the row.
         * @param produced The reply's produced tokens, required with a presence penalty, and marked with the token drawn.
         * @throws std::invalid_argument For a penalty that is not finite, or one without the produced tokens.
         */
        void forward( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeSample( logits, token_out, params, r, produced );
        }

        /// As `forward`, remembering the token for `awaitToken`. One may be outstanding at a time.
        void enqueueForward( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeSample( logits, token_out, params, r, produced );
            pending_token_ = &token_out;
        }

        void enqueueForwardOnDevice( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeSample( logits, token_out, params, r, produced );
        }

        /// Wait for the sample `enqueueForward` encoded, and return its token. @throws std::logic_error When there is none.
        [[nodiscard]] std::int32_t awaitToken() const
        {
            if ( pending_token_ == nullptr )
            {
                throw std::logic_error( "MetalSamplingOp::awaitToken: no enqueueForward() outstanding" );
            }

            context_->synchronize();

            const std::int32_t token = *static_cast<const std::int32_t*>( static_cast<const ITensor*>( pending_token_ )->rawData() );

            pending_token_ = nullptr;

            return token;
        }

        /// Encode clearing the reply's produced tokens.
        void enqueueClearProduced( ITensor& produced ) const
        {
            requireProduced( produced );
            context_->zeroBytes( produced.rawData(), static_cast<std::size_t>( config_.getVocabularySize() ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::SamplingOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Sampling";
        }

    private:

        // The kernels' SamplingParameters, field for field.
        struct Parameters
        {
            std::int32_t vocab;
            std::int32_t top_k;
            float top_p;
            float temperature;
            float softcap;
            float presence_penalty;
            float uniform;
            std::uint32_t use_suppressed;
            std::uint32_t use_produced;
        };

        [[nodiscard]] static SamplingConfig validated( const SamplingConfig& config )
        {
            config.validate();

            return config;
        }

        void uploadSuppressedTokens()
        {
            const auto& tokens = config_.getSuppressedTokens();

            if ( tokens.empty() )
            {
                return;
            }

            const std::int64_t vocab = config_.getVocabularySize();

            suppressed_ = std::make_unique<Tensor<TensorDataType::UINT8, MetalMemoryResource>>( context_->getDeviceId(), shape_t{ vocab } );

            std::vector<std::int32_t> flags( static_cast<std::size_t>( vocab ), 0 );

            for ( const std::int32_t token : tokens )
            {
                flags[static_cast<std::size_t>( token )] = 1;
            }

            fill( *suppressed_, std::span<const std::int32_t>( flags ) );
        }

        void requireProduced( const ITensor& produced ) const
        {
            if ( produced.getDataType() != TensorDataType::UINT8 || produced.size() < config_.getVocabularySize() )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp: the produced tokens are one UINT8 flag per token of the {} in the vocabulary; given {} of {}",
                    config_.getVocabularySize(), produced.size(), produced.getDataTypeName() ) );
            }
        }

        void encodeSample( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced ) const
        {
            if ( !std::isfinite( params.presence_penalty ) )
            {
                throw std::invalid_argument( std::format( "MetalSamplingOp: the presence penalty must be finite; given {}", params.presence_penalty ) );
            }

            const bool penalized = params.presence_penalty != 0.0f;

            if ( penalized )
            {
                if ( produced == nullptr )
                {
                    throw std::invalid_argument( "MetalSamplingOp: a presence penalty needs the reply's produced tokens" );
                }

                requireProduced( *produced );
            }

            const std::int64_t vocab = config_.getVocabularySize();
            const auto* const row = static_cast<const std::byte*>( logits.rawData() )
                + metalElementBytes<TPrecision>( logits.size() - vocab );

            const Parameters parameters{
                narrowToKernelIndex( vocab ), params.top_k, params.top_p, params.temperature, config_.getFinalLogitSoftcap(),
                params.presence_penalty, r, suppressed_ ? 1u : 0u, penalized ? 1u : 0u };

            // What is absent is bound to something valid, and the kernel's flags keep it unread.
            const void* const scratch = scratch_.rawData();
            const void* const suppressed = suppressed_ ? static_cast<const ITensor*>( suppressed_.get() )->rawData() : scratch;
            const void* const flags = penalized ? produced->rawData() : scratch;

            const std::string_view type = MetalTensorDataTypeMap<TPrecision>::kernel_type;
            const bool greedy = params.temperature <= 0.0f || params.top_k == 1;

            if ( greedy )
            {
                const std::array<const void*, 4> buffers{ row, token_out.rawData(), suppressed, flags };

                context_->dispatchThreadgroups(
                    context_->device().kernels().pipeline( std::format( "sampling_greedy_{}", type ) ),
                    buffers, std::as_bytes( std::span( &parameters, 1 ) ), 1, 1 );
            } else {
                const std::array<const void*, 5> buffers{ row, token_out.rawData(), scratch, suppressed, flags };

                context_->dispatchThreadgroups(
                    context_->device().kernels().pipeline( std::format( "sampling_draw_{}", type ) ),
                    buffers, std::as_bytes( std::span( &parameters, 1 ) ), 1, 1 );
            }

            if ( penalized )
            {
                const std::array<const void*, 2> buffers{ token_out.rawData(), produced->rawData() };

                context_->dispatch( context_->device().kernels().pipeline( "sampling_mark_produced" ), buffers, {}, 1 );
            }
        }

        MetalExecutionContext* context_;
        SamplingConfig config_;
        mutable Tensor<TensorDataType::FP32, MetalMemoryResource> scratch_;
        std::unique_ptr<Tensor<TensorDataType::UINT8, MetalMemoryResource>> suppressed_;
        mutable ITensor* pending_token_{ nullptr };
    };
}
