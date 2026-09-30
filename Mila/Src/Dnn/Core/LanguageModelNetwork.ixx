/**
 * @file LanguageModelNetwork.ixx
 * @brief Abstract base for language model networks.
 */

module;
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

export module Dnn.LanguageModelNetwork;

export import Dnn.SequenceLogLikelihood;

import Dnn.Network;
import Dnn.Tensor;
import Dnn.TensorOps;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.IDecodeRecording;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The type erasure boundary between a language model and its transformer.
     *
     * A concrete transformer is templated on its weight-quantization and KV-cache policies,
     * and those parameters have no business reaching the model layer -- GemmaModel should
     * not be a different type because its weights are FP4. LanguageModelNetwork is where they
     * stop. LanguageModel holds one of these and drives generation through it, so the
     * interface below is the whole vocabulary a model needs from a transformer:
     * prefill/decode, the prefix-reuse pair (prefillFrom + rewindKvCache), and nothing else.
     *
     * The virtual boundary is deliberately coarse. One dispatch per layer per token step is
     * negligible against the per-layer GEMMs, so the interface is drawn at whole passes
     * rather than at anything finer.
     *
     * Implemented by all four transformer families:
     *
     *   Network<TDeviceType, TPrecision>
     *     +- LanguageModelNetwork<TDeviceType, TPrecision>
     *          +- GptTransformer<...>          training + inference
     *          +- LlamaTransformer<...>        training + inference
     *          +- GemmaTransformer<...>        inference only
     *          +- QwenTransformer<...>         inference only
     *
     * Not every member is implemented by every family; each states its own support.
     */
    // REVIEW: this base claims more than every family provides -- forward/backward are pure
    // yet two families throw, and prefix reuse is optional and absent from the other two. The
    // fix is subtraction, not a capability parameter: a template argument for trainability
    // would propagate into LanguageModel and destroy the type erasure this class exists for,
    // and "inference-only" is a status rather than a property (nobody wrote Gemma's backward;
    // the architecture does not forbid one). Specifications/Notebooks/TransformerApiReadiness.md items
    // 7 and 8.
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LanguageModelNetwork : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TensorDataType::INT32, MR>;

        explicit LanguageModelNetwork( const std::string& name )
            : NetworkBase( name )
        {}

        ~LanguageModelNetwork() override = default;

        /**
         * @brief Full-sequence forward pass -- the training path, not the generation path.
         *
         * Retains per-layer activations for backward. Implemented by GptTransformer and
         * LlamaTransformer; the inference-only families (GemmaTransformer, QwenTransformer)
         * throw. Generation does not use this -- see prefill/decode below.
         *
         * @param input  Token indices [B, T].
         * @return       Logits [B, T, vocab_size].
         */
        virtual TensorType& forward( const TokenIndexType& input ) = 0;

        /**
         * @brief Full backward pass (training). Requires training mode and a prior forward.
         *
         * Implemented by GptTransformer and LlamaTransformer; the inference-only families
         * throw.
         *
         * The return type is the INT32 token-index tensor and carries no usable gradient:
         * the input is discrete, so the token embedding has no input gradient to produce.
         * Callers should discard it.
         *
         * @param input       Token indices [B, T].
         * @param output_grad Gradient of the loss w.r.t. logits.
         */
        // REVIEW: the return value is vestigial -- BardTrainer, the only caller, discards it.
        // Either it should be void or the type is wrong. TransformerApiReadiness.md item 8a.
        virtual TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad ) = 0;

        /**
         * @brief Inference prefill -- process full prompt and populate the KV cache.
         *
         * Equivalent to prefillFrom( input, 0 ). Starting from position 0 also discards any
         * state carried from a previous sequence, so a fresh prompt needs no explicit reset:
         * the KV cache, the DeltaNet recurrent state and the convolution windows all treat
         * offset 0 as a new sequence.
         *
         * @param input  Full prompt token indices [B, T].
         * @return       Logits for the last token position.
         */
        // REVIEW: prefillFrom is the primitive and this is the offset-0 case, but the base
        // has it backwards -- this one is pure and the general form carries a default.
        // ITransformerBlock models it as one method taking the offset.
        // TransformerApiReadiness.md item 7.
        virtual TensorType& prefill( const TokenIndexType& input ) = 0;

        /**
         * @brief Inference decode -- single-token autoregressive step.
         *
         * Writes `position` into the execution context, where the step's kernels read it
         * (DecodeGraph.md section 4.1), runs the family's step -- called, or replayed from a
         * recording when replay is on (setDecodeReplay) -- and advances the number of positions
         * the KV caches hold.
         *
         * @param input    Single token index [B, 1].
         * @param position Current sequence position (0-based).
         * @return         Logits [B, 1, vocab_size].
         */
        TensorType& decode( const TokenIndexType& input, dim_t position )
        {
            IExecutionContext& context = *this->getExecutionContext();

            context.setDecodePosition( position );

            TensorType& logits = decodeStep( context, input, position );

            cached_length_ = std::max( cached_length_, position + 1 );

            return logits;
        }

        /**
         * @brief Replay each decode step from one recording instead of calling it (DecodeGraph.md).
         *
         * Off for a network built directly; a model turns it on at load. With it on, the first
         * step is called, the second is recorded and replayed, the third checks a replay against
         * the called step bit for bit, and every later step is one replay. A step that cannot be
         * recorded, or a replay that differs, turns replay off for the network's life with a
         * warning. A step taken while an activation observer is installed is called. Debugging a
         * decode step means turning this off first.
         */
        void setDecodeReplay( bool enabled )
        {
            decode_replay_ = enabled;
            discardDecodeRecording();
        }

        /// Whether decode steps are replayed: on, and not turned off by a failed recording or check.
        [[nodiscard]] bool isDecodeReplayed() const noexcept
        {
            return decode_replay_;
        }

        /**
         * @brief Chunked prefill starting at an absolute position (prompt-prefix reuse).
         *
         * @param input        The FULL prompt token indices [B, T] -- not a pre-sliced
         *                     tail; token index and absolute position coincide.
         * @param start_offset First position to prefill; [0, start_offset) must already
         *                     be resident in the KV caches (see rewindKvCache).
         * @return             Logits for the last token position.
         *
         * Implemented by GemmaTransformer and QwenTransformer, which override both this and
         * rewindKvCache; on any other network it throws. Because rewindKvCache defaults to
         * false and a caller reaches this only after a successful rewind, the throw is not
         * reachable through the intended sequence.
         */
        virtual TensorType& prefillFrom( const TokenIndexType& input, dim_t start_offset )
        {
            ( void )input;
            ( void )start_offset;
            throw std::logic_error( "LanguageModelNetwork::prefillFrom: not supported by this network" );
        }

        /**
         * @brief The log-likelihood the model assigns to a given sequence, teacher-forced.
         *
         * Runs the sequence through the prefill path and, at every position, reads the
         * probability the model assigned to the token that actually came next. Nothing is
         * sampled and nothing is generated, so the result is a property of the model and the
         * text alone -- which is what makes it usable as a quality measure between two
         * quantizations of the same weights. Perplexity is exp( -total / scored positions ).
         *
         * Distinct from prefill() rather than an option on it: prefill returns logits for one
         * position and its callers sample that row, so widening what prefill returns would
         * silently move which row a sampler reads.
         *
         * The head is evaluated in windows of the family config's log-likelihood window; at the
         * default of 1 it costs one head pass per position. Implemented by GemmaTransformer and
         * QwenTransformer; on any other network it throws.
         *
         * @param input Token indices [1, T], T >= 2. Batching is not supported: the targets
         *              are the sequence's own next tokens, so two sequences in one call would
         *              need per-row lengths the shape cannot carry.
         */
        virtual SequenceLogLikelihood sequenceLogLikelihood( const TokenIndexType& input )
        {
            ( void )input;
            throw std::logic_error( "LanguageModelNetwork::sequenceLogLikelihood: not supported by this network" );
        }

        /**
         * @brief Rewind the KV caches to `position` for prompt-prefix reuse
         * (PromptCaching.md). Positions [0, position) stay valid; device contents
         * are untouched.
         *
         * @return true when every layer accepted the rewind, which then sets the cached
         * length to `position`. False on a network with no reuse capability; a full
         * prefill positionally overwrites regardless, so a refused rewind never needs
         * cleanup.
         */
        bool rewindKvCache( dim_t position )
        {
            // NOTE: there is deliberately no resetKvCache counterpart. Starting a prefill at
            // position 0 already discards carried state in every stateful component, so a new
            // sequence needs no explicit reset. TransformerApiReadiness.md item 2.
            if ( !onRewindKvCache( position, cached_length_ ) )
                return false;

            cached_length_ = position;

            return true;
        }

    protected:

        /// The family's decode step, the one decode() runs (DecodeGraph.md section 4.4).
        virtual TensorType& onDecode( const TokenIndexType& input, dim_t position ) = 0;

        /**
         * @brief Whether every layer accepts a rewind to `position` from `cached_length`.
         *
         * Implemented by GemmaTransformer and QwenTransformer; false on any other network.
         */
        virtual bool onRewindKvCache( dim_t position, dim_t cached_length )
        {
            ( void )position;
            ( void )cached_length;

            return false;
        }

        /**
         * @brief Set the number of positions the KV caches hold.
         *
         * Every prefill path calls this when it returns, with the end of the span it wrote.
         * The count is the network's rather than each attention op's, so a decode step leaves
         * no host state in an op behind it (DecodeGraph.md section 4.3).
         */
        void setCachedLength( dim_t length ) noexcept
        {
            cached_length_ = length;
        }

        /**
         * @brief Keep a copy of the state a decode step advances in place, for the self-check.
         *
         * The self-check runs one step twice, called and then replayed. A KV-cache write repeated
         * lands the same rows in the same slots, so a network with only KV caches keeps nothing
         * (the default). A recurrent state would advance twice; a network that carries one copies
         * it here and puts it back in restoreDecodeState().
         */
        virtual void holdDecodeState()
        {
        }

        /// Put back the state holdDecodeState() copied.
        virtual void restoreDecodeState()
        {
        }

        /// Free the copy holdDecodeState() took; the self-check is done with it.
        virtual void releaseDecodeState()
        {
        }

        /// Drop the decode recording; the next steps prime and record it again.
        void discardDecodeRecording() noexcept
        {
            decode_recording_.reset();
            recorded_input_ = nullptr;
            recorded_logits_.reset();
            replay_stage_ = ReplayStage::Unprimed;
        }

    private:

        /// Where the recording's life stands (DecodeGraph.md section 4.4).
        enum class ReplayStage
        {
            Unprimed,  ///< The next step is called, doing any lazy setup and touching every buffer.
            Primed,    ///< The next step is recorded, then replayed.
            Recorded,  ///< The next step checks a replay against the called step.
            Verified   ///< Every step is replayed.
        };

        TensorType& decodeStep( IExecutionContext& context, const TokenIndexType& input, dim_t position )
        {
            if ( !decode_replay_ || context.hasActivationObserver() )
                return onDecode( input, position );

            // The recording holds the input's address; a different input tensor needs a new one.
            if ( decode_recording_ && input.rawData() != recorded_input_ )
                discardDecodeRecording();

            switch ( replay_stage_ )
            {
                case ReplayStage::Unprimed:
                {
                    TensorType& logits = onDecode( input, position );
                    replay_stage_ = ReplayStage::Primed;

                    return logits;
                }

                case ReplayStage::Primed:
                    return recordDecodeStep( context, input, position );

                case ReplayStage::Recorded:
                    return checkDecodeRecording( context, input, position );

                case ReplayStage::Verified:
                default:
                    decode_recording_->replay();

                    return *recorded_logits_;
            }
        }

        TensorType& recordDecodeStep( IExecutionContext& context, const TokenIndexType& input, dim_t position )
        {
            decode_recording_ = context.createDecodeRecording();
            TensorType* logits = nullptr;

            if ( !decode_recording_
                || !decode_recording_->record( [ & ] { logits = &onDecode( input, position ); } ) )
            {
                turnOffDecodeReplay( "its decode step could not be recorded" );

                return onDecode( input, position );
            }

            // The recording outlives the step that made it: a later pass may run the head at another shape
            // (a log-likelihood window), and the component's view then describes that. The network keeps its
            // own description of the region the recording writes the logits to.
            recorded_input_ = input.rawData();
            recorded_logits_ = std::make_unique<TensorType>( logits->view( logits->shape() ) );
            replay_stage_ = ReplayStage::Recorded;

            decode_recording_->replay();

            return *recorded_logits_;
        }

        /**
         * The self-check (DecodeGraph.md section 5.1): the step called, then replayed at the same
         * position from the same state. Writing the same K and V to the same cache slot twice
         * leaves the cache as it was, and a recurrent state is put back between the two, so the
         * replay sees what the called step saw and any difference is a value the recording froze.
         */
        TensorType& checkDecodeRecording( IExecutionContext& context, const TokenIndexType& input, dim_t position )
        {
            holdDecodeState();

            TensorType& called = onDecode( input, position );
            const std::vector<float> called_logits = hostLogits( called, context );

            restoreDecodeState();
            decode_recording_->replay();
            const std::vector<float> replayed_logits = hostLogits( *recorded_logits_, context );

            const bool identical = called.rawData() == recorded_logits_->rawData()
                && called_logits.size() == replayed_logits.size()
                && std::memcmp( called_logits.data(), replayed_logits.data(),
                    called_logits.size() * sizeof( float ) ) == 0;

            if ( !identical )
            {
                turnOffDecodeReplay( "a replayed decode step differed from the called one" );

                // The replay overwrote the logits and advanced any recurrent state; the called step,
                // from the held state, writes both again.
                restoreDecodeState();
                releaseDecodeState();

                return onDecode( input, position );
            }

            releaseDecodeState();
            replay_stage_ = ReplayStage::Verified;

            return called;
        }

        std::vector<float> hostLogits( const TensorType& logits, IExecutionContext& context )
        {
            auto host = toHost<TensorDataType::FP32>( logits, &context );
            context.synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        void turnOffDecodeReplay( std::string_view reason )
        {
            decode_replay_ = false;
            discardDecodeRecording();

            Logging::Logger::warning( std::format(
                "{}: decode replay is off for this network -- {}; every decode step is called", this->getName(), reason ) );
        }

        dim_t cached_length_{ 0 };

        bool decode_replay_{ false };
        ReplayStage replay_stage_{ ReplayStage::Unprimed };
        std::unique_ptr<IDecodeRecording> decode_recording_;
        const void* recorded_input_{ nullptr };
        std::unique_ptr<TensorType> recorded_logits_;
    };
}
