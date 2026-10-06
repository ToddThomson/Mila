/**
 * @file LanguageModelNetwork.ixx
 * @brief Abstract base for language model networks.
 */

module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
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
     * prefill/decode, a draft and its check (draftTokens, decodeTokens), prefix reuse (prefillFrom,
     * rewindKvCache, savePosition), and nothing else.
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
            this->getExecutionContext()->setDecodePosition( position );

            TensorType& logits = replayed( decode_replay_, input, [] {},
                [ & ]() -> TensorType& { return onDecode( input, position ); } );

            cached_length_ = std::max( cached_length_, position + 1 );

            return logits;
        }

        /**
         * @brief Decode of a few tokens in a row from `position`, each at decode's arithmetic, in one pass.
         *
         * The verify of speculative decoding (Gemma4Mtp.md 4.7), and any run of tokens already known: token t
         * sits at position + t, attends everything before it, and leaves its keys in the caches, as t + 1 decode()
         * calls would; every weight is read once for all of them. Writes `position` into the execution context as
         * decode() does and advances the cached length past the last token. Each token's logits equal its
         * decode()'s to FP32 rounding, not bit for bit (the sums run in another order).
         *
         * Replayed as decode() is (setDecodeReplay), with one recording for each number of tokens. Implemented by
         * GemmaTransformer (dense layers); any other network throws, as does one whose configuration declares fewer
         * decode tokens.
         *
         * @param input    Token indices [1, T], T up to the decode tokens the network was built for.
         * @param position Position of the first token (0-based).
         * @return         Logits [1, T, vocab_size].
         */
        TensorType& decodeTokens( const TokenIndexType& input, dim_t position )
        {
            this->getExecutionContext()->setDecodePosition( position );

            TensorType& logits = replayed( decode_tokens_replay_, input, [] {},
                [ & ]() -> TensorType& { return onDecodeTokens( input, position ); } );

            cached_length_ = std::max( cached_length_, position + input.shape()[ 1 ] );

            return logits;
        }

        /**
         * @brief Propose the tokens after a known one with the network's draft model, for decodeTokens() to check.
         *
         * Slot 0 of `tokens` holds the token at `position`, not yet in the caches; the draft model fills slots 1 to
         * T - 1, each its most likely next token, on the device, reading the caches up to position - 1 and writing
         * nothing to them (Gemma4Mtp.md 4.3). It continues from the final-normed hidden state of the last pass at
         * `hidden_row`: the row of the token before `position` -- 0 after a prefill or decode, which keep one row,
         * and after decodeTokens the row of the last token kept. Writes `position` into the execution context as
         * decode() does, and is replayed as decode() is, with one recording for each number of tokens; the row is
         * taken before the replay, so it may differ from one draft to the next. Implemented by GemmaTransformer built
         * with a draft model; any other network throws.
         *
         * @param tokens     Token indices [1, T], T up to the decode tokens the network was built for.
         * @param position   Position of slot 0's token (0-based).
         * @param hidden_row Row of the last pass's final-normed hidden state the draft continues from.
         */
        void draftTokens( TokenIndexType& tokens, dim_t position, dim_t hidden_row )
        {
            this->getExecutionContext()->setDecodePosition( position );

            ( void )replayed( draft_replay_, tokens, [ & ] { onDraftFrom( hidden_row ); },
                [ & ]() -> TokenIndexType& { onDraftTokens( tokens, position ); return tokens; } );
        }

        /**
         * @brief Replay each decode pass from a recording instead of calling it (DecodeGraph.md).
         *
         * Covers decode(), decodeTokens() and draftTokens(), each kind with its own recordings: one for each number of
         * tokens, made from the first call with that input tensor. Off for a network built directly; a model turns it
         * on at load. With it on, a recording's first pass is called, the second is recorded and replayed, the third
         * checks a replay against the called pass bit for bit, and every later pass is one replay. A pass that cannot
         * be recorded, or a replay that differs, turns replay off for that kind of pass for the network's life with a
         * warning. A pass taken while an activation observer is installed is called. Debugging a decode pass means
         * turning this off first.
         */
        void setDecodeReplay( bool enabled )
        {
            replay_enabled_ = enabled;

            decode_replay_.reset();
            decode_tokens_replay_.reset();
            draft_replay_.reset();
        }

        /// Whether decode passes are replayed: on, and no kind of pass turned off by a failed recording or check.
        [[nodiscard]] bool isDecodeReplayed() const noexcept
        {
            return replay_enabled_ && !decode_replay_.off && !decode_tokens_replay_.off && !draft_replay_.off;
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
         * Implemented by GemmaTransformer, LlamaTransformer and QwenTransformer, which override both
         * this and rewindKvCache; on any other network it throws. Because rewindKvCache defaults to
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
         * default of 1 it costs one head pass per position. Implemented by GemmaTransformer,
         * LlamaTransformer and QwenTransformer; on any other network it throws.
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
         * @brief The log-likelihood of `input` after a cached prefix, as prefillFrom continues a prefill.
         *
         * Positions [0, start_offset) must already be resident in the caches (rewindKvCache). Scores the
         * T - 1 - start_offset tokens after start_offset, each given everything before it, through the same
         * prefill path sequenceLogLikelihood runs. The log-likelihood of a continuation is then the difference
         * of two calls from one rewound position.
         *
         * @param input        The FULL sequence [1, T]; token index and absolute position coincide.
         * @param start_offset First position to prefill, at most T - 2 so one position is scored.
         */
        virtual SequenceLogLikelihood sequenceLogLikelihoodFrom( const TokenIndexType& input, dim_t start_offset )
        {
            ( void )input;
            ( void )start_offset;
            throw std::logic_error( "LanguageModelNetwork::sequenceLogLikelihoodFrom: not supported by this network" );
        }

        /**
         * @brief Keep what a later rewindKvCache needs to return to the current fill.
         *
         * A network whose caches are positional can rewind to any position it holds and keeps
         * nothing. A network with recurrent layers cannot rewind them, so it copies their state
         * here, and a rewind to exactly this position succeeds by putting the copy back. One
         * position is held at a time; saving again replaces it, and a prefill or decode that
         * writes inside the saved prefix discards it.
         *
         * @return The saved position: the number of positions the caches hold.
         */
        dim_t savePosition()
        {
            onSavePosition( cached_length_ );

            return cached_length_;
        }

        /**
         * @brief Rewind the KV caches to `position` for prompt-prefix reuse
         * (PromptCaching.md). Positions [0, position) stay valid; device contents
         * are untouched.
         *
         * A network with recurrent layers accepts only the position it saved (savePosition).
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

        /// The family's multi-token decode, the one decodeTokens() runs. Throws on a family without one.
        virtual TensorType& onDecodeTokens( const TokenIndexType& input, dim_t position )
        {
            ( void )input;
            ( void )position;

            throw std::logic_error( "LanguageModelNetwork::decodeTokens: not supported by this network" );
        }

        /**
         * @brief Put the final-normed hidden state at `hidden_row` where the draft's first step reads it.
         *
         * Called before every run of a draft, called or replayed, and never recorded: the row differs between drafts,
         * and onDraftTokens(), which a recording holds, reads it from one place that its later steps overwrite. Throws
         * on a network built without a draft model.
         */
        virtual void onDraftFrom( dim_t hidden_row )
        {
            ( void )hidden_row;

            throw std::logic_error( "LanguageModelNetwork::draftTokens: this network has no draft model" );
        }

        /// The family's draft, the one draftTokens() runs, from the hidden state onDraftFrom() put in place.
        virtual void onDraftTokens( TokenIndexType& tokens, dim_t position )
        {
            ( void )tokens;
            ( void )position;

            throw std::logic_error( "LanguageModelNetwork::draftTokens: this network has no draft model" );
        }

        /**
         * @brief Whether every layer accepts a rewind to `position` from `cached_length`.
         *
         * Implemented by GemmaTransformer, LlamaTransformer and QwenTransformer; false on any other network.
         */
        virtual bool onRewindKvCache( dim_t position, dim_t cached_length )
        {
            ( void )position;
            ( void )cached_length;

            return false;
        }

        /// Keep the state a rewind to `position` will need. Nothing, for a network whose caches are positional.
        virtual void onSavePosition( dim_t position )
        {
            ( void )position;
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

        /// Drop every decode recording; the next passes prime and record them again. Replay stays as it was.
        void discardDecodeRecordings()
        {
            decode_replay_.recordings.clear();
            decode_tokens_replay_.recordings.clear();
            draft_replay_.recordings.clear();
        }

    private:

        /// Where a recording's life stands (DecodeGraph.md section 4.4).
        enum class ReplayStage
        {
            Unprimed,  ///< The next pass is called, doing any lazy setup and touching every buffer.
            Primed,    ///< The next pass is recorded, then replayed.
            Recorded,  ///< The next pass checks a replay against the called pass.
            Verified   ///< Every pass is replayed.
        };

        /// One recorded pass: its launches, the input tensor they read and the output they write.
        template<typename TOutput>
        struct Recording
        {
            std::unique_ptr<IDecodeRecording> recording;
            const void* input{ nullptr };
            std::unique_ptr<TOutput> output;
            ReplayStage stage{ ReplayStage::Unprimed };
        };

        /// The recordings of one kind of pass, by its number of tokens.
        template<typename TOutput>
        struct PassReplay
        {
            std::string_view pass;
            bool off{ false };
            std::map<dim_t, Recording<TOutput>> recordings;

            void reset()
            {
                off = false;
                recordings.clear();
            }
        };

        /**
         * `step` called or replayed, as the recording for its input's number of tokens stands. A recording holds the
         * input's address, so a different input tensor of that number needs a new one; another number has its own.
         * `prepare` puts in place what the pass reads that is not in the recording, and runs before every run of the
         * pass, the self-check's two included, since a pass may overwrite it.
         */
        template<typename TOutput, typename TPrepare, typename TStep>
        TOutput& replayed( PassReplay<TOutput>& replay, const TokenIndexType& input, TPrepare&& prepare, TStep&& step )
        {
            IExecutionContext& context = *this->getExecutionContext();

            prepare();

            if ( !replay_enabled_ || replay.off || context.hasActivationObserver() )
                return step();

            Recording<TOutput>& recording = replay.recordings[ input.shape()[ 1 ] ];

            if ( recording.recording && input.rawData() != recording.input )
                recording = Recording<TOutput>{};

            switch ( recording.stage )
            {
                case ReplayStage::Unprimed:
                {
                    TOutput& output = step();
                    recording.stage = ReplayStage::Primed;

                    return output;
                }

                case ReplayStage::Primed:
                    return record( replay, recording, context, input, step );

                case ReplayStage::Recorded:
                    return check( replay, recording, context, prepare, step );

                case ReplayStage::Verified:
                default:
                    recording.recording->replay();

                    return *recording.output;
            }
        }

        template<typename TOutput, typename TStep>
        TOutput& record( PassReplay<TOutput>& replay, Recording<TOutput>& recording, IExecutionContext& context,
            const TokenIndexType& input, TStep& step )
        {
            recording.recording = context.createDecodeRecording();
            TOutput* output = nullptr;

            if ( !recording.recording || !recording.recording->record( [ & ] { output = &step(); } ) )
            {
                turnOffReplay( replay, "it could not be recorded" );

                return step();
            }

            // The recording outlives the pass that made it: a later pass may run the component at another shape
            // (a log-likelihood window), and the component's view then describes that. The network keeps its
            // own description of the region the recording writes its output to.
            recording.input = input.rawData();
            recording.output = std::make_unique<TOutput>( output->view( output->shape() ) );
            recording.stage = ReplayStage::Recorded;

            recording.recording->replay();

            return *recording.output;
        }

        /**
         * The self-check (DecodeGraph.md section 5.1): the pass called, then replayed at the same
         * position from the same state. Writing the same K and V to the same cache slot twice
         * leaves the cache as it was, and a recurrent state is put back between the two, so the
         * replay sees what the called pass saw and any difference is a value the recording froze.
         */
        template<typename TOutput, typename TPrepare, typename TStep>
        TOutput& check( PassReplay<TOutput>& replay, Recording<TOutput>& recording, IExecutionContext& context,
            TPrepare& prepare, TStep& step )
        {
            holdDecodeState();

            TOutput& called = step();
            const std::vector<std::byte> called_bytes = hostBytes( called, context );

            restoreDecodeState();
            prepare();
            recording.recording->replay();
            const std::vector<std::byte> replayed_bytes = hostBytes( *recording.output, context );

            const bool identical = called.rawData() == recording.output->rawData() && called_bytes == replayed_bytes;

            if ( !identical )
            {
                turnOffReplay( replay, std::format( "a replayed {} differed from the called one", replay.pass ) );

                // The replay overwrote the output and advanced any recurrent state; the called pass,
                // from the held state, writes both again.
                restoreDecodeState();
                releaseDecodeState();
                prepare();

                return step();
            }

            releaseDecodeState();
            recording.stage = ReplayStage::Verified;

            return called;
        }

        /// The tensor's values on the host: token ids as they are, anything else widened to FP32, which every
        /// reduced-precision value converts to exactly, since the host holds no reduced-precision tensor.
        template<TensorDataType TDataType>
        std::vector<std::byte> hostBytes( const Tensor<TDataType, MR>& tensor, IExecutionContext& context )
        {
            constexpr TensorDataType kHostType = TDataType == TensorDataType::INT32 ? TensorDataType::INT32 : TensorDataType::FP32;

            auto host = toHost<kHostType>( tensor, &context );
            context.synchronize();

            const auto* first = static_cast<const std::byte*>( host.rawData() );

            return std::vector<std::byte>( first, first + host.size() * TensorDataTypeTraits<kHostType>::size_in_bytes );
        }

        template<typename TOutput>
        void turnOffReplay( PassReplay<TOutput>& replay, std::string_view reason )
        {
            replay.off = true;
            replay.recordings.clear();

            Logging::Logger::warning( std::format( "{}: replay of the {} is off for this network -- {}; every {} is called",
                this->getName(), replay.pass, reason, replay.pass ) );
        }

        dim_t cached_length_{ 0 };

        bool replay_enabled_{ false };
        PassReplay<TensorType> decode_replay_{ "decode step" };
        PassReplay<TensorType> decode_tokens_replay_{ "multi-token decode" };
        PassReplay<TokenIndexType> draft_replay_{ "draft" };
    };
}
