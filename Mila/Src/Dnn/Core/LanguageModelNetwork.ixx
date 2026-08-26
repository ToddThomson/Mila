/**
 * @file LanguageModelNetwork.ixx
 * @brief Abstract base for language model networks.
 */

module;
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>

export module Dnn.LanguageModelNetwork;

import Dnn.Network;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Teacher-forced log-likelihood of a token sequence under a model.
     *
     * The two fields are reported separately rather than pre-averaged because the caller
     * usually sums several sequences before dividing: perplexity over a corpus is
     * exp( -total_log_probability / total_scored_positions ), and averaging per sequence
     * first would weight a short sequence like a long one.
     *
     * Accumulated in double. Per-position log-probabilities are small negative numbers and a
     * corpus contributes tens of thousands of them, so the running total is where precision
     * is actually at risk -- not in any single term.
     */
    export struct SequenceLogLikelihood
    {
        /// Summed natural log of the probability the model assigned to each actual next token.
        double total_log_probability{ 0.0 };

        /// Positions that contributed. One less than the sequence length: the first token has
        /// no preceding context, so nothing predicts it.
        dim_t scored_positions{ 0 };
    };

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
    // the architecture does not forbid one). Specifications/TransformerApiReadiness.md items
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
         * @brief Observer called with each intermediate activation during prefill.
         *
         * Diagnostic. Two loads of one model can hold byte-identical parameters and still
         * compute differently, and only the activations show where they diverge. A probe on
         * the real prefill path rather than a parallel diagnostic one, because a second
         * implementation is free to not reproduce the bug.
         *
         * `stage` is implementation-defined, not a contract: the implementing networks emit
         * "embedding" and "layer_N". Prefill only -- nothing fires during decode, so a value
         * that first goes bad during generation is invisible here.
         */
        using StageProbe = std::function<void( std::string_view stage, const TensorType& value )>;

        /**
         * @brief Install a stage probe, or clear it by passing an empty function.
         *
         * Implemented by GemmaTransformer and QwenTransformer. On any other network the
         * default accepts the probe and never fires it, so an empty result means "not
         * instrumented" and not "nothing to report" -- the two are indistinguishable from
         * the caller's side.
         */
        virtual void setStageProbe( StageProbe probe )
        {
            // REVIEW: a diagnostic hook that reached the public API to unblock artifact
            // debugging, never designed as a feature -- undocumented stage vocabulary,
            // prefill only, no test, one internal consumer found by `requires`. The silent
            // default is the worst part: a false negative in a NaN detector.
            // TransformerApiReadiness.md item 6 proposes taking it off the public base.
            (void)probe;
        }

        /**
         * @brief Inference decode -- single-token autoregressive step.
         *
         * @param input    Single token index [B, 1].
         * @param position Current sequence position (0-based).
         * @return         Logits [B, 1, vocab_size].
         */
        virtual TensorType& decode( const TokenIndexType& input, dim_t position ) = 0;

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
         * @brief Rewind the KV caches to `position` for prompt-prefix reuse
         * (PromptCaching.md). Positions [0, position) stay valid; device contents
         * are untouched.
         *
         * @return true when every layer accepted the rewind. Default: false (no
         * reuse capability); a full prefill positionally overwrites regardless,
         * so a refused or partial rewind never needs cleanup.
         */
        /**
         * @brief Teacher-forced scoring: how well the model predicted a sequence it is given.
         *
         * Runs the sequence through the prefill path and, at every position, reads the
         * probability the model assigned to the token that actually came next. Nothing is
         * sampled and nothing is generated, so the result is a property of the model and the
         * text alone -- which is what makes it usable as a quality measure between two
         * quantizations of the same weights.
         *
         * Distinct from prefill() rather than an option on it: prefill returns logits for one
         * position and its callers sample that row, so widening what prefill returns would
         * silently move which row a sampler reads.
         *
         * Requires a head built wide enough to evaluate more than the final position -- see
         * the family's head-width configuration. Implemented by QwenTransformer; the other
         * families throw until their heads take a width.
         *
         * @param input Token indices [1, T], T >= 2. Batching is not supported: the targets
         *              are the sequence's own next tokens, so two sequences in one call would
         *              need per-row lengths the shape cannot carry.
         */
        virtual SequenceLogLikelihood scoreTokens( const TokenIndexType& input )
        {
            ( void )input;
            throw std::logic_error( "LanguageModelNetwork::scoreTokens: not supported by this network" );
        }

        virtual bool rewindKvCache( dim_t position )
        {
            // NOTE: there is deliberately no resetKvCache counterpart. Starting a prefill at
            // position 0 already discards carried state in every stateful component, so a new
            // sequence needs no explicit reset. TransformerApiReadiness.md item 2.
            ( void )position;
            return false;
        }
    };
}
