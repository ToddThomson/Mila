/**
 * @file LanguageModelConfig.ixx
 * @brief CRTP base configuration for all deployable Mila language models.
 *
 * LanguageModelConfig<TDerived> owns the deployment concerns that are universal
 * across all language model architectures:
 *
 *  1. context_length        -- maximum sequence length the model is built for.
 *                             RoPE embeddings and KV cache buffers are sized to this.
 *
 *  2. WeightQuantization    -- weight storage and matmul strategy for Linear components.
 *                             Defaults to WeightQuantization::None (BF16 weights).
 *
 *  3. KvCacheCompression    -- KV cache storage and compression strategy for
 *                             GroupedQueryAttention components.
 *                             Defaults to KvCacheCompression::None (no compression).
 *
 * ## CRTP Pattern
 *
 * All fluent setters return TDerived& so that chains work correctly across
 * both base and derived methods without casting at the call site:
 *
 * @code
 * QwenModelConfig config = QwenModelConfig( context_length )
 *     .withFP8Quantization()   // returns QwenModelConfig&
 *     .withThinkingMode();     // returns QwenModelConfig&
 * @endcode
 *
 * ## Relationship to ModelConfig
 *
 * ModelConfig<TDevice, TPrecision> is the structural base for all Mila models.
 * LanguageModelConfig is the deployment configuration counterpart for the
 * language model branch of that hierarchy. Vision model configurations would
 * derive from a sibling VisionModelConfig<TDerived>, not from this class.
 *
 * ## Relationship to BuildContext
 *
 * LanguageModelConfig is the public API surface for deployment configuration.
 * BuildContext is the internal carrier through the component tree.
 * fromPretrained() projects LanguageModelConfig into BuildContext once --
 * they are never the same object.
 *
 * ## Quantization Presets vs Fine-Grained Control
 *
 * Convenience preset methods express common deployment decisions in user
 * vocabulary. Fine-grained setters are available for atypical configurations:
 *
 * @code
 * // Preset -- FP8 weights + FP8 KV cache
 * LlamaModelConfig config = LlamaModelConfig( context_length )
 *     .withFP8Quantization();
 *
 * // Fine-grained -- FP4 weights, no KV compression
 * LlamaModelConfig config = LlamaModelConfig( context_length )
 *     .withWeightQuantization( WeightQuantization::FP4 )
 *     .withKvCacheCompression( KvCacheCompression::None );
 * @endcode
 */

module;
#include <stdexcept>
#include <string>

export module Dnn.LanguageModelConfig;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    // =========================================================================
    // WeightQuantization
    // =========================================================================

    /**
     * @brief Weight storage and matmul strategy for Linear components.
     *
     * Maps to the TWeightQuant template parameter on Linear and CudaLinearOp
     * via the fromPretrained() runtime->compile-time bridge. The mapping is:
     *
     *   None  -> NoWeightQuant        (BF16 weights, standard cuBLASLt plan)
     *   FP8   -> PerChannelFp8<>      (FP8_E4M3 weights, per-channel float32 scales)
     *   FP4   -> PerGroupFp4<>        (future)
     *
     * This enum is Mila API vocabulary. Callers set it via fluent methods on
     * the concrete model config -- they do not interact with the policy structs
     * directly.
     */
    export enum class WeightQuantization
    {
        None,   ///< BF16 weights -- default; no quantization overhead.
        FP8,    ///< FP8_E4M3 per-channel weight quantization -- Alpha.5 target.
        FP4,    ///< Per-group FP4 weight quantization -- future target.

        /**
         * The family's own designed per-role allocation, rather than one uniform format.
         *
         * The three values above name a STORAGE FORMAT that applies to every Linear alike.
         * This one does not name a format at all: it says "build this model the way its
         * designers allocated its bits", and which formats that means is the family's to
         * define -- Qwen 3.8 spends 2.5 bits on the feed-forward gate/up pair and 4.125 on
         * full attention (Specifications/Qwen3.8.md section 5).
         *
         * A family with no plan must REFUSE this value rather than fall back to a uniform
         * policy, which is why dispatchWeightQuantization handles it explicitly.
         */
        Plan,
    };

    /**
     * @brief The scheme name recorded in an artifact and in its manifest.
     *
     * It lives beside the enum because it is written by the model that saves the artifact and
     * read by the tool that packages it, and those two must agree exactly: the load side
     * refuses an artifact whose scheme disagrees with the build's compile-time policy, since
     * the bytes are packed differently per scheme and reinterpreting them produces a model
     * that runs and is wrong. It was previously spelled out in both places.
     */
    export inline std::string weightQuantizationName( WeightQuantization quantization )
    {
        switch ( quantization )
        {
            case WeightQuantization::FP4: return "per_group_fp4_128";
            case WeightQuantization::FP8: return "per_channel_fp8_e4m3";

            // A plan's artifact scheme is the FAMILY's, not this enum's -- Qwen 3.8 writes
            // "codebook" because its sub-4-bit rows are what a load cannot reconstruct. One
            // family has a plan today, so the mapping is stated here; a second one with a
            // different scheme is what forces it to move behind a family accessor.
            case WeightQuantization::Plan: return "codebook";

            case WeightQuantization::None:
            default:
                return "none";
        }
    }

    /**
     * @brief True when a load can derive this format from reference weights.
     *
     * The distinction the artifact check turns on. FP4 and FP8 are computed from the weights
     * at load time -- absmax scales and a format-defined level table -- so a BF16 artifact is
     * a legitimate source for them, and every family already relies on that: Qwen's own
     * packed artifact carries codebook tensors only and quantizes its attention and head
     * projections on load (Qwen3.8.md section 8).
     *
     * A plan's codebooks are the opposite case. They are FITTED offline against calibration
     * data, so nothing in a BF16 tensor recovers them and the artifact must carry them.
     *
     * Refusing both alike would be the safe-looking answer and the wrong one: it would make a
     * uniform FP4 build of any family unreachable without a repack that adds nothing, which
     * is exactly what the Phase 5 FP4 oracle needs to load.
     */
    export inline bool isDerivableFromReferenceWeights( WeightQuantization quantization )
    {
        return quantization == WeightQuantization::FP4
            || quantization == WeightQuantization::FP8;
    }

    // =========================================================================
    // KvCacheCompression
    // =========================================================================

    /**
     * @brief KV cache storage and compression strategy for GroupedQueryAttention.
     *
     * Maps to the TKvPolicy template parameter on GroupedQueryAttention and
     * CudaGqaOp via the fromPretrained() runtime->compile-time bridge. The mapping is:
     *
     *   None  -> NoKvCompression      (BF16 cache, no compression overhead)
     *   FP8   -> PerChannelKvFp8<>   (FP8_E4M3 cache, per-head per-token float32 scales)
     *
     * New compression algorithms (SlidingWindow, LowRank, TurboQuant) add a
     * value here and a corresponding policy struct in KvCache.QuantPolicy --
     * no other changes are required at this level.
     */
    export enum class KvCacheCompression
    {
        None,   ///< No compression -- default; BF16 KV cache.
        FP8,    ///< FP8_E4M3 per-head per-token KV cache compression -- Alpha.6 target.
    };

    // =========================================================================
    // LanguageModelConfig<TDerived>
    // =========================================================================

    /**
     * @brief CRTP base configuration for all deployable Mila language models.
     *
     * @tparam TDerived  Concrete config type (e.g. LlamaModelConfig). All fluent
     *                   setters return TDerived& to support unbroken chain syntax
     *                   across base and derived methods.
     */
    export template<typename TDerived>
        struct LanguageModelConfig
    {
        // =====================================================================
        // Construction
        // =====================================================================

        LanguageModelConfig() = default;

        /**
         * @brief Construct with a required context length.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        explicit LanguageModelConfig( dim_t context_length )
            : context_length_( context_length )
        {
            if ( context_length == 0 )
            {
                throw std::invalid_argument(
                    "LanguageModelConfig: context_length must be greater than zero" );
            }
        }

        // =====================================================================
        // Fine-grained fluent setters
        // =====================================================================

        /**
         * @brief Set the maximum sequence length.
         *
         * Required before passing the config to fromPretrained(). RoPE embeddings
         * and KV cache buffers are sized to this value at build time.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        TDerived& withContextLength( dim_t context_length )
        {
            if ( context_length == 0 )
            {
                throw std::invalid_argument(
                    "LanguageModelConfig: context_length must be greater than zero" );
            }

            context_length_ = context_length;
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief Positions the language-model head evaluates per pass. Default 1.
         *
         * Generation reads a logit only at the final position, which is what the default
         * pays for. Teacher-forced scoring needs one at every position, and a whole prefill
         * chunk of logit rows does not fit -- at a 248,320 vocabulary a BF16 row is
         * 0.474 MiB -- so a scoring deployment raises this to the number of rows it can
         * afford and the head is evaluated in windows of that width.
         *
         * Sizes buffers at build time exactly as withContextLength does, and like it,
         * describes the deployment rather than the checkpoint. Families that have not
         * implemented scoring ignore it.
         *
         * @param positions  Head width in positions. Must be > 0.
         * @throws std::invalid_argument if positions is zero.
         */
        TDerived& withLanguageModelHeadPositions( dim_t positions )
        {
            if ( positions <= 0 )
            {
                throw std::invalid_argument(
                    "LanguageModelConfig: language_model_head_positions must be greater than zero" );
            }

            language_model_head_positions_ = positions;
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief Set the weight quantization mode independently.
         *
         * Use when the desired weight quantization does not pair with the
         * default KV cache compression of a preset, or when a preset does
         * not exist for the desired combination.
         *
         * @param wq  Weight quantization mode to apply.
         */
        TDerived& withWeightQuantization( WeightQuantization wq )
        {
            weight_quantization_ = wq;
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief Set the KV cache compression mode independently.
         *
         * Use when the desired KV cache compression does not pair with the
         * default weight quantization of a preset, or when a preset does
         * not exist for the desired combination.
         *
         * @param kv  KV cache compression mode to apply.
         */
        TDerived& withKvCacheCompression( KvCacheCompression kv )
        {
            kv_cache_compression_ = kv;
            return static_cast<TDerived&>(*this);
        }

        // =====================================================================
        // Convenience preset fluent setters
        // =====================================================================

        /**
         * @brief Full precision -- BF16 weights, BF16 KV cache.
         *
         * Resets both quantization axes to their defaults. Useful for
         * explicitly documenting intent or overriding a previously set preset.
         */
        TDerived& withFullPrecision()
        {
            weight_quantization_ = WeightQuantization::None;
            kv_cache_compression_ = KvCacheCompression::None;
            
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief FP8 quantization -- FP8 weights, FP8 KV cache.
         *
         * Maps to PerChannelFp8<> on Linear and PerChannelKvFp8<> on
         * GroupedQueryAttention. Good quality/compression tradeoff for
         * standard inference on Ada Lovelace and later.
         */
        TDerived& withFP8Quantization()
        {
            weight_quantization_ = WeightQuantization::FP8;
            kv_cache_compression_ = KvCacheCompression::FP8;
            
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief FP4 quantization -- FP4 weights, FP8 KV cache.
         *
         * Maps to PerGroupFp4<> on Linear (future) and PerChannelKvFp8<> on
         * GroupedQueryAttention. Aggressive compression; some quality loss
         * relative to FP8. FP4 KV cache is not a Mila target.
         */
        TDerived& withFP4Quantization()
        {
            weight_quantization_ = WeightQuantization::FP4;
            kv_cache_compression_ = KvCacheCompression::FP8;
            
            return static_cast<TDerived&>(*this);
        }

        /**
         * @brief The family's designed per-role allocation, from a pre-quantized artifact.
         *
         * Unlike the two above, this sets no KV compression: a plan allocates WEIGHT bits,
         * and Qwen 3.8's baseline pairs its 2.90-bit body with a BF16 KV cache, which fits
         * at 16K without compression (Qwen3.8.md section 5). A deployment that wants FP8 KV
         * on top asks for it separately.
         *
         * There is no quantize-on-load path here and there cannot be one: a codebook is
         * fitted offline against calibration data, so the artifact must already carry the
         * codes. A load refuses an artifact whose scheme is not the compiled one.
         */
        TDerived& withPrecisionPlan()
        {
            weight_quantization_ = WeightQuantization::Plan;

            return static_cast<TDerived&>(*this);
        }

        // =====================================================================
        // Accessors
        // =====================================================================

        dim_t getContextLength() const noexcept
        {
            return context_length_;
        }

        dim_t getLanguageModelHeadPositions() const noexcept
        {
            return language_model_head_positions_;
        }

        WeightQuantization getWeightQuantization() const noexcept
        {
            return weight_quantization_;
        }

        KvCacheCompression getKvCacheCompression() const noexcept
        {
            return kv_cache_compression_;
        }

        // =====================================================================
        // Diagnostics
        // =====================================================================

        /**
         * @brief Produce the base fields portion of a toString() summary.
         *
         * Concrete model configs call this from their own toString()
         * implementation and append architecture-specific fields.
         */
        std::string baseToString() const
        {
            auto weightQuantStr = []( WeightQuantization wq ) -> std::string
                {
                    switch ( wq )
                    {
                        case WeightQuantization::None: return "None (BF16)";
                        case WeightQuantization::FP8:  return "FP8 (PerChannelFp8)";
                        case WeightQuantization::FP4:  return "FP4 (PerGroupFp4)";
                        case WeightQuantization::Plan: return "Plan (per-role allocation)";
                        default:                       return "Unknown";
                    }
                };

            auto kvCacheStr = []( KvCacheCompression kv ) -> std::string
                {
                    switch ( kv )
                    {
                        case KvCacheCompression::None: return "None (BF16)";
                        case KvCacheCompression::FP8:  return "FP8 (PerChannelKvFp8)";
                        default:                       return "Unknown";
                    }
                };

            std::string result;
            result += "  context_length:      " + std::to_string( context_length_ ) + "\n";
            result += "  weight_quantization: " + weightQuantStr( weight_quantization_ ) + "\n";
            result += "  kv_cache_compression:" + kvCacheStr( kv_cache_compression_ ) + "\n";
            result += "  lm_head_positions:   " + std::to_string( language_model_head_positions_ ) + "\n";

            return result;
        }

    protected:

        // =====================================================================
        // Data members -- accessible to derived configs
        // =====================================================================

        dim_t              context_length_{ 0 };
        WeightQuantization weight_quantization_{ WeightQuantization::None };
        KvCacheCompression kv_cache_compression_{ KvCacheCompression::None };

        // 1 is what generation reads; only a scoring deployment raises it.
        dim_t              language_model_head_positions_{ 1 };
    };
}
