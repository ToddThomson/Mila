/**
 * @file LanguageModelConfig.ixx
 * @brief CRTP base configuration for all deployable Mila language models.
 *
 * LanguageModelConfig<TDerived> owns the deployment concerns that are universal
 * across all language model architectures:
 *
 *  1. context_length        — maximum sequence length the model is built for.
 *                             RoPE embeddings and KV cache buffers are sized to this.
 *
 *  2. WeightQuantization    — weight storage and matmul strategy for Linear components.
 *                             Defaults to WeightQuantization::None (BF16 weights).
 *
 *  3. KvCacheCompression    — KV cache storage and compression strategy for
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
 * fromPretrained() projects LanguageModelConfig into BuildContext once —
 * they are never the same object.
 *
 * ## Quantization Presets vs Fine-Grained Control
 *
 * Convenience preset methods express common deployment decisions in user
 * vocabulary. Fine-grained setters are available for atypical configurations:
 *
 * @code
 * // Preset — FP8 weights + FP8 KV cache
 * LlamaModelConfig config = LlamaModelConfig( context_length )
 *     .withFP8Quantization();
 *
 * // Fine-grained — FP4 weights, no KV compression
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
     * via the fromPretrained() runtime→compile-time bridge. The mapping is:
     *
     *   None  → NoWeightQuant        (BF16 weights, standard cuBLASLt plan)
     *   FP8   → PerChannelFp8<>      (FP8_E4M3 weights, per-channel float32 scales)
     *   FP4   → PerGroupFp4<>        (future)
     *
     * This enum is Mila API vocabulary. Callers set it via fluent methods on
     * the concrete model config — they do not interact with the policy structs
     * directly.
     */
    export enum class WeightQuantization
    {
        None,   ///< BF16 weights — default; no quantization overhead.
        FP8,    ///< FP8_E4M3 per-channel weight quantization — Alpha.5 target.
        FP4,    ///< Per-group FP4 weight quantization — future target.
    };

    // =========================================================================
    // KvCacheCompression
    // =========================================================================

    /**
     * @brief KV cache storage and compression strategy for GroupedQueryAttention.
     *
     * Maps to the TKvPolicy template parameter on GroupedQueryAttention and
     * CudaGqaOp via the fromPretrained() runtime→compile-time bridge. The mapping is:
     *
     *   None  → NoKvCompression      (BF16 cache, no compression overhead)
     *   FP8   → PerChannelKvFp8<>   (FP8_E4M3 cache, per-head per-token float32 scales)
     *
     * New compression algorithms (SlidingWindow, LowRank, TurboQuant) add a
     * value here and a corresponding policy struct in KvCache.QuantPolicy —
     * no other changes are required at this level.
     */
    export enum class KvCacheCompression
    {
        None,   ///< No compression — default; BF16 KV cache.
        FP8,    ///< FP8_E4M3 per-head per-token KV cache compression — Alpha.6 target.
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
         * @brief Full precision — BF16 weights, BF16 KV cache.
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
         * @brief FP8 quantization — FP8 weights, FP8 KV cache.
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
         * @brief FP4 quantization — FP4 weights, FP8 KV cache.
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

        // =====================================================================
        // Accessors
        // =====================================================================

        dim_t getContextLength() const noexcept
        {
            return context_length_;
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
            return result;
        }

    protected:

        // =====================================================================
        // Data members — accessible to derived configs
        // =====================================================================

        dim_t              context_length_{ 0 };
        WeightQuantization weight_quantization_{ WeightQuantization::None };
        KvCacheCompression kv_cache_compression_{ KvCacheCompression::None };
    };

}  // namespace Mila::Dnn
