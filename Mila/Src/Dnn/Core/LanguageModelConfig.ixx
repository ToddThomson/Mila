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
 * load() projects LanguageModelConfig into BuildContext once --
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
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

export module Dnn.LanguageModelConfig;

export import Dnn.WeightQuantization;
export import Dnn.KvCacheCompression;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
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
         * Required before passing the config to load(). RoPE embeddings
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
         * @brief Q4_0 quantization -- Q4_0 weights, FP8 KV cache.
         *
         * Maps to PerGroupInt4<32> on Linear. The format of quantization-aware checkpoints
         * trained for Q4_0, such as Gemma 4's.
         */
        TDerived& withQ4_0Quantization()
        {
            weight_quantization_ = WeightQuantization::Q4_0;
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
                        case WeightQuantization::Q4_0: return "Q4_0 (PerGroupInt4<32>)";
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

            return result;
        }

    protected:

        // =====================================================================
        // Data members -- accessible to derived configs
        // =====================================================================

        dim_t              context_length_{ 0 };
        WeightQuantization weight_quantization_{ WeightQuantization::None };
        KvCacheCompression kv_cache_compression_{ KvCacheCompression::None };
    };
}
