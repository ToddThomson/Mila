/**
 * @file QuantizationDispatch.ixx
 * @brief The one place a runtime quantization setting becomes a compile-time policy.
 *
 * Every model entry point that has to reach a template instantiation from a ModelConfig
 * routes through here, so a newly supported mode is added once rather than per model per
 * entry point. See Specifications/MemoryFootprint.md.
 */

module;
#include <format>
#include <stdexcept>
#include <string_view>

export module Dnn.Models.QuantizationDispatch;

import Dnn.LanguageModelConfig;
import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Resolve a runtime quantization setting to a policy type and invoke an action.
     *
     * This bridge existed in four copies -- load and footprint, for each of Gemma and Llama --
     * which differed only in the KV policy and the name in their error messages. Four copies of
     * a runtime-to-compile-time mapping is four chances for a newly supported mode to reach the
     * load path and not the footprint path, which would make a model report a figure it does not
     * allocate. That is the exact defect class the footprint work exists to prevent, so the
     * mapping lives once.
     *
     * @tparam TPrecision      Compute precision; quantized weights require BF16.
     * @tparam TKvCachePolicy  KV policy the caller's chassis uses -- Gemma's sliding-window
     *                         ring, or NoKvCompression. Not derived from the config: it is an
     *                         architecture property, not a deployment choice.
     * @tparam TResult         What the action returns -- a model, or a MemoryStats.
     *
     * @param weight_quantization Runtime weight-quantization setting to resolve to a policy type.
     * @param kv_cache_compression Runtime KV-cache setting accompanying it.
     * @param caller Prefix for error messages, e.g. "GemmaModel::fromPretrained".
     * @param action Invoked as action.template operator()<TWeightQuantization, TKvCachePolicy>().
     *
     * @throws std::runtime_error if the requested combination is unsupported.
     */
    export template<
        TensorDataType TPrecision,
        KvCachePolicy TKvCachePolicy,
        typename TResult,
        typename TAction>
    TResult dispatchWeightQuantization(
        WeightQuantization weight_quantization,
        KvCacheCompression kv_cache_compression,
        std::string_view caller,
        TAction&& action )
    {
        switch ( weight_quantization )
        {
            case WeightQuantization::FP4:
                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    return action.template operator()<PerGroupFp4<128>, TKvCachePolicy>();
                }
                else
                {
                    throw std::runtime_error( std::format(
                        "{}: FP4 weight quantization requires BF16 compute precision", caller ) );
                }

            case WeightQuantization::FP8:
                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    return action.template operator()<PerChannelFp8<>, TKvCachePolicy>();
                }
                else
                {
                    throw std::runtime_error( std::format(
                        "{}: FP8 weight quantization requires BF16 compute precision", caller ) );
                }

            case WeightQuantization::None:
            default:
                // KV compression is only consulted on the unquantized path because that is where
                // it was ever refused; the quantized paths accepted either value and ignored it.
                // Behaviour preserved exactly -- see the enum, which has no third value.
                if ( kv_cache_compression == KvCacheCompression::FP8 )
                {
                    throw std::runtime_error( std::format(
                        "{}: FP8 KV cache compression is not yet supported", caller ) );
                }

                return action.template operator()<NoWeightQuant, TKvCachePolicy>();
        }
    }
}
