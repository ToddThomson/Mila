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
import Dnn.Quantization.KvCache.PerTokenKvFp8;
import Compute.DeviceType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Resolve a runtime weight-quantization setting to a policy type and invoke an action.
     *
     * This bridge existed in four copies -- load and footprint, for each of Gemma and Llama --
     * which differed only in the name in their error messages. Four copies of a
     * runtime-to-compile-time mapping is four chances for a newly supported mode to reach the
     * load path and not the footprint path, which would make a model report a figure it does not
     * allocate. That is the exact defect class the footprint work exists to prevent, so the
     * mapping lives once.
     *
     * @tparam TPrecision      Compute precision; quantized weights require BF16.
     * @tparam TResult         What the action returns -- a model, a plan or a footprint.
     * @tparam kFp4GroupSize   The FP4 group the caller's geometry requires. A compile-time value so
     *                         a family instantiates only the group it can build.
     *
     * @param weight_quantization Runtime weight-quantization setting to resolve to a policy type.
     * @param caller Prefix for error messages, e.g. "GemmaModel::load".
     * @param action Its call operator is invoked with the weight policy as its one template argument.
     *
     * @throws std::runtime_error if the requested mode is unsupported at this precision.
     */
    export template<
        TensorDataType TPrecision,
        typename TResult,
        int kFp4GroupSize = 128,
        typename TAction>
    TResult dispatchWeightQuantization(
        WeightQuantization weight_quantization,
        std::string_view caller,
        TAction&& action )
    {
        switch ( weight_quantization )
        {
            case WeightQuantization::FP4:
                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    return action.template operator()<PerGroupFp4<kFp4GroupSize>>();
                }
                else
                {
                    throw std::runtime_error( std::format(
                        "{}: FP4 weight quantization requires BF16 compute precision", caller ) );
                }

            case WeightQuantization::FP8:
                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    return action.template operator()<PerChannelFp8<>>();
                }
                else
                {
                    throw std::runtime_error( std::format(
                        "{}: FP8 weight quantization requires BF16 compute precision", caller ) );
                }

            case WeightQuantization::Q4_0:
                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    return action.template operator()<PerGroupInt4<32>>();
                }
                else
                {
                    throw std::runtime_error( std::format(
                        "{}: Q4_0 weight quantization requires BF16 compute precision", caller ) );
                }

            case WeightQuantization::Plan:
                // Explicit rather than left to the default: a per-role plan is a property of
                // one family's design, and this dispatcher yields a single uniform policy. If
                // Plan fell through it would build an UNQUANTIZED body and report success --
                // the silent-wrong-build failure this file exists to prevent. A family that
                // has a plan resolves it in its own dispatcher and never arrives here.
                throw std::runtime_error( std::format(
                    "{}: this model has no designed precision plan; ask for a uniform "
                    "quantization mode, or load a family that defines one", caller ) );

            case WeightQuantization::None:
            default:
                return action.template operator()<NoWeightQuant>();
        }
    }

    /**
     * @brief Resolve a runtime KV-cache setting to the policy of a family's full-context layers.
     *
     * Only full-context caches are resolved here: a bounded sliding-window ring is the family's own
     * architecture, not a deployment choice (Quantization.md, KV decision 4). FP8 is served only by the
     * CUDA flash and fused-decode kernels at BF16 compute, so any other build refuses it rather than
     * quietly building a BF16 cache under a plan that reports FP8.
     *
     * @tparam TDeviceType The device the caller's model runs on.
     * @tparam TPrecision  Compute precision.
     * @tparam TResult     What the action returns -- a model, a plan or a footprint.
     *
     * @param kv_cache_compression Runtime KV-cache setting to resolve to a policy type.
     * @param caller Prefix for error messages, e.g. "GemmaModel::load".
     * @param action Its call operator is invoked with the KV policy as its one template argument.
     *
     * @throws std::runtime_error when FP8 is asked of a device or precision without an FP8 cache.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, typename TResult, typename TAction>
    TResult dispatchKvCacheCompression(
        KvCacheCompression kv_cache_compression,
        std::string_view caller,
        TAction&& action )
    {
        if ( kv_cache_compression == KvCacheCompression::FP8 )
        {
            if constexpr ( TDeviceType == DeviceType::Cuda && TPrecision == TensorDataType::BF16 )
            {
                return action.template operator()<PerTokenKvFp8<>>();
            }
            else
            {
                throw std::runtime_error( std::format(
                    "{}: an FP8 KV cache needs a CUDA device and BF16 compute precision", caller ) );
            }
        }

        return action.template operator()<NoKvCompression>();
    }
}
