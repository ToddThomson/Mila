/**
 * @file QuantizationDispatch.ixx
 * @brief The one place a runtime quantization setting becomes a compile-time policy.
 *
 * Every model entry point that has to reach a template instantiation from a ModelConfig
 * routes through here, so a newly supported mode is added once rather than per model per
 * entry point. See Specifications/MemoryFootprint.md.
 */

export module Dnn.Models.QuantizationDispatch;

import std;
import Dnn.LanguageModelConfig;
import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.PerTokenKvFp8;
import Compute.DeviceType;
import Compute.OperationTraits;
import Compute.OperationType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Refuse a weight format with no Linear operation on this device at this precision.
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TResult>
    TResult refuseWeightFormat( std::string_view caller, std::string_view format )
    {
        throw std::runtime_error( std::format( "{}: {} weights have no {} operation at {} compute precision",
            caller, format, deviceTypeToString( TDeviceType ), tensorDataTypeToString( TPrecision ) ) );
    }

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
     * @tparam TDeviceType     The device the caller's model runs on.
     * @tparam TPrecision      Compute precision. A format is built only where its Linear operation exists on
     *                         this device at this precision (`OperationSupported`), and refused otherwise, so
     *                         a format a device lacks is never instantiated for it.
     * @tparam TResult         What the action returns -- a model, a plan or a footprint.
     * @tparam kFp4GroupSize   The FP4 group the caller's geometry requires. A compile-time value so
     *                         a family instantiates only the group it can build.
     *
     * @param weight_quantization Runtime weight-quantization setting to resolve to a policy type.
     * @param caller Prefix for error messages, e.g. "GemmaModel::load".
     * @param action Its call operator is invoked with the weight policy as its one template argument.
     *
     * @throws std::runtime_error if the requested mode has no operation on this device at this precision.
     */
    export template<
        DeviceType TDeviceType,
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
                if constexpr ( OperationSupported<OperationType::LinearOp, TDeviceType, TPrecision, PerGroupFp4<kFp4GroupSize>> )
                {
                    return action.template operator()<PerGroupFp4<kFp4GroupSize>>();
                }
                else
                {
                    return refuseWeightFormat<TDeviceType, TPrecision, TResult>( caller, "FP4" );
                }

            case WeightQuantization::FP8:
                if constexpr ( OperationSupported<OperationType::LinearOp, TDeviceType, TPrecision, PerChannelFp8<>> )
                {
                    return action.template operator()<PerChannelFp8<>>();
                }
                else
                {
                    return refuseWeightFormat<TDeviceType, TPrecision, TResult>( caller, "FP8" );
                }

            case WeightQuantization::Q4_0:
                if constexpr ( OperationSupported<OperationType::LinearOp, TDeviceType, TPrecision, PerGroupInt4<32>> )
                {
                    return action.template operator()<PerGroupInt4<32>>();
                }
                else
                {
                    return refuseWeightFormat<TDeviceType, TPrecision, TResult>( caller, "Q4_0" );
                }

            case WeightQuantization::Plan:
            case WeightQuantization::Fp4Codebook3:
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
