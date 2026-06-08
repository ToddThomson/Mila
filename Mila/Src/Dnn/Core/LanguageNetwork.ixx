/**
 * @file LanguageNetwork.ixx
 * @brief Abstract base for language model networks.
 *
 * LanguageNetwork sits between Network and concrete transformer implementations
 * (LlamaTransformer, GptTransformer). It defines the virtual compute interface —
 * forward, backward, prefill, and decode — that LanguageModel uses to drive the
 * autoregressive generation loop without knowing the concrete network type or its
 * quantization policy template parameters.
 *
 * The virtual boundary here is intentionally coarse: one virtual dispatch per
 * decode step is negligible cost, and it lets LlamaModel and GptModel remain
 * free of quantization and architecture template parameters that belong only
 * at the transformer level.
 *
 * ## Hierarchy
 *
 *   Network<TDev, TPrec>
 *     └─ LanguageNetwork<TDev, TPrec>              [this file]
 *          └─ LlamaTransformer<TDev, TPrec, TWeightQuantization, TKvCachePolicy>
 *          └─ GptTransformer<TDev, TPrec>
 */

module;
#include <string>

export module Dnn.LanguageNetwork;

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

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LanguageNetwork : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TensorDataType::INT32, MR>;

        explicit LanguageNetwork( const std::string& name )
            : NetworkBase( name )
        {}

        ~LanguageNetwork() override = default;

        /**
         * @brief Full-sequence forward pass.
         *
         * @param input  Token indices [B, T].
         * @return       Logits [B, T, vocab_size].
         */
        virtual TensorType& forward( const TokenIndexType& input ) = 0;

        /**
         * @brief Full backward pass (training).
         *
         * @param input       Token indices [B, T].
         * @param output_grad Gradient of the loss w.r.t. logits.
         * @return            Gradient w.r.t. the input embeddings.
         */
        virtual TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad ) = 0;

        /**
         * @brief Inference prefill — process full prompt and populate the KV cache.
         *
         * @param input  Full prompt token indices [B, T].
         * @return       Logits for the last token position.
         */
        virtual TensorType& prefill( const TokenIndexType& input ) = 0;

        /**
         * @brief Inference decode — single-token autoregressive step.
         *
         * @param input    Single token index [B, 1].
         * @param position Current sequence position (0-based).
         * @return         Logits [B, 1, vocab_size].
         */
        virtual TensorType& decode( const TokenIndexType& input, int position ) = 0;
    };
}
