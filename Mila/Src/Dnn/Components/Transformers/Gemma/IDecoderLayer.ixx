/**
 * @file IDecoderLayer.ixx
 * @brief Virtual inference interface for a heterogeneous decoder layer list.
 *
 * Gemma interleaves two structurally different block types (local sliding and
 * global full attention), so its transformer cannot hold a homogeneous
 * vector<Block> the way GptTransformer / LlamaTransformer do. Both GemmaBlock
 * instantiations implement this interface; the transformer drives the layer list
 * polymorphically (one virtual call per layer per token step -- negligible against
 * the per-layer GEMMs). See Specifications/Gemma.md section 8.
 *
 * Inference-only: Gemma is an inference target, so the interface exposes the
 * prefill/decode KV-cache path and the shared-GQA-workspace wiring, not training
 * forward/backward (those remain on the concrete CompositeComponent if ever needed).
 */

module;
#include <memory>

export module Dnn.Components.IDecoderLayer;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.GqaState;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Polymorphic inference interface for one decoder layer.
     *
     * @tparam TDeviceType Compile-time device.
     * @tparam TPrecision  Activation/compute precision (must match across the layer list).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class IDecoderLayer
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        virtual ~IDecoderLayer() = default;

        /**
         * @brief Chunked prefill: process [B, T_chunk, model_dim] at an absolute offset.
         * @return Reference to the block-owned output [B, T_chunk, model_dim].
         */
        virtual TensorType& prefill( const TensorType& input, int position_offset ) = 0;

        /**
         * @brief Single-token decode at an absolute position (T == 1).
         * @return Reference to the block-owned output [B, 1, model_dim].
         */
        virtual TensorType& decode( const TensorType& input, int position ) = 0;

        /**
         * @brief Wire the shared GQA transient workspace (owned by the transformer).
         *
         * Called once after build, before any prefill/decode.
         */
        virtual void setState( const GqaState& state ) = 0;

        /**
         * @brief True when the block's attention supports the KV-cache inference path.
         */
        virtual bool supportsKVCache() const noexcept = 0;

        /**
         * @brief Reset the KV cache (new generation session).
         */
        virtual void resetKVCache() = 0;
    };
}
