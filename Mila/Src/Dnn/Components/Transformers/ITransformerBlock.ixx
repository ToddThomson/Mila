/**
 * @file ITransformerBlock.ixx
 * @brief Virtual inference interface for a heterogeneous transformer block list.
 *
 * A transformer whose stack interleaves structurally different block kinds cannot hold a
 * homogeneous vector<Block> the way GptTransformer / LlamaTransformer do, so it drives its
 * layer list through this interface -- one virtual call per layer per token step, negligible
 * against the per-layer GEMMs. Implemented by both GemmaBlock instantiations (local sliding /
 * global full attention) and by both Qwen block kinds (attention / DeltaNet); see
 * Specifications/Gemma.md section 8 and Specifications/Qwen3.8.md section 8.
 *
 * Inference-only: the interface exposes the prefill/decode KV-cache path and the shared GQA
 * workspace wiring, not training forward/backward (those remain on the concrete
 * CompositeComponent if ever needed).
 */

module;
#include <memory>

export module Dnn.Components.ITransformerBlock;

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
     * @brief Polymorphic inference interface for one transformer block.
     *
     * @tparam TDeviceType Compile-time device.
     * @tparam TPrecision  Activation/compute precision (must match across the block list).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class ITransformerBlock
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        virtual ~ITransformerBlock() = default;

        /**
         * @brief Chunked prefill: process [B, T_chunk, model_dim] at an absolute offset.
         * @return Reference to the block-owned output [B, T_chunk, model_dim].
         */
        virtual TensorType& prefill( const TensorType& input, dim_t position_offset ) = 0;

        /**
         * @brief Single-token decode at an absolute position (T == 1).
         * @return Reference to the block-owned output [B, 1, model_dim].
         */
        virtual TensorType& decode( const TensorType& input, dim_t position ) = 0;

        /**
         * @brief Wire the shared GQA transient workspace (owned by the transformer).
         *
         * Called once after build, before any prefill/decode.
         */
        virtual void setState( const GqaState& state ) = 0;

        /**
         * @brief True when the block's attention supports the KV-cache inference path.
         */
        virtual bool supportsKvCache() const noexcept = 0;

        /**
         * @brief Reset the KV cache (new generation session).
         */
        virtual void resetKvCache() = 0;

        /**
         * @brief Rewind the KV cache fill position for prompt-prefix reuse.
         *
         * Keeps the cache session live; positions [0, position) stay valid.
         * @return true when the layer's attention accepted the rewind (a bounded
         * sliding-window ring refuses when the stale tail has overwritten the
         * window a continuation would attend to).
         */
        virtual bool rewindKvCache( dim_t position ) = 0;
    };
}
