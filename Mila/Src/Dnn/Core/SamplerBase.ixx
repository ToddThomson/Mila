/**
 * @file SamplerBase.ixx
 * @brief Base interface for Mila token samplers.
 *
 * Defines the abstract Sampler class template: a model-owned orchestrator tool
 * (the structural sibling of Optimizer) that selects the next token from a logits
 * tensor. Not a graph Component -- it runs after the network produces logits. See
 * Specifications/TokenSampling.md.
 */

module;
#include <cstdint>

export module Compute.SamplerBase;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.GenerateParams;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base for token samplers.
     *
     * A Sampler maps a logits tensor to a single next-token id. The device-dispatched
     * TokenSampler is the one concrete sampler; the base is retained as the seam for a
     * future stateful strategy (e.g. Mirostat). The model owns the sampler and shares
     * its ExecutionContext; the sampler is invoked by the generation loop, never by the
     * operation graph.
     *
     * @tparam TDeviceType Device the logits reside on (Cpu or Cuda).
     * @tparam TPrecision  Logits tensor precision (e.g. FP32, BF16).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Sampler
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TokenTensor = Tensor<TensorDataType::INT32, MR>;

        virtual ~Sampler() = default;

        /**
         * @brief Select the next token from the final row of a logits tensor.
         *
         * Samples from the last position of @p logits (last dim == vocab_size), writing
         * the chosen int32 token into @p token_out on the device stream and returning the
         * same value on the host after the readback. @p token_out is the caller-owned
         * [1, 1] INT32 device tensor (the model's decode-input buffer), so the sampled
         * token is already in place for the next decode step.
         *
         * @param logits    Device logits tensor; the last `vocab_size` elements are the
         *                   active distribution. Read in place, never copied to host.
         * @param token_out Device INT32 [1, 1] tensor; written in place.
         * @param params    Per-call sampling parameters (temperature / top_k / top_p / seed).
         * @return The sampled token id (host value, after device readback).
         */
        virtual int32_t sample(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params ) = 0;
    };
}
