/**
 * @file TokenSampler.ixx
 * @brief Device-agnostic token sampler facade.
 *
 * Model-owned orchestrator tool (sibling of the Optimizer facade). Resolves the
 * device SamplingOp through OperationTraits, owns the host RNG, and performs the
 * 4-byte device->host token readback. See Specifications/TokenSampling.md.
 */

module;
#include <memory>
#include <random>
#include <cstdint>
#include <stdexcept>

export module Dnn.Samplers.TokenSampler;

import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Compute.SamplerBase;
import Compute.OperationTraits;
import Compute.OperationType;
import Dnn.Tensor;
import Dnn.TensorOps;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The standard token sampler: temperature / top-k / top-p multinomial.
     *
     * Dispatches to the device SamplingOp resolved by OperationTraits (CudaSamplingOp /
     * CpuSamplingOp). Shares the model's ExecutionContext so the op runs on the decode
     * stream and writes the token in place. The host-drawn uniform is generated here so
     * the op stays pure and deterministic (Phase A: greedy only).
     *
     * @tparam TDeviceType Device the logits reside on.
     * @tparam TPrecision  Logits precision (FP32 or BF16).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class TokenSampler : public Sampler<TDeviceType, TPrecision>
    {
    public:
        using Base = Sampler<TDeviceType, TPrecision>;
        using TokenTensor = typename Base::TokenTensor;
        using SamplingOpType = typename OperationTraits<OperationType::SamplingOp, TDeviceType, TPrecision>::type;

        TokenSampler( IExecutionContext* context, const SamplingConfig& config )
            : context_( context ),
              config_( config ),
              host_token_( Device::Cpu(), shape_t{ 1, 1 } ),
              rng_( std::random_device{}() )
        {
            if (!context_)
            {
                throw std::invalid_argument( "TokenSampler: ExecutionContext cannot be null" );
            }

            config_.validate();
            op_ = std::make_shared<SamplingOpType>( context_, config_ );
        }

        int32_t sample(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params ) override
        {
            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            const float r = dist( rng_ );

            op_->forward( logits, token_out, params, r );

            // 4-byte device->host readback. Phase A runs the sampling kernel on the default
            // stream, so a synchronous copy is ordered after it and leaves the host token
            // valid before the caller's stop-check / on_token callback. (Stream-sharing with
            // the decode stream is Phase D.)
            copy( token_out, host_token_ );

            return host_token_.data()[ 0 ];
        }

        /**
         * @brief Enqueue one sampling step without waiting for the token readback.
         *
         * Decode-ahead half of the pipelined generation loop: the op samples on the
         * model's stream (ordered after the forward pass that produced @p logits) and
         * writes the token into @p token_out in place, so the next decode step can be
         * enqueued before the host knows the token id. The host uniform is drawn here
         * at enqueue time -- one draw per sampled token, same RNG sequence as sample().
         * awaitToken() must be called before the next enqueueSample().
         */
        void enqueueSample(
            const ITensor& logits,
            TokenTensor& token_out,
            const SamplingParams& params )
        {
            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            const float r = dist( rng_ );

            op_->enqueueForward( logits, token_out, params, r );
        }

        /**
         * @brief Block until the last enqueueSample()'s token id is host-visible and return it.
         */
        int32_t awaitToken()
        {
            return op_->awaitToken();
        }

        /**
         * @brief Reseed the host RNG for reproducible sampling.
         *
         * Called once per generation run (not per token) when the caller supplies a seed.
         */
        void reseed( uint64_t seed )
        {
            rng_.seed( static_cast<std::mt19937::result_type>( seed ) );
        }

    private:
        IExecutionContext* context_;
        SamplingConfig config_;
        Tensor<TensorDataType::INT32, CpuMemoryResource> host_token_;
        std::shared_ptr<SamplingOpType> op_;
        std::mt19937 rng_;
    };
}
