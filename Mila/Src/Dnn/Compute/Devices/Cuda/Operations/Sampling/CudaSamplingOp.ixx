/**
 * @file CudaSamplingOp.ixx
 * @brief CUDA device-side token sampling operation.
 *
 * Phase A: greedy (argmax) sampling on the device. The logits never leave the
 * device; the op writes the chosen int32 token into the caller's device buffer.
 * Stochastic top-k/top-p land in later phases. See Specifications/TokenSampling.md.
 */

module;
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cuda_bf16.h>
#include "Kernels/Sampling.cuh"

export module Compute.CudaSamplingOp;

import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Sampling
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA token sampler op: maps a logits row to a single int32 token id.
     *
     * Reads the final `vocab_size` elements of @p logits (the last decode position)
     * and, for the greedy branch, runs a block-wide argmax, writing the result into
     * @p token_out on the execution context's stream. Internal compute is FP32.
     *
     * @tparam TPrecision Logits precision (FP32 or BF16).
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaSamplingOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using NativeType = typename Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSamplingOp( IExecutionContext* context, const SamplingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSamplingOp" ) ),
              config_( config ),
              prob_scratch_( context->getDeviceId(), shape_t{ config.getVocabularySize() } )
        {
            config_.validate();
        }

        /**
         * @brief Sample one token from the final logits row.
         *
         * @param logits    Device logits; the last `vocab_size` elements are the active row.
         * @param token_out Device INT32 [1,1] tensor; written in place on the stream.
         * @param params    Per-call sampling parameters.
         * @param r         Host-drawn uniform in [0,1) (unused by the greedy branch).
         */
        void forward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            const int64_t vocab = config_.getVocabularySize();
            const int64_t offset = static_cast<int64_t>( logits.size() ) - vocab;

            const NativeType* row = static_cast<const NativeType*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

            // Phase A/B run on the default stream (0). The model synchronizes the network
            // before sampling, so the logits are complete, and the TokenSampler reads the
            // token back synchronously, so default-stream ordering is correct. Sharing the
            // decode stream (context_->getStream()) is a Phase D optimization.
            cudaStream_t stream = 0;

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                cuda_sample_argmax<NativeType>( row, out, static_cast<int>( vocab ), stream );
                return;
            }

            float* scratch = static_cast<float*>( prob_scratch_.rawData() );

            cuda_sample_stochastic<NativeType>(
                row, out, scratch, static_cast<int>( vocab ),
                config_.getFinalLogitSoftcap(), params.temperature,
                params.top_k, params.top_p, r, stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::SamplingOp;
        }

        std::string getName() const override
        {
            return "Cuda::SamplingOp";
        }

    private:

        CudaExecutionContext* context_;
        SamplingConfig config_;
        // Working store for the stochastic softmax/inverse-CDF (vocab FP32). Mutable so
        // the const forward() can write through it (the op holds no logical state).
        mutable Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> prob_scratch_;
    };
}
