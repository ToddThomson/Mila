/**
 * @file SpeculativeSampler.ixx
 * @brief The drafting counterpart of TokenSampler: chooses a verified round's tokens on the device.
 *
 * See Specifications/Gemma4Mtp.md section 4.8.
 */

module;
#include <memory>
#include <span>
#include <cstdint>
#include <stdexcept>

export module Dnn.Samplers.SpeculativeSampler;

import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
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
     * @brief Chooses the tokens of a speculative round from its verified logits rows.
     *
     * A round is K drafts checked by one pass of the target, which returns K + 1 rows. The sampler draws every row
     * with its own uniform and the caller's settings -- all rows in each launch of the sampling pipeline -- then keeps
     * a draft while the row before it drew it, and the draw at the first row that differs, or at the last row, is the
     * next token. With a greedy draft that is the lossless rule of Leviathan et al., *Fast Inference from Transformers
     * via Speculative Decoding* (arXiv 2211.17192). Owned by the model beside its TokenSampler, on the network's
     * execution context; the uniforms come from the model's one sampling stream.
     *
     * @tparam TDeviceType Device the logits reside on.
     * @tparam TPrecision  Logits precision.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class SpeculativeSampler
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TokenTensor = Tensor<TensorDataType::INT32, MR>;
        using SamplingOpType = typename OperationTraits<OperationType::SamplingOp, TDeviceType, TPrecision>::type;

        /**
         * @param context The network's execution context.
         * @param config  The model's sampling configuration; its maximum rows is the deployment's K + 1.
         */
        SpeculativeSampler( IExecutionContext* context, const SamplingConfig& config )
            : context_( requireContext( context ) ),
              maximum_rows_( config.getMaximumRows() ),
              op_( std::make_shared<SamplingOpType>( context_, config ) ),
              chosen_( context_->getDeviceId(), shape_t{ config.getMaximumRows() } ),
              result_( context_->getDeviceId(), shape_t{ config.getMaximumRows() + 1 } ),
              host_result_( Device::Cpu(), shape_t{ config.getMaximumRows() + 1 } )
        {
        }

        /// The most rows a round may check, which the sampler's working stores are sized for.
        dim_t maximumRows() const noexcept
        {
            return maximum_rows_;
        }

        /**
         * @brief Enqueue a round's choice on the execution context's stream: draw every row, walk acceptance.
         *
         * Writes the round's next token into slot 0 of @p tokens, where the next round's draft reads it, and leaves
         * the kept tokens for awaitRound().
         *
         * @param logits   The verify's logits; the first `rows x vocab` elements are its rows.
         * @param tokens   The round's tokens: the known one at slot 0, then the drafts.
         * @param rows     The round's K + 1, from 1 to maximumRows().
         * @param params   The caller's sampling settings, the same for every row.
         * @param uniforms One uniform in [0, 1) per row, in row order.
         */
        void enqueueRound(
            const ITensor& logits,
            TokenTensor& tokens,
            dim_t rows,
            const SamplingParams& params,
            std::span<const float> uniforms )
        {
            op_->enqueueRowsOnDevice( logits, chosen_, rows, params, uniforms );
            op_->enqueueAcceptOnDevice( tokens, chosen_, rows, result_ );

            auto counted = result_.view( shape_t{ rows + 1 }, 0 );
            auto counted_on_host = host_result_.view( shape_t{ rows + 1 }, 0 );
            copy( counted, counted_on_host, context_ );

            pending_ = true;
        }

        /**
         * @brief Block until the last enqueueRound()'s choice is on the host, and return it.
         *
         * @return The m kept drafts followed by the next token: m + 1 tokens. Valid until the next enqueueRound().
         */
        std::span<const int32_t> awaitRound()
        {
            if ( !pending_ )
            {
                throw std::logic_error( "SpeculativeSampler::awaitRound: no enqueueRound() outstanding" );
            }

            context_->synchronize();
            pending_ = false;

            const int32_t* result = host_result_.data();
            const dim_t accepted = result[ 0 ];

            return std::span<const int32_t>( result + 1, static_cast<size_t>( accepted + 1 ) );
        }

    private:

        static IExecutionContext* requireContext( IExecutionContext* context )
        {
            if ( !context )
            {
                throw std::invalid_argument( "SpeculativeSampler: ExecutionContext cannot be null" );
            }

            return context;
        }

        IExecutionContext* context_;
        dim_t maximum_rows_;
        std::shared_ptr<SamplingOpType> op_;

        // The token drawn at each row, then the round's choice `[m, d_1 .. d_m, next]` and its host copy.
        TokenTensor chosen_;
        TokenTensor result_;
        Tensor<TensorDataType::INT32, CpuMemoryResource> host_result_;
        bool pending_{ false };
    };
}
