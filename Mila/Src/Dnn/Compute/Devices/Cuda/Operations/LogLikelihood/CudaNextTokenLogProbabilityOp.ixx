/**
 * @file CudaNextTokenLogProbabilityOp.ixx
 * @brief CUDA reduction of logits to each position's log-probability of its actual next token.
 *
 * The logits never leave the device: each window's rows are reduced where they are and only the
 * log-probabilities reach the host, written by the kernel into pinned memory the device addresses directly.
 */

module;
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include "Kernels/NextTokenLogProbability.cuh"

export module Compute.CudaNextTokenLogProbabilityOp;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::LogLikelihood
{
    using namespace Mila::Dnn;

    /**
     * @brief Each scored position's log-probability of the token that actually followed it, on the device.
     *
     * begin() sizes the host-readable store; forward() enqueues one window on the execution context's stream
     * with no host synchronization; logProbabilities() reads the store once the caller has synchronized the
     * model. Each value is computed with the row maximum subtracted and the exponentials summed in double, and
     * stored as FP32 -- a rounding of about 1e-7 relative per position, where a sequence's total is summed in
     * double by the caller. Dnn::nextTokenLogProbability is the host reference.
     *
     * @tparam TPrecision Logits precision (FP32 or BF16).
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaNextTokenLogProbabilityOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using NativeType = typename Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        /**
         * @param context             The model's execution context; windows run on its stream.
         * @param final_logit_softcap Applied as cap * tanh( logit / cap ) before the log-softmax; 0 means none.
         */
        CudaNextTokenLogProbabilityOp( IExecutionContext* context, float final_logit_softcap )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaNextTokenLogProbabilityOp" ) ),
              final_logit_softcap_( final_logit_softcap )
        {
        }

        /**
         * @brief Size the store for a sequence's scored positions. Call before its first forward(), with no
         *        window of an earlier sequence still in flight: the store may be reallocated.
         */
        void begin( dim_t positions )
        {
            if ( positions > capacity_ )
            {
                store_ = std::make_unique<StoreType>( context_->getDeviceId(), shape_t{ positions } );
                capacity_ = positions;
            }
        }

        /**
         * @brief Enqueue one window: positions [first_position, first_position + rows).
         *
         * @param logits         Device logits [.., rows, vocab]; row r holds position first_position + r.
         * @param tokens         The whole sequence's token ids on the device, INT32.
         * @param first_position Absolute position of row 0.
         * @param rows           Rows to score; the caller stops before the sequence's last position.
         */
        void forward( const ITensor& logits, const ITensor& tokens, dim_t first_position, dim_t rows ) const
        {
            if ( first_position + rows > capacity_ )
            {
                throw std::out_of_range( std::format(
                    "CudaNextTokenLogProbabilityOp::forward: positions up to {} exceed the {} begin() sized",
                    first_position + rows, capacity_ ) );
            }

            const dim_t vocab = logits.shape().back();

            cuda_next_token_log_probability<NativeType>(
                static_cast<const NativeType*>( logits.rawData() ),
                static_cast<const int32_t*>( tokens.rawData() ) + first_position,
                static_cast<float*>( store_->rawData() ) + first_position,
                static_cast<int>( rows ), static_cast<int>( vocab ), final_logit_softcap_, context_->getStream() );

            const cudaError_t status = cudaGetLastError();

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaNextTokenLogProbabilityOp::forward: launch failed: {}", cudaGetErrorString( status ) ) );
            }
        }

        /// The first `count` positions' log-probabilities. Valid once the model's stream is synchronized.
        std::span<const float> logProbabilities( dim_t count ) const
        {
            if ( !store_ )
            {
                return {};
            }

            return { static_cast<const float*>( store_->rawData() ), static_cast<std::size_t>( std::min( count, capacity_ ) ) };
        }

        OperationType getOperationType() const override
        {
            return OperationType::NextTokenLogProbabilityOp;
        }

        std::string getName() const override
        {
            return "Cuda::NextTokenLogProbabilityOp";
        }

    private:
        CudaExecutionContext* context_;
        float final_logit_softcap_;

        using StoreType = Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>;

        // Host memory, so the reduction adds nothing to the device footprint the planner prices.
        std::unique_ptr<StoreType> store_;
        dim_t capacity_{ 0 };
    };
}
