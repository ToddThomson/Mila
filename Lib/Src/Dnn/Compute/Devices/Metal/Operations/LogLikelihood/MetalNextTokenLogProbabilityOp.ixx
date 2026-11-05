/**
 * @file MetalNextTokenLogProbabilityOp.ixx
 * @brief Each scored position's log-probability of the token that followed it, on Metal.
 */

export module Compute.MetalNextTokenLogProbabilityOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::LogLikelihood
{
    /**
     * @brief Encodes one window at a time, and the caller reads the store once the model is synchronized.
     *
     * `Dnn::nextTokenLogProbability` is the host reference. Its exponentials sum in double and these in 32 bits, since Metal has no double.
     */
    export template<TensorDataType TPrecision>
    class MetalNextTokenLogProbabilityOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        /// @param final_logit_softcap Applied as cap * tanh( logit / cap ) before the log-softmax; 0 means none.
        MetalNextTokenLogProbabilityOp( IExecutionContext* context, float final_logit_softcap )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalNextTokenLogProbabilityOp" ) ),
              final_logit_softcap_( final_logit_softcap )
        {
        }

        /// Size the store for a sequence's scored positions, with no window of an earlier sequence still in flight.
        void begin( dim_t positions )
        {
            if ( positions > capacity_ )
            {
                store_ = std::make_unique<StoreType>( context_->getDeviceId(), shape_t{ positions } );
                capacity_ = positions;
            }
        }

        /**
         * @brief Encode positions [first_position, first_position + rows).
         *
         * @param logits Metal logits [.., rows, vocab]; row r holds position first_position + r.
         * @param tokens The whole sequence's token ids, INT32.
         * @throws std::out_of_range Past what `begin` sized.
         */
        void forward( const ITensor& logits, const ITensor& tokens, dim_t first_position, dim_t rows ) const
        {
            if ( first_position + rows > capacity_ )
            {
                throw std::out_of_range( std::format(
                    "MetalNextTokenLogProbabilityOp::forward: positions up to {} exceed the {} begin() sized", first_position + rows, capacity_ ) );
            }

            const std::array<const void*, 3> buffers{
                logits.rawData(),
                static_cast<const std::int32_t*>( tokens.rawData() ) + first_position,
                static_cast<const float*>( static_cast<const ITensor*>( store_.get() )->rawData() ) + first_position };

            const Parameters parameters{ narrowToKernelIndex( logits.shape().back() ), final_logit_softcap_ };

            context_->dispatchThreadgroups(
                context_->device().kernels().pipeline( std::format( "next_token_log_probability_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &parameters, 1 ) ), static_cast<std::size_t>( narrowToKernelIndex( rows ) ), 1 );
        }

        /// The first `count` positions' log-probabilities. Valid once the model is synchronized.
        [[nodiscard]] std::span<const float> logProbabilities( dim_t count ) const
        {
            if ( !store_ )
            {
                return {};
            }

            return { static_cast<const float*>( static_cast<const ITensor*>( store_.get() )->rawData() ),
                static_cast<std::size_t>( std::min( count, capacity_ ) ) };
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::NextTokenLogProbabilityOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.NextTokenLogProbability";
        }

    private:

        using StoreType = Tensor<TensorDataType::FP32, MetalMemoryResource>;

        // The kernel's NextTokenParameters, field for field.
        struct Parameters
        {
            std::int32_t vocab;
            float softcap;
        };

        MetalExecutionContext* context_;
        float final_logit_softcap_;
        std::unique_ptr<StoreType> store_;
        dim_t capacity_{ 0 };
    };
}
