/**
 * @file MetalSwigluOp.ixx
 * @brief The SwiGLU activation on Metal.
 */

export module Compute.MetalSwigluOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Components.SwigluConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Swiglu
{
    /// output = silu( gate ) * up, each input row being its gate half and then its up half.
    export template<TensorDataType TPrecision>
    class MetalSwigluOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        MetalSwigluOp( IExecutionContext* context, const SwigluConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalSwigluOp" ) )
        {
            config.validate();
        }

        /// @throws std::invalid_argument When the output does not hold half of the input.
        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( input.size() % 2 != 0 || output.size() != input.size() / 2 )
            {
                throw std::invalid_argument( "MetalSwigluOp::forward - the output must hold half of the input" );
            }

            const std::array<const void*, 2> buffers{ input.rawData(), output.rawData() };
            const std::uint32_t half_width = static_cast<std::uint32_t>( narrowToKernelIndex( input.shape().back() / 2 ) );

            context_->dispatch(
                context_->device().kernels().pipeline( std::format( "swiglu_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &half_width, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( output.size() ) ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::SwigluOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Swiglu";
        }

    private:

        MetalExecutionContext* context_;
    };
}
