/**
 * @file MetalResidualOp.ixx
 * @brief The additive residual connection on Metal.
 */

export module Compute.MetalResidualOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Components.ResidualConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Residual
{
    /// output = a + b.
    export template<TensorDataType TPrecision>
    class MetalResidualOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        // ResidualConfig::validate has already refused a scaling factor other than 1.
        MetalResidualOp( IExecutionContext* context, const ResidualConfig& )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalResidualOp" ) )
        {
        }

        /// @throws std::invalid_argument For inputs of different sizes.
        void forward( const ITensor& a, const ITensor& b, ITensor& output ) const
        {
            if ( a.size() != b.size() )
            {
                throw std::invalid_argument( "MetalResidualOp::forward - the inputs differ in size" );
            }

            const std::array<const void*, 3> buffers{ a.rawData(), b.rawData(), output.rawData() };

            context_->dispatch(
                context_->device().kernels().pipeline( std::format( "residual_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ),
                buffers, {}, static_cast<std::size_t>( narrowToKernelIndex( a.size() ) ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::ResidualOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Residual";
        }

    private:

        MetalExecutionContext* context_;
    };
}
