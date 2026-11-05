/**
 * @file MetalLinearOp.ixx
 * @brief A Linear's matrix multiply on Metal, unquantized.
 */

export module Compute.MetalLinearOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMatrixProduct;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.LinearConfig;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Linear
{
    /**
     * @brief output = input * weight^T + bias.
     *
     * @tparam TComputePrecision FP16 or BF16.
     */
    export template<TensorDataType TComputePrecision>
    class MetalLinearOp : public Operation<DeviceType::Metal, TComputePrecision>
    {
    public:

        using TensorType = Tensor<TComputePrecision, MetalMemoryResource>;

        MetalLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalLinearOp" ) ),
              config_( config )
        {
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            weight_ = weight;
            bias_ = bias;
        }

        void build( const BuildContext& build_context ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "MetalLinearOp::build - setParameters() must be called before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "MetalLinearOp::build - bias expected by config but not bound" );
            }

            const shape_t& input_shape = build_context.inputShape();

            if ( input_shape.empty() || input_shape.back() != config_.getInputFeatures() )
            {
                throw std::invalid_argument( "MetalLinearOp::build - the input's last axis must be the configured input features" );
            }

            inputs_ = narrowToKernelIndex( config_.getInputFeatures() );
            outputs_ = narrowToKernelIndex( config_.getOutputFeatures() );

            Operation<DeviceType::Metal, TComputePrecision>::build( build_context );
        }

        /// Encode the multiply. The rows are however many the input holds this call, and `output` holds at least that many.
        void forward( const TensorType& input, TensorType& output ) const
        {
            const int rows = narrowToKernelIndex( input.size() / config_.getInputFeatures() );
            const void* const output_address = static_cast<const ITensor&>( output ).rawData();

            encodeMatrixProduct<TComputePrecision>( *context_, MatrixProduct{
                static_cast<const ITensor&>( input ).rawData(), static_cast<const ITensor*>( weight_ )->rawData(), output_address,
                rows, inputs_, outputs_, outputs_ } );

            if ( config_.hasBias() )
            {
                encodeBias( output_address, rows );
            }
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Linear";
        }

    private:

        void encodeBias( const void* output, int rows ) const
        {
            const std::array<const void*, 2> buffers{ static_cast<const ITensor*>( bias_ )->rawData(), output };
            const std::uint32_t outputs = static_cast<std::uint32_t>( outputs_ );

            context_->dispatch(
                context_->device().kernels().pipeline( std::format( "linear_add_bias_{}", MetalTensorDataTypeMap<TComputePrecision>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &outputs, 1 ) ),
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( outputs_ ) );
        }

        MetalExecutionContext* context_;
        LinearConfig config_;
        ITensor* weight_{ nullptr };
        ITensor* bias_{ nullptr };
        int inputs_{ 0 };
        int outputs_{ 0 };
    };
}
