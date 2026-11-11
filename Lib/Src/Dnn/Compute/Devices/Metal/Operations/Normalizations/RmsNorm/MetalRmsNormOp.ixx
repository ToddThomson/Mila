/**
 * @file MetalRmsNormOp.ixx
 * @brief RMS normalization on Metal, over the innermost axis.
 */

export module Compute.MetalRmsNormOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.RmsNormConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::RmsNorm
{
    /**
     * @brief output = input * rstd * ( weight + unit offset ) + bias, with rstd taken over each row or each group of it.
     *
     * Only the innermost axis: every model Mila runs normalizes that one, and a norm over another axis is refused at build.
     */
    export template<TensorDataType TPrecision>
    class MetalRmsNormOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        MetalRmsNormOp( IExecutionContext* context, const RmsNormConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalRmsNormOp" ) ),
              config_( config ),
              pipeline_( context_->device().kernels().pipeline( std::format( "rms_norm_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ) )
        {
            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            weight_ = weight;
            bias_ = bias;
        }

        /// @throws std::invalid_argument For a norm over any axis but the innermost, or a group size that does not divide it.
        void build( const BuildContext& build_context ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "MetalRmsNormOp::build - setParameters() must be called before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "MetalRmsNormOp::build - bias expected by config but not bound" );
            }

            const shape_t& shape = build_context.inputShape();
            const bool innermost = config_.getAxis().has_value()
                ? config_.getAxis().value() == -1 || config_.getAxis().value() == static_cast<std::int64_t>( shape.size() ) - 1
                : config_.getNormalizedShape().size() == 1;

            if ( !innermost )
            {
                throw std::invalid_argument( "MetalRmsNormOp::build - only the innermost axis can be normalized on Metal" );
            }

            const dim_t width = shape.back();
            const dim_t group_size = config_.getGroupSize() > 0 ? static_cast<dim_t>( config_.getGroupSize() ) : width;

            if ( width % group_size != 0 )
            {
                throw std::invalid_argument( "MetalRmsNormOp::build - the group size does not divide the normalized extent" );
            }

            width_ = width;
            groups_ = narrowToKernelIndex( width / group_size );

            Operation<DeviceType::Metal, TPrecision>::build( build_context );
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            const RmsNormShape shape{
                static_cast<std::uint32_t>( narrowToKernelIndex( width_ / groups_ ) ),
                static_cast<std::uint32_t>( groups_ ),
                bias_ != nullptr ? 1u : 0u,
                config_.getEpsilon(),
                config_.getUnitOffset() };

            const std::array<const void*, 4> buffers{
                input.rawData(), weight_->rawData(), bias_ != nullptr ? bias_->rawData() : weight_->rawData(), output.rawData() };

            context_->dispatchThreadgroups(
                pipeline_, buffers, std::as_bytes( std::span( &shape, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( input.size() / width_ * groups_ ) ), 1, kSimdgroups );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::RmsNormOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.RmsNorm";
        }

    private:

        // The kernel's RmsNormShape, field for field.
        struct RmsNormShape
        {
            std::uint32_t elements;
            std::uint32_t groups;
            std::uint32_t has_bias;
            float epsilon;
            float unit_offset;
        };

        // RmsNorm.metal's threadgroup: eight SIMD-groups to a slice.
        static constexpr std::size_t kSimdgroups = 8;

        MetalExecutionContext* context_;
        RmsNormConfig config_;
        MetalPipeline pipeline_;
        ITensor* weight_{ nullptr };
        ITensor* bias_{ nullptr };
        dim_t width_{ 0 };
        int groups_{ 1 };
    };
}
