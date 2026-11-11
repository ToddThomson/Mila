/**
 * @file MetalCausalConv1dOp.ixx
 * @brief The depthwise causal 1-D convolution on Metal, Qwen's short convolution over q, k and v.
 */

export module Compute.MetalCausalConv1dOp;

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
import Dnn.Components.CausalConv1dConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Convolution
{
    struct CausalConvolutionShape
    {
        std::uint32_t batch;
        std::uint32_t steps;
        std::uint32_t channels;
        std::uint32_t kernel_width;
        std::uint32_t dilation;
        std::uint32_t has_state;
        std::uint32_t has_bias;
    };

    /**
     * @brief out[b, t, c] = bias[c] + sum over i of weight[c, i] * x[b, t - (K - 1 - i) * D, c].
     *
     * As CUDA's: `forward` convolves a chunk against the retained rows, or against zeros where the sequence starts, and
     * `updateState` then keeps the last (K - 1) * D rows of the state and the chunk.
     */
    export template<TensorDataType TPrecision>
    class MetalCausalConv1dOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        using ConfigType = CausalConv1dConfig;

        MetalCausalConv1dOp( IExecutionContext* context, const CausalConv1dConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalCausalConv1dOp" ) ),
              config_( config ),
              forward_pipeline_( context_->device().kernels().pipeline(
                  std::format( "causal_convolution_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ) ),
              update_state_pipeline_( context_->device().kernels().pipeline(
                  std::format( "causal_convolution_update_state_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ) )
        {
            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            weight_ = weight;
            bias_ = bias;
        }

        /**
         * @param input  [B, T, C].
         * @param state  [B, (K - 1) * D, C], the rows before the chunk, or null for a sequence that starts here.
         * @param output [B, T, C].
         *
         * @throws std::runtime_error When the parameters are not set or the shapes do not match the configuration.
         */
        void forward( const ITensor& input, const ITensor* state, ITensor& output ) const
        {
            if ( !weight_ )
            {
                throw std::runtime_error( "MetalCausalConv1dOp::forward - parameters not set" );
            }

            if ( output.size() != input.size() )
            {
                throw std::runtime_error( "MetalCausalConv1dOp::forward - input/output size mismatch" );
            }

            const CausalConvolutionShape shape = shapeOf( input, state != nullptr );

            // A missing state or bias is never read: the shape says so, and the input stands in for its binding.
            const std::array<const void*, 5> buffers{
                input.rawData(), state ? state->rawData() : input.rawData(), weight_->rawData(),
                bias_ ? bias_->rawData() : input.rawData(), output.rawData() };

            context_->dispatch(
                forward_pipeline_, buffers, std::as_bytes( std::span( &shape, 1 ) ), static_cast<std::size_t>( narrowToKernelIndex( input.size() ) ) );
        }

        /// Refresh `state` to the last (K - 1) * D rows of [state ; input]. Runs after `forward` on the same chunk, whose rows it replaces.
        void updateState( const ITensor& input, ITensor& state ) const
        {
            const CausalConvolutionShape shape = shapeOf( input, true );
            const std::array<const void*, 2> buffers{ state.rawData(), input.rawData() };

            context_->dispatch(
                update_state_pipeline_, buffers, std::as_bytes( std::span( &shape, 1 ) ),
                static_cast<std::size_t>( shape.batch ) * static_cast<std::size_t>( shape.channels ) );
        }

        void build( const BuildContext& /*context*/ ) override
        {
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::CausalConv1dOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.CausalConv1d";
        }

    private:

        [[nodiscard]] CausalConvolutionShape shapeOf( const ITensor& input, bool has_state ) const
        {
            const auto& shape = input.shape();

            if ( shape.size() != 3 )
            {
                throw std::runtime_error( "MetalCausalConv1dOp - input must be rank 3 [B, T, C]" );
            }

            if ( shape[2] != config_.getChannels() )
            {
                throw std::runtime_error( "MetalCausalConv1dOp - channel count mismatch" );
            }

            return CausalConvolutionShape{
                static_cast<std::uint32_t>( narrowToKernelIndex( shape[0] ) ), static_cast<std::uint32_t>( narrowToKernelIndex( shape[1] ) ),
                static_cast<std::uint32_t>( narrowToKernelIndex( shape[2] ) ), static_cast<std::uint32_t>( config_.getKernelWidth() ),
                static_cast<std::uint32_t>( config_.getDilation() ), has_state ? 1u : 0u, bias_ ? 1u : 0u };
        }

        MetalExecutionContext* context_;
        CausalConv1dConfig config_;
        MetalPipeline forward_pipeline_;
        MetalPipeline update_state_pipeline_;
        ITensor* weight_{ nullptr };
        ITensor* bias_{ nullptr };
    };
}
