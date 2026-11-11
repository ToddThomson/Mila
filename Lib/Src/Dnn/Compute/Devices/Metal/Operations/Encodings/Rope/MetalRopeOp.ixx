/**
 * @file MetalRopeOp.ixx
 * @brief Rotary position embeddings on Metal.
 */

export module Compute.MetalRopeOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.IPositionalPairedOp;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.RopeConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Rope
{
    /**
     * @brief Rotates Q and K at absolute positions, by angles calculated in the kernel.
     *
     * Follows `CudaRopeOp`'s angle and pair rules. A prefill takes its position as an argument; a decode reads it from the
     * execution context, which the network wrote at the start of the step.
     */
    export template<TensorDataType TPrecision>
    class MetalRopeOp : public Operation<DeviceType::Metal, TPrecision>, public IPositionalPairedOp
    {
    public:

        MetalRopeOp( IExecutionContext* context, const RopeConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalRopeOp" ) ),
              config_( config ),
              prefill_pipeline_( context_->device().kernels().pipeline( kernelName( "rope_prefill" ) ) ),
              decode_pipeline_( context_->device().kernels().pipeline( kernelName( "rope_decode" ) ) )
        {
            config_.validate();

            const int head_dim = narrowToKernelIndex( config_.getHeadDim() );
            const int rotary_dim = narrowToKernelIndex( config_.getRotaryDim() );
            const bool partial = rotary_dim > 0 && rotary_dim < head_dim;
            const RopeFrequencyScaling& scaling = config_.getFrequencyScaling();

            parameters_.base = config_.getBase();
            parameters_.rope_pairs = partial ? rotary_dim / 2 : head_dim / 2;
            parameters_.frequency_denominator = isRotaryPrefix() && partial ? rotary_dim : head_dim;
            parameters_.scaling_factor = scaling.factor;
            parameters_.scaling_low_frequency_factor = scaling.low_frequency_factor;
            parameters_.scaling_high_frequency_factor = scaling.high_frequency_factor;
            parameters_.scaling_original_context_length = narrowToKernelIndex( scaling.original_context_length );
            parameters_.heads = narrowToKernelIndex( config_.getNumHeads() );
            parameters_.kv_heads = narrowToKernelIndex( config_.getNumKVHeads() );
            parameters_.head_dim = head_dim;
            parameters_.pair_half = isRotaryPrefix() && partial ? rotary_dim / 2 : head_dim / 2;
        }

        /// @throws std::invalid_argument When the sequence is longer than the trained maximum.
        void build( const BuildContext& build_context ) override
        {
            const shape_t& shape = build_context.inputShape();

            if ( shape[1] > config_.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "MetalRopeOp::build: sequence length {} exceeds the trained maximum {}", shape[1], config_.getMaxSequenceLength() ) );
            }

            sequence_length_ = shape[1];

            Operation<DeviceType::Metal, TPrecision>::build( build_context );
        }

        void forward( const ITensor& q_in, const ITensor& k_in, ITensor& q_out, ITensor& k_out ) const
        {
            encodePrefill( q_in, k_in, q_out, k_out, 0 );
        }

        /// @throws std::invalid_argument For positions this op was not built for, or an out-of-place prefix layout.
        void prefill( const ITensor& q_in, const ITensor& k_in, ITensor& q_out, ITensor& k_out, dim_t position_offset ) override
        {
            const dim_t tokens = q_in.shape()[1];

            if ( position_offset < 0 || position_offset + tokens > sequence_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalRopeOp::prefill: position_offset {} + T {} exceeds the {} positions this op was built for",
                    position_offset, tokens, sequence_length_ ) );
            }

            // A prefix layout writes only the leading channels, so out of place would leave the rest unwritten.
            if ( isRotaryPrefix() && ( q_in.rawData() != q_out.rawData() || k_in.rawData() != k_out.rawData() ) )
            {
                throw std::invalid_argument( "MetalRopeOp::prefill: RotaryLayout::RotaryPrefix needs an in-place call" );
            }

            encodePrefill( q_in, k_in, q_out, k_out, narrowToKernelIndex( position_offset ) );
        }

        /// `position` is checked here and the kernel reads the same value from the context.
        void decode( const ITensor& q_in, const ITensor& k_in, ITensor& q_out, ITensor& k_out, dim_t position ) override
        {
            const dim_t tokens = q_in.shape()[1];

            if ( position < 0 || position + tokens > sequence_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalRopeOp::decode: {} tokens from position {} exceed the {} positions this op was built for",
                    tokens, position, sequence_length_ ) );
            }

            RopeParameters parameters = parameters_;
            parameters.tokens_in_chunk = narrowToKernelIndex( tokens );

            const std::array<const void*, 5> buffers{
                q_out.rawData(), q_in.rawData(), k_out.rawData(), k_in.rawData(), context_->decodePositionAddress() };

            context_->dispatch(
                decode_pipeline_, buffers, std::as_bytes( std::span( &parameters, 1 ) ), threadCount( q_in ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::RopeOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Rope";
        }

    private:

        // The kernel's RopeParameters, field for field.
        struct RopeParameters
        {
            float base{ 0.0f };
            int rope_pairs{ 0 };
            int frequency_denominator{ 0 };
            float scaling_factor{ 1.0f };
            float scaling_low_frequency_factor{ 1.0f };
            float scaling_high_frequency_factor{ 1.0f };
            int scaling_original_context_length{ 0 };
            int tokens_in_chunk{ 1 };
            int heads{ 0 };
            int kv_heads{ 0 };
            int head_dim{ 0 };
            int pair_half{ 0 };
            int position_offset{ 0 };
        };

        [[nodiscard]] bool isRotaryPrefix() const noexcept
        {
            return config_.getRotaryLayout() == RotaryLayout::RotaryPrefix;
        }

        [[nodiscard]] static std::string kernelName( std::string_view family )
        {
            return std::format( "{}_{}", family, MetalTensorDataTypeMap<TPrecision>::kernel_type );
        }

        /// One thread for each (token, pair), every batch's tokens counted. A token is a row of Q.
        [[nodiscard]] std::size_t threadCount( const ITensor& q_in ) const
        {
            const dim_t q_row = q_in.shape().back();

            return static_cast<std::size_t>( narrowToKernelIndex( q_in.size() / q_row * parameters_.pair_half ) );
        }

        void encodePrefill( const ITensor& q_in, const ITensor& k_in, ITensor& q_out, ITensor& k_out, int position_offset ) const
        {
            RopeParameters parameters = parameters_;
            parameters.tokens_in_chunk = narrowToKernelIndex( q_in.shape()[1] );
            parameters.position_offset = position_offset;

            const std::array<const void*, 4> buffers{ q_out.rawData(), q_in.rawData(), k_out.rawData(), k_in.rawData() };

            context_->dispatch(
                prefill_pipeline_, buffers, std::as_bytes( std::span( &parameters, 1 ) ), threadCount( q_in ) );
        }

        MetalExecutionContext* context_;
        RopeConfig config_;
        MetalPipeline prefill_pipeline_;
        MetalPipeline decode_pipeline_;
        RopeParameters parameters_;
        dim_t sequence_length_{ 0 };
    };
}
