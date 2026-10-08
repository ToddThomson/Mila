/**
 * @file CpuCausalConv1dOp.ixx
 * @brief CPU implementation of the depthwise causal 1-D convolution (FP32).
 *
 * The reference loop behind CausalConv1d<Cpu>, and the one the CUDA kernels are checked against: the same
 * two entry points, the same state layout, the same tap order.
 */

module;
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

export module Compute.CpuCausalConv1dOp;

import Dnn.Components.CausalConv1dConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Depthwise causal 1-D convolution over [B, T, C], one length-K filter per channel at dilation D.
     *
     *   out[b, t, c] = bias[c] + sum(i = 0 .. K-1) weight[c, i] * x[b, t - (K-1-i) * D, c]
     *
     * Positions left of the chunk come from a state of R = (K-1) * D rows, [B, R, C], row j at relative
     * position -R + j; a null state means the sequence starts here and they are zero. Taps accumulate in
     * order i = 0 .. K-1 in float, as the CUDA kernels do.
     */
    export class CpuCausalConv1dOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ConfigType = CausalConv1dConfig;

        CpuCausalConv1dOp( IExecutionContext* context, const CausalConv1dConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuCausalConv1dOp requires a CPU execution context" );
            }

            config_.validate();

            channels_ = static_cast<int64_t>( config_.getChannels() );
            kernel_width_ = static_cast<int64_t>( config_.getKernelWidth() );
            dilation_ = static_cast<int64_t>( config_.getDilation() );
            state_rows_ = static_cast<int64_t>( config_.getStateRows() );
        }

        void setParameters( ITensor* weight, ITensor* bias )
        {
            weight_ = weight ? static_cast<const float*>( weight->rawData() ) : nullptr;
            bias_ = bias ? static_cast<const float*>( bias->rawData() ) : nullptr;
        }

        void forward( const ITensor& input, const ITensor* state, ITensor& output ) const
        {
            if ( !weight_ )
            {
                throw std::runtime_error( "CpuCausalConv1dOp::forward - parameters not set" );
            }

            const Extent extent = extentOf( input, "forward" );

            if ( output.size() != input.size() )
            {
                throw std::runtime_error( "CpuCausalConv1dOp::forward - input/output size mismatch" );
            }

            const auto* x = static_cast<const float*>( input.rawData() );
            const auto* s = state ? static_cast<const float*>( state->rawData() ) : nullptr;
            auto* y = static_cast<float*>( output.rawData() );

            for ( int64_t b = 0; b < extent.batch; ++b )
            {
                for ( int64_t t = 0; t < extent.length; ++t )
                {
                    for ( int64_t c = 0; c < channels_; ++c )
                    {
                        float accumulator = bias_ ? bias_[ c ] : 0.0f;

                        for ( int64_t i = 0; i < kernel_width_; ++i )
                        {
                            const int64_t source_t = t - state_rows_ + i * dilation_;
                            float value = 0.0f;

                            if ( source_t >= 0 )
                            {
                                value = x[ ( b * extent.length + source_t ) * channels_ + c ];
                            }
                            else if ( s )
                            {
                                value = s[ ( b * state_rows_ + ( state_rows_ + source_t ) ) * channels_ + c ];
                            }

                            accumulator += weight_[ c * kernel_width_ + i ] * value;
                        }

                        y[ ( b * extent.length + t ) * channels_ + c ] = accumulator;
                    }
                }
            }
        }

        /**
         * @brief Refresh @p state to the last (K-1) * D rows of [state ; input].
         *
         * Must run after forward() for the same chunk. A chunk shorter than the window keeps the old
         * state's still-relevant tail, so one routine serves prefill and decode.
         */
        void updateState( const ITensor& input, ITensor& state ) const
        {
            const Extent extent = extentOf( input, "updateState" );

            const auto* x = static_cast<const float*>( input.rawData() );
            auto* s = static_cast<float*>( state.rawData() );

            std::vector<float> staged( static_cast<size_t>( state_rows_ ) );

            for ( int64_t b = 0; b < extent.batch; ++b )
            {
                for ( int64_t c = 0; c < channels_; ++c )
                {
                    for ( int64_t j = 0; j < state_rows_; ++j )
                    {
                        const int64_t source_t = extent.length - state_rows_ + j;

                        staged[ static_cast<size_t>( j ) ] = source_t >= 0
                            ? x[ ( b * extent.length + source_t ) * channels_ + c ]
                            : s[ ( b * state_rows_ + ( state_rows_ + source_t ) ) * channels_ + c ];
                    }

                    for ( int64_t j = 0; j < state_rows_; ++j )
                    {
                        s[ ( b * state_rows_ + j ) * channels_ + c ] = staged[ static_cast<size_t>( j ) ];
                    }
                }
            }
        }

        void build( const BuildContext& build_context ) override
        {
            Operation<DeviceType::Cpu, TensorDataType::FP32>::build( build_context );
        }

        OperationType getOperationType() const override
        {
            return OperationType::CausalConv1dOp;
        }

        std::string getName() const override
        {
            return "Cpu::CausalConv1dOp";
        }

    private:
        struct Extent
        {
            int64_t batch{ 0 };
            int64_t length{ 0 };
        };

        Extent extentOf( const ITensor& input, const char* caller ) const
        {
            const auto& shape = input.shape();

            if ( shape.size() != 3 )
            {
                throw std::runtime_error( std::string( "CpuCausalConv1dOp::" ) + caller + " - input must be rank 3 [B, T, C]" );
            }

            if ( static_cast<int64_t>( shape[ 2 ] ) != channels_ )
            {
                throw std::runtime_error( std::string( "CpuCausalConv1dOp::" ) + caller + " - channel count mismatch" );
            }

            return Extent{ static_cast<int64_t>( shape[ 0 ] ), static_cast<int64_t>( shape[ 1 ] ) };
        }

        IExecutionContext* context_;
        CausalConv1dConfig config_;

        const float* weight_{ nullptr };
        const float* bias_{ nullptr };

        int64_t channels_{ 0 };
        int64_t kernel_width_{ 0 };
        int64_t dilation_{ 1 };
        int64_t state_rows_{ 0 };
    };
}
