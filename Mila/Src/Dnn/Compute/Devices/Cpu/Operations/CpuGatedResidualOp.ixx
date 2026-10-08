/**
 * @file CpuGatedResidualOp.ixx
 * @brief CPU implementation of the gated residual's elementwise steps (FP32).
 *
 * The reference loops behind GatedResidual<Cpu>; the projections and the grouped norm are the component's
 * children, so this holds only what lies between them.
 */

module;
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

export module Compute.CpuGatedResidualOp;

import Dnn.Components.GatedResidualConfig;
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
     * @brief The four elementwise steps of a gated residual over n streams of width H.
     *
     *   activateDown:      a = silu( down / n )                              [rows, rank]
     *   mix:               x[h] = mean_i( sigmoid( up[i, h] ) * N[i, h] )    [rows, H]
     *   injectionWeights:  g[i] = 2 * sigmoid( inject[i] / n )               [rows, n]
     *   inject:            S'[i, h] = S[i, h] + g[i] * y[h]                  [rows, n * H]
     *
     * N is the grouped-normalized stream and S the stream as it entered; the update adds to S, not to N.
     */
    export class CpuGatedResidualOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ConfigType = GatedResidualConfig;

        CpuGatedResidualOp( IExecutionContext* context, const GatedResidualConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuGatedResidualOp requires a CPU execution context" );
            }

            config_.validate();

            streams_ = static_cast<int64_t>( config_.getStreams() );
            model_dim_ = static_cast<int64_t>( config_.getModelDim() );
        }

        void activateDown( const ITensor& down, ITensor& output ) const
        {
            const auto* z = static_cast<const float*>( down.rawData() );
            auto* a = static_cast<float*>( output.rawData() );
            const float inverse_streams = 1.0f / static_cast<float>( streams_ );
            const int64_t count = static_cast<int64_t>( down.size() );

            for ( int64_t i = 0; i < count; ++i )
            {
                const float scaled = z[ i ] * inverse_streams;
                a[ i ] = scaled / ( 1.0f + std::exp( -scaled ) );
            }
        }

        void mix( const ITensor& normed, const ITensor& up, ITensor& mixed ) const
        {
            const int64_t rows = rowsOf( normed, streams_ * model_dim_, "mix" );
            const auto* n = static_cast<const float*>( normed.rawData() );
            const auto* u = static_cast<const float*>( up.rawData() );
            auto* x = static_cast<float*>( mixed.rawData() );
            const float inverse_streams = 1.0f / static_cast<float>( streams_ );

            for ( int64_t row = 0; row < rows; ++row )
            {
                const int64_t base = row * streams_ * model_dim_;

                for ( int64_t h = 0; h < model_dim_; ++h )
                {
                    float sum = 0.0f;

                    for ( int64_t stream = 0; stream < streams_; ++stream )
                    {
                        const int64_t at = base + stream * model_dim_ + h;
                        sum += n[ at ] / ( 1.0f + std::exp( -u[ at ] ) );
                    }

                    x[ row * model_dim_ + h ] = sum * inverse_streams;
                }
            }
        }

        void injectionWeights( const ITensor& inject, ITensor& weights ) const
        {
            const auto* z = static_cast<const float*>( inject.rawData() );
            auto* g = static_cast<float*>( weights.rawData() );
            const float inverse_streams = 1.0f / static_cast<float>( streams_ );
            const int64_t count = static_cast<int64_t>( inject.size() );

            for ( int64_t i = 0; i < count; ++i )
            {
                g[ i ] = 2.0f / ( 1.0f + std::exp( -z[ i ] * inverse_streams ) );
            }
        }

        void inject( const ITensor& stream, const ITensor& weights, const ITensor& sublayer_output, ITensor& output ) const
        {
            const int64_t rows = rowsOf( stream, streams_ * model_dim_, "inject" );
            const auto* s = static_cast<const float*>( stream.rawData() );
            const auto* g = static_cast<const float*>( weights.rawData() );
            const auto* y = static_cast<const float*>( sublayer_output.rawData() );
            auto* out = static_cast<float*>( output.rawData() );

            if ( static_cast<int64_t>( sublayer_output.size() ) != rows * model_dim_ )
            {
                throw std::runtime_error( "CpuGatedResidualOp::inject - the sublayer output is not [rows, model_dim]" );
            }

            for ( int64_t row = 0; row < rows; ++row )
            {
                for ( int64_t stream_index = 0; stream_index < streams_; ++stream_index )
                {
                    const float weight = g[ row * streams_ + stream_index ];
                    const int64_t base = ( row * streams_ + stream_index ) * model_dim_;

                    for ( int64_t h = 0; h < model_dim_; ++h )
                    {
                        out[ base + h ] = s[ base + h ] + y[ row * model_dim_ + h ] * weight;
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
            return OperationType::GatedResidualOp;
        }

        std::string getName() const override
        {
            return "Cpu::GatedResidualOp";
        }

    private:
        int64_t rowsOf( const ITensor& tensor, int64_t width, const char* caller ) const
        {
            const int64_t size = static_cast<int64_t>( tensor.size() );

            if ( size % width != 0 )
            {
                throw std::runtime_error( std::string( "CpuGatedResidualOp::" ) + caller + " - input is not a whole number of rows" );
            }

            return size / width;
        }

        IExecutionContext* context_;
        GatedResidualConfig config_;

        int64_t streams_{ 0 };
        int64_t model_dim_{ 0 };
    };
}
