/**
 * @file CudaGatedResidualOp.ixx
 * @brief CUDA implementation of the gated residual's elementwise steps, FP32 and BF16.
 *
 * The projections and the grouped norm are GatedResidual's children; this holds only what lies between them.
 */

module;
#include "Kernels/GatedResidual.cuh"
#include <cuda_bf16.h>

export module Compute.CudaGatedResidualOp;

import std;
import Dnn.Components.GatedResidualConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Residual
{
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
    export template<TensorDataType TPrecision>
    class CudaGatedResidualOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GatedResidualConfig;

        CudaGatedResidualOp( IExecutionContext* context, const GatedResidualConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGatedResidualOp" ) ), config_( config )
        {
            config_.validate();

            streams_ = config_.getStreams();
            model_dim_ = config_.getModelDim();
        }

        void activateDown( const ITensor& down, ITensor& output ) const
        {
            cuda_gated_residual_activate_down( data( output ), data( down ),
                narrowToKernelIndex( down.size() ), narrowToKernelIndex( streams_ ), context_->getStream() );
        }

        void mix( const ITensor& normed, const ITensor& up, ITensor& mixed ) const
        {
            const dim_t rows = rowsOf( normed, streams_ * model_dim_, "mix" );

            cuda_gated_residual_mix( data( mixed ), data( normed ), data( up ),
                narrowToKernelIndex( rows ), narrowToKernelIndex( streams_ ), narrowToKernelIndex( model_dim_ ),
                context_->getStream() );
        }

        void injectionWeights( const ITensor& inject, ITensor& weights ) const
        {
            cuda_gated_residual_injection_weights( data( weights ), data( inject ),
                narrowToKernelIndex( inject.size() ), narrowToKernelIndex( streams_ ), context_->getStream() );
        }

        void inject( const ITensor& stream, const ITensor& weights, const ITensor& sublayer_output, ITensor& output ) const
        {
            const dim_t rows = rowsOf( stream, streams_ * model_dim_, "inject" );

            if ( sublayer_output.size() != rows * model_dim_ )
            {
                throw std::runtime_error( "CudaGatedResidualOp::inject - the sublayer output is not [rows, model_dim]" );
            }

            cuda_gated_residual_inject( data( output ), data( stream ), data( weights ), data( sublayer_output ),
                narrowToKernelIndex( rows ), narrowToKernelIndex( streams_ ), narrowToKernelIndex( model_dim_ ),
                context_->getStream() );
        }

        void build( const BuildContext& build_context ) override
        {
            Operation<DeviceType::Cuda, TPrecision>::build( build_context );
        }

        OperationType getOperationType() const override
        {
            return OperationType::GatedResidualOp;
        }

        std::string getName() const override
        {
            return "Cuda::GatedResidualOp";
        }

    private:
        static NativeType* data( ITensor& tensor )
        {
            return static_cast<NativeType*>( tensor.rawData() );
        }

        static const NativeType* data( const ITensor& tensor )
        {
            return static_cast<const NativeType*>( tensor.rawData() );
        }

        static dim_t rowsOf( const ITensor& tensor, dim_t width, const char* caller )
        {
            if ( tensor.size() % width != 0 )
            {
                throw std::runtime_error( std::string( "CudaGatedResidualOp::" ) + caller + " - input is not a whole number of rows" );
            }

            return tensor.size() / width;
        }

        CudaExecutionContext* context_{ nullptr };
        GatedResidualConfig config_;

        dim_t streams_{ 0 };
        dim_t model_dim_{ 0 };
    };
}
