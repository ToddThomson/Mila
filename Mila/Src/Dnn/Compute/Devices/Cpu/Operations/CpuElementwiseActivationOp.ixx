/**
 * @file CpuElementwiseActivationOp.ixx
 * @brief CPU implementation of the functor-templated elementwise activation op.
 *
 * One implementation yields every elementwise activation (GELU, SiLU, ReLU, Tanh,
 * Sigmoid, LeakyReLU, Mish, identity) on CPU by templating on the functor from the
 * shared ElementwiseActivation library. This closes the "CPU only has GELU" gap for
 * the elementwise family (see Specifications/FfnAndMoE.md section 5.1).
 */

module;
#include <string>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuElementwiseActivationOp;

import Dnn.Components.ActivationConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU elementwise activation operation, specialized on a compile-time functor.
     *
     * @tparam TFunctor POD functor from Mila::Dnn::Activations exposing fwd(x)/df(x).
     *
     * FP32 only — FP32 is the sole CPU-supported precision. The functor is selected
     * at compile time by the Activation component; this op holds an instance so
     * function-specific scalars (LeakyReLU alpha) are carried by value.
     */
    export template<typename TFunctor>
    class CpuElementwiseActivationOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using TensorType = Tensor<TensorDataType::FP32, MR>;

        CpuElementwiseActivationOp( IExecutionContext* context, const ActivationConfig& config )
            : context_( context )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuElementwiseActivationOp requires a CPU execution context" );
            }

            // LeakyReLU carries a runtime negative-slope scalar; inject it where present.
            if constexpr ( requires( TFunctor f ) { f.alpha; } )
            {
                functor_.alpha = config.getLeakyReluAlpha();
            }
        }

        void build( const BuildContext& ) override
        {
            this->is_built_ = true;
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( !this->is_built_ )
            {
                throw std::runtime_error( "CpuElementwiseActivationOp not built - call build() first" );
            }

            const float* input_data = static_cast<const float*>( input.rawData() );
            float* output_data = static_cast<float*>( output.rawData() );

            if ( !input_data || !output_data )
            {
                throw std::runtime_error( "CpuElementwiseActivationOp::forward - null tensor data pointer" );
            }

            const size_t N = input.size();

#pragma omp parallel for if(N > 1000)
            for ( int i = 0; i < static_cast<int>( N ); i++ )
            {
                output_data[ i ] = functor_.fwd( input_data[ i ] );
            }
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) const
        {
            const float* inp = static_cast<const float*>( input.rawData() );
            const float* dout = static_cast<const float*>( output_grad.rawData() );
            float* dinp = static_cast<float*>( input_grad.rawData() );

            if ( !inp || !dout || !dinp )
            {
                throw std::runtime_error( "CpuElementwiseActivationOp::backward - null tensor data pointer" );
            }

            const size_t N = input.size();

            // Accumulate into a pre-zeroed buffer (the component zeroes input_grad
            // before each backward), matching the CpuGeluOp accumulation contract.
#pragma omp parallel for if(N > 1000)
            for ( int i = 0; i < static_cast<int>( N ); i++ )
            {
                dinp[ i ] += functor_.df( inp[ i ] ) * dout[ i ];
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::ElementwiseActivationOp;
        }

        std::string getName() const override
        {
            return "Cpu::ElementwiseActivationOp";
        }

    private:
        IExecutionContext* context_{ nullptr };
        TFunctor functor_{};
    };
}
