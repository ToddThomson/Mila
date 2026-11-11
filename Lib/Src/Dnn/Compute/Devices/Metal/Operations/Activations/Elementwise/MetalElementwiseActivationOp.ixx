/**
 * @file MetalElementwiseActivationOp.ixx
 * @brief An elementwise activation on Metal, the functor fixed at compile time by the Activation component.
 */

module;

#include "../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

export module Compute.MetalElementwiseActivationOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Components.ActivationConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Activation
{
    /// The kernel of each functor in `Kernels/ElementwiseActivation.metal`, which computes what the functor's fwd computes.
    template<typename TFunctor>
    consteval std::string_view kernelNameOf()
    {
        using namespace Mila::Dnn::Activations;

        if constexpr ( std::is_same_v<TFunctor, Identity> ) { return "identity"; }
        else if constexpr ( std::is_same_v<TFunctor, GeluTanh> ) { return "gelu_tanh"; }
        else if constexpr ( std::is_same_v<TFunctor, Silu> ) { return "silu"; }
        else if constexpr ( std::is_same_v<TFunctor, Relu> ) { return "relu"; }
        else if constexpr ( std::is_same_v<TFunctor, Tanh> ) { return "tanh"; }
        else if constexpr ( std::is_same_v<TFunctor, Sigmoid> ) { return "sigmoid"; }
        else if constexpr ( std::is_same_v<TFunctor, LeakyRelu> ) { return "leaky_relu"; }
        else if constexpr ( std::is_same_v<TFunctor, Mish> ) { return "mish"; }
        else
        {
            static_assert( sizeof( TFunctor ) == 0, "Metal has no kernel for this activation" );
        }
    }

    /**
     * @brief output = f( input ), elementwise, in 32 bits and narrowed at the store.
     *
     * @tparam TFunctor One of `Mila::Dnn::Activations`' functors. LeakyRelu's alpha comes from the config.
     */
    export template<TensorDataType TPrecision, typename TFunctor>
    class MetalElementwiseActivationOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        MetalElementwiseActivationOp( IExecutionContext* context, const ActivationConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalElementwiseActivationOp" ) ),
              pipeline_( context_->device().kernels().pipeline(
                  std::format( "activation_{}_{}", kernelNameOf<TFunctor>(), MetalTensorDataTypeMap<TPrecision>::kernel_type ) ) )
        {
            if constexpr ( std::is_same_v<TFunctor, Mila::Dnn::Activations::LeakyRelu> )
            {
                alpha_ = config.getLeakyReluAlpha();
            }
        }

        /// The input and output may be the same tensor. @throws std::invalid_argument When the output holds fewer elements.
        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( output.size() < input.size() )
            {
                throw std::invalid_argument( "MetalElementwiseActivationOp::forward - the output must hold the input's elements" );
            }

            if ( input.size() == 0 )
            {
                return;
            }

            const std::array<const void*, 2> buffers{ input.rawData(), output.rawData() };

            context_->dispatch( pipeline_, buffers, std::as_bytes( std::span( &alpha_, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( input.size() ) ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::ElementwiseActivationOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.ElementwiseActivation";
        }

    private:

        MetalExecutionContext* context_;
        MetalPipeline pipeline_;
        float alpha_{ 0.0f };
    };
}
