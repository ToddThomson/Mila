/**
 * @file MLP.Dispatch.ixx
 * @brief Activation dispatch helpers for MLP.
 */

module;
#include <memory>
#include <functional>
#include <stdexcept>
#include <string>
#include <optional>

export module Dnn.Components.MLP:Dispatch;

import Dnn.ActivationType;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.Precision;
import Dnn.Components.Gelu;
import Dnn.Components.Swiglu;
import Dnn.Component;

namespace Mila::Dnn::Detail
{
    using namespace Mila::Dnn::Compute;

    template<ActivationType TActivation, DeviceType TDeviceType, TensorDataType TPrecision>
    struct mlp_activation_impl;

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    struct mlp_activation_impl<ActivationType::Gelu, TDeviceType, TPrecision>
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ActivationComponentType = Gelu<TDeviceType, TPrecision>;
        using ActivationComponentBase = Component<TDeviceType, TPrecision>;
        using ForwardFn = std::function<TensorType& (const TensorType&)>;
        using BackwardFn = std::function<TensorType& (const TensorType&, const TensorType&)>;

        static std::shared_ptr<ActivationComponentType> create( const std::string& name )
        {
            auto cfg = GeluConfig();

            return std::make_shared<ActivationComponentType>( name, cfg, std::nullopt );
        }

        static void bind(
            const std::shared_ptr<ActivationComponentType>& activation,
            std::shared_ptr<ActivationComponentBase>& out_base,
            ForwardFn& out_forward,
            BackwardFn& out_backward )
        {
            if ( !activation )
            {
                throw std::invalid_argument( "mlp_activation_impl<Gelu>::bind: activation is null" );
            }

            out_base = activation;

            out_forward = [activation]( const TensorType& input ) -> TensorType&
                {
                    return activation->forward( input );
                };

            out_backward = [activation]( const TensorType& input, const TensorType& output_grad ) -> TensorType&
                {
                    return activation->backward( input, output_grad );
                };
        }
    };

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    struct mlp_activation_impl<ActivationType::Swiglu, TDeviceType, TPrecision>
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ActivationComponentType = Swiglu<TDeviceType, TPrecision>;
        using ActivationComponentBase = Component<TDeviceType, TPrecision>;
        using ForwardFn = std::function<TensorType& (const TensorType&)>;
        using BackwardFn = std::function<TensorType& (const TensorType&, const TensorType&)>;

        static std::shared_ptr<ActivationComponentType> create( const std::string& name )
        {
            auto cfg = SwigluConfig();

            return std::make_shared<ActivationComponentType>( name, cfg, std::nullopt );
        }

        static void bind(
            const std::shared_ptr<ActivationComponentType>& activation,
            std::shared_ptr<ActivationComponentBase>& out_base,
            ForwardFn& out_forward,
            BackwardFn& out_backward )
        {
            if ( !activation )
            {
                throw std::invalid_argument( "mlp_activation_impl<Swiglu>::bind: activation is null" );
            }

            out_base = activation;

            out_forward = [activation]( const TensorType& input ) -> TensorType&
                {
                    return activation->forward( input );
                };

            out_backward = [activation]( const TensorType& input, const TensorType& output_grad ) -> TensorType&
                {
                    return activation->backward( input, output_grad );
                };
        }
    };
}