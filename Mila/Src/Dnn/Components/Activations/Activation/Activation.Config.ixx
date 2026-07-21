/**
 * @file Activation.Config.ixx
 * @brief Configuration for the unified elementwise Activation component.
 *
 * Carries the ActivationType as serializable architecture metadata plus the
 * function-specific runtime scalars (LeakyReLU alpha, GELU approximation flavor).
 * The component itself is specialized on a compile-time ActivationType TFn; this
 * config is the data side of the config-describes / type-realizes split documented
 * in Specifications/FfnAndMoE.md section 5. The model factory bridges the runtime
 * enum to the compile-time instantiation.
 */

module;
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>

export module Dnn.Components.ActivationConfig;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Dnn.ApproximationMethod;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief True for the elementwise unary activations expressible by Activation.
     *
     * Swiglu is the lone structural (gated) function in ActivationType; it lives in
     * the GatedMLP/Swiglu component family, not here. None is the identity function
     * and is a valid elementwise selection.
     */
    export constexpr bool isElementwiseActivation( ActivationType type ) noexcept
    {
        switch ( type )
        {
            case ActivationType::None:
            case ActivationType::Relu:
            case ActivationType::Gelu:
            case ActivationType::Silu:
            case ActivationType::Tanh:
            case ActivationType::Sigmoid:
            case ActivationType::LeakyRelu:
            case ActivationType::Mish:
                return true;
            default:
                return false;
        }
    }

    /**
     * @brief Configuration class for the elementwise Activation component.
     */
    export class ActivationConfig : public ComponentConfig
    {
    public:

        ActivationConfig() = default;

        explicit ActivationConfig( ActivationType activation_type )
            : activation_type_( activation_type )
        {
        }

        /**
         * @brief Set the activation function type (serializable metadata).
         */
        template <typename Self>
        Self&& withActivationType( this Self&& self, ActivationType activation_type )
        {
            self.activation_type_ = activation_type;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the LeakyReLU negative-slope coefficient. Ignored by other functions.
         */
        template <typename Self>
        Self&& withLeakyReluAlpha( this Self&& self, float alpha )
        {
            self.leaky_relu_alpha_ = alpha;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the GELU approximation flavor. Ignored by other functions.
         */
        template <typename Self>
        Self&& withGeluApproximation( this Self&& self, ApproximationMethod method )
        {
            self.gelu_approximation_ = method;
            return std::forward<Self>( self );
        }

        ActivationType getActivationType() const noexcept { return activation_type_; }
        float getLeakyReluAlpha() const noexcept { return leaky_relu_alpha_; }
        ApproximationMethod getGeluApproximation() const noexcept { return gelu_approximation_; }

        /**
         * @brief Validate configuration.
         *
         * @throws std::invalid_argument if the activation type is not elementwise
         *         (e.g. Swiglu), which belongs to the gated FFN family, not here.
         */
        void validate() const override
        {
            if ( !isElementwiseActivation( activation_type_ ) )
            {
                throw std::invalid_argument(
                    "ActivationConfig::validate: '" + activationTypeToString( activation_type_ ) +
                    "' is not an elementwise activation; gated functions belong to GatedMLP/Swiglu" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "activation_type", activationTypeToString( activation_type_ ) )
                .set( "leaky_relu_alpha", leaky_relu_alpha_ )
                .set( "gelu_approximation", std::string( ApproximationMethodToString( gelu_approximation_ ) ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto type = meta.tryGetString( "activation_type" ) )
            {
                activation_type_ = stringToActivationType( *type );
            }

            if ( auto alpha = meta.tryGetFloat( "leaky_relu_alpha" ) )
            {
                leaky_relu_alpha_ = *alpha;
            }

            if ( auto approx = meta.tryGetString( "gelu_approximation" ) )
            {
                const std::string m = *approx;

                if ( m == "Exact" )
                {
                    gelu_approximation_ = ApproximationMethod::Exact;
                }
                else if ( m == "Tanh" )
                {
                    gelu_approximation_ = ApproximationMethod::Tanh;
                }
                else if ( m == "Sigmoid" )
                {
                    gelu_approximation_ = ApproximationMethod::Sigmoid;
                }
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "ActivationConfig( ";
            oss << "activation_type=" << activationTypeToString( activation_type_ ) << ", ";
            oss << "leaky_relu_alpha=" << leaky_relu_alpha_ << ", ";
            oss << "gelu_approximation=" << static_cast<std::string_view>( ApproximationMethodToString( gelu_approximation_ ) );
            oss << " )";

            return oss.str();
        }

    private:
        ActivationType activation_type_{ ActivationType::Gelu };
        float leaky_relu_alpha_{ 0.01f };
        ApproximationMethod gelu_approximation_{ ApproximationMethod::Tanh };
    };
}
