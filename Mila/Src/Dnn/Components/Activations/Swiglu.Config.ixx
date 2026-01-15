/**
 * @file Swiglu.Config.ixx
 * @brief Configuration for the SwiGLU activation module.
 *
 * Provides fluent setters and serialization/validation for the SwiGLU activation.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>

export module Dnn.Components.Swiglu:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ApproximationMethod;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for the SwiGLU activation component.
     *
     * SwiGLU applies a gated activation: out = x1 * GELU(x2) where the input is split
     * along the channel dimension into two equal parts. The config controls the inner
     * GELU approximation method used by the gate.
     */
    export class SwigluConfig : public ComponentConfig
    {
    public:
        SwigluConfig() = default;

        /**
         * @brief Configure the inner GELU approximation used by the gate.
         *
         * @param method GELU approximation method
         * @return Self&& for chaining
         */
        template<typename Self>
        Self&& withInnerGeluMethod( this Self&& self, ApproximationMethod method )
        {
            self.inner_gelu_method_ = method;
            return std::forward<Self>( self );
        }

        ApproximationMethod getInnerGeluMethod() const noexcept
        {
            return inner_gelu_method_;
        }

        /**
         * @brief Validate configuration.
         *
         * Currently only the Tanh approximation is supported for the inner GELU.
         */
        void validate() const override
        {
            if ( inner_gelu_method_ != ApproximationMethod::Tanh )
            {
                throw std::invalid_argument( "SwigluConfig::validate: only Gelu Tanh approximation is supported" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) );
                // FIXME: .set( "inner_gelu_method", std::string( toString( inner_gelu_method_ ) ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*p);
            }

            if ( auto m = meta.tryGetString( "inner_gelu_method" ) )
            {
                const std::string v = *m;
                if ( v == "Tanh" ) inner_gelu_method_ = ApproximationMethod::Tanh;
                else if ( v == "Exact" ) inner_gelu_method_ = ApproximationMethod::Exact;
                else if ( v == "Sigmoid" ) inner_gelu_method_ = ApproximationMethod::Sigmoid;
                else throw std::invalid_argument( "SwigluConfig::fromMetadata: unknown inner_gelu_method: " + v );
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            /* FIXME: oss << "SwigluConfig{ inner_gelu_method=" << std::string( toString( inner_gelu_method_ ) )
                << ", precision=" << static_cast<int>(precision_) << " }";*/
            return oss.str();
        }

    private:
        ApproximationMethod inner_gelu_method_ = ApproximationMethod::Tanh;
    };
}
