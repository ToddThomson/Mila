/**
 * @file GeluConfig.ixx
 * @brief Configuration for the GELU activation module.
 *
 * Provides fluent setters and serialization/validation hooks for GELU-specific options.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <string_view>

export module Dnn.Components.Gelu:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for GELU module.
     *
     * Provides a type-safe fluent interface for configuring GELU modules.
     */
    export class GeluConfig : public ComponentConfig
    {
    public:

        /**
         * @brief Approximation methods for the GELU activation function.
         */
        enum class ApproximationMethod {
            Exact,       ///< Exact implementation using error function
            Tanh,        ///< Fast approximation using tanh
            Sigmoid      ///< Fast approximation using sigmoid
        };

        static constexpr std::string_view toString( GeluConfig::ApproximationMethod method ) noexcept
        {
            switch ( method )
            {
                case GeluConfig::ApproximationMethod::Exact:
                    return "Exact";
                case GeluConfig::ApproximationMethod::Tanh:
                    return "Tanh";
                case GeluConfig::ApproximationMethod::Sigmoid:
                    return "Sigmoid";
                default:
                    return "Unknown";
            }
        }

        /**
         * @brief Configure the approximation method for GELU computation.
         *
         * Note: Currently, only the Tanh approximation method is supported.
         *
         * @tparam Self Deduction of the concrete config type via C++23 explicit object parameter
         * @param method The approximation method to use
         * @return Self&& Reference to this for method chaining
         */
        template <typename Self>
        Self&& withApproximationMethod( this Self&& self, ApproximationMethod method )
        {
            self.approximation_method_ = method;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the configured approximation method.
         *
         * @return ApproximationMethod The approximation method
         */
        ApproximationMethod getApproximationMethod() const { return approximation_method_; }

        /**
         * @brief Validate configuration parameters.
         *
         * Implementations must throw std::invalid_argument on invalid configuration.
         */
        void validate() const override
        {
            // Only Tanh currently supported
            if ( approximation_method_ != ApproximationMethod::Tanh )
            {
                throw std::invalid_argument(
                    "GeluConfig::validate: only the Tanh approximation method is currently supported" );
            }
        }

        /**
         * @brief Convert configuration into framework metadata.
         *
         * Includes base fields (precision) and GELU-specific options.
         *
         * @return SerializationMetadata Metadata representing this configuration.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "approximation_method", std::string( GeluConfig::toString( approximation_method_ ) ) );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored leaving defaults intact. Unknown approximation
         * method strings result in std::invalid_argument.
         *
         * @param meta Metadata to read configuration values from.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *prec );
            }

            if ( auto method = meta.tryGetString( "approximation_method" ) )
            {
                const std::string m = *method;

                if ( m == "Exact" )
                {
                    approximation_method_ = ApproximationMethod::Exact;
                }
                else if ( m == "Tanh" )
                {
                    approximation_method_ = ApproximationMethod::Tanh;
                }
                else if ( m == "Sigmoid" )
                {
                    approximation_method_ = ApproximationMethod::Sigmoid;
                }
                else
                {
                    throw std::invalid_argument(
                        "GeluConfig::fromMetadata: unknown approximation_method: " + m );
                }
            }
        }

        /**
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Suitable for logging and debugging.
         *
         * @return std::string Human-readable summary of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GeluConfig: { ";
            oss << "precision=" << static_cast<int>( precision_ ) << ", ";
            oss << "approximation_method=" << static_cast<std::string_view>(
                GeluConfig::toString( approximation_method_ ) );
            oss << " }";

            return oss.str();
        }

    private:
        ApproximationMethod approximation_method_ = ApproximationMethod::Tanh;
    };
}