/**
 * @file GeluConfig.ixx
 * @brief Configuration interface for the GELU activation module in the Mila DNN framework.
 *
 * Defines the GeluConfig class, providing a type-safe fluent interface for configuring
 * Gaussian Error Linear Unit (GELU) activation function modules. Inherits from ModuleConfig 
 * CRTP base and adds GELU-specific options: approximation method.
 *
 * Exposed as part of the Gelu module via module partitions.
 */

module;
#include <memory> // for nlohmann::json to compile in VS2026
#include <stdexcept>
#include <string>
#include <sstream>
#include <string_view>

export module Dnn.Components.Gelu:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

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
            switch (method)
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
         * @brief Default constructor with name "gelu".
         *
         * @note When adding multiple GELU components to a container,
         *       use .withName() to provide unique names.
		 */
        GeluConfig()
            : ComponentConfig( "gelu" )
        {
        }

        /**
         * @brief Configure the approximation method for GELU computation.
         *
         * Note: Currently, only the Tanh approximation method is supported.
         * Setting other methods will cause validation to fail when the configuration is used.
         *
         * Uses C++23 "explicit object parameter" fluent API to enable perfect
         * forwarding of derived types when chaining.
         *
         * @param method The approximation method to use (only ApproximationMethod::Tanh is currently supported)
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
         * Currently, only the Tanh approximation method is supported for GELU computation.
         * Setting other approximation methods will cause validation to fail.
         *
         * @throws std::invalid_argument If validation fails or an unsupported approximation method is selected
         */
        void validate() const {
            ComponentConfig::validate();

            // Validate that only Tanh approximation method is used
            if ( approximation_method_ != ApproximationMethod::Tanh ) {
                throw std::invalid_argument( "Only the Tanh approximation method is currently supported for GELU" );
            }
        }

        /**
         * @brief Serialize this configuration to JSON.
         *
         * The serialized form contains:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "approximation_method" : string (one of "Exact", "Tanh", "Sigmoid")
         */
        json toJson() const
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>(precision_);
            j["approximation_method"] = std::string( GeluConfig::toString( approximation_method_ ) );

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON.
         *
         * Accepts the same keys produced by toJson(). Missing keys leave fields
         * at their current (default) values. Unknown approximation method strings
         * result in std::invalid_argument.
         */
        void fromJson( const json& j )
        {
            // Read name if present
            if (j.contains( "name" ))
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            // Read precision if present (stored as integer)
            if (j.contains( "precision" ))
            {
                precision_ = static_cast<decltype(precision_)>(j.at( "precision" ).get<int>());
            }

            // Read approximation method if present
            if (j.contains( "approximation_method" ))
            {
                const std::string method = j.at( "approximation_method" ).get<std::string>();

                // Map string values to enum; exact-match required
                if (method == "Exact")
                {
                    approximation_method_ = ApproximationMethod::Exact;
                }
                else if (method == "Tanh")
                {
                    approximation_method_ = ApproximationMethod::Tanh;
                }
                else if (method == "Sigmoid")
                {
                    approximation_method_ = ApproximationMethod::Sigmoid;
                }
                else
                {
                    throw std::invalid_argument( "GeluConfig::fromJson: unknown approximation_method: " + method );
                }
            }
        }

        /**
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Overrides ComponentConfig::toString() to include GELU-specific fields.
		 */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GeluConfig: { ";
            oss << "name=" << name_ << ", ";
            oss << "precision=" << static_cast<int>( precision_ ) << ", ";
            oss << "approximation_method=" << static_cast<std::string_view>(GeluConfig::toString( approximation_method_ ));
            oss << " }";
            
            return oss.str();
		}

    private:
        ApproximationMethod approximation_method_ = ApproximationMethod::Tanh;
    };
}