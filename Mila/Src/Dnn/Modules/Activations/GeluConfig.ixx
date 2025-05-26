/**
 * @file GeluConfig.ixx
 * @brief Configuration interface for the GELU activation module in the Mila DNN framework.
 *
 * Defines the GeluConfig class, providing a type-safe fluent interface for configuring
 * Gaussian Error Linear Unit (GELU) activation function modules. Inherits from ComponentConfig 
 * CRTP base and adds GELU-specific options: approximation method.
 *
 * Exposed as part of the Gelu module via module partitions.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Gelu:Config;

import Dnn.Module;
import Dnn.ComponentConfig;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for GELU module.
     *
     * Provides a type-safe fluent interface for configuring GELU modules.
     */
    export class GeluConfig : public ComponentConfig {
    public:
        /**
         * @brief Approximation methods for the GELU activation function.
         */
        enum class ApproximationMethod {
            Exact,       ///< Exact implementation using error function
            Tanh,        ///< Fast approximation using tanh
            Sigmoid      ///< Fast approximation using sigmoid
        };

        /**
         * @brief Default constructor.
         */
        GeluConfig() = default;

        /**
         * @brief Configure the approximation method for GELU computation.
         *
         * Note: Currently, only the Tanh approximation method is supported.
         * Setting other methods will cause validation to fail when the configuration is used.
         *
         * @param method The approximation method to use (only ApproximationMethod::Tanh is currently supported)
         * @return GeluConfig& Reference to this for method chaining
         */
        GeluConfig& withApproximationMethod( ApproximationMethod method ) {
            approximation_method_ = method;
            return *this;
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

    private:
        ApproximationMethod approximation_method_ = ApproximationMethod::Tanh;
    };
}