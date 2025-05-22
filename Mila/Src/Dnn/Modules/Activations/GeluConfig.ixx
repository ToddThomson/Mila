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
#include <stdexcept>

export module Dnn.Modules.Gelu:Config;

import Dnn.Module;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for GELU module.
     *
     * Provides a type-safe fluent interface for configuring GELU modules.
     */
    export class GeluConfig : public ModuleConfig<GeluConfig> {
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
         * @param method The approximation method to use
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
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ModuleConfig<GeluConfig>::validate();
            // No additional validation needed for GELU
        }

    private:
        ApproximationMethod approximation_method_ = ApproximationMethod::Tanh;
    };
}