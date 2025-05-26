/**
 * @file ComponentConfig.ixx
 * @brief Modern configuration system using fluent interface design for neural network components
 *
 * Provides a base configuration class with a type-safe fluent interface that all
 * component-specific configuration classes should inherit from.
 */

module;
#include <string>
#include <stdexcept>

export module Dnn.ComponentConfig;

import Compute.Precision;
import Compute.DeviceContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Base configuration class for all neural network components
     *
     * ComponentConfig serves as the foundation for all configuration classes in the Mila DNN
     * framework. It provides common configuration properties like name, precision policy,
     * and training mode that are shared across different neural network components.
     *
     * Derived configuration classes should extend this base with component-specific
     * configuration options and override validate() when needed.
     *
     * @see GeluConfig
     * @see LinearConfig
     */
    export class ComponentConfig {
    public:
        /**
         * @brief Virtual destructor to support proper polymorphic destruction
         */
        virtual ~ComponentConfig() = default;

        /**
         * @brief Sets the name of the component with fluent interface
         *
         * @param name The name to assign to this component
         * @return Reference to self for method chaining
         *
         * @note Names are validated during the validate() call, not at assignment time
         */
        template <typename Self>
        auto& withName( this Self&& self, std::string name ) {
            self.name_ = std::move( name );
            return self;
        }

        /**
         * @brief Sets the compute precision policy with fluent interface
         *
         * @param policy The compute precision policy to use
         * @return Reference to self for method chaining
         */
        template <typename Self>
        auto& withPrecision( this Self&& self, ComputePrecision::Policy policy ) {
            self.precision_ = policy;
            return self;
        }

        /**
         * @brief Sets the training mode with fluent interface
         *
         * @param is_training True to put component in training mode, false for inference mode
         * @return Reference to self for method chaining
         */
        template <typename Self>
        auto& withTraining( this Self&& self, bool is_training ) {
            self.is_training_ = is_training;
            return self;
        }

        /**
         * @brief Gets the configured component name
         *
         * @return const std::string& The component name
         */
        const std::string& getName() const { return name_; }

        /**
         * @brief Gets the configured precision policy
         *
         * @return ComputePrecision::Policy The precision policy
         */
        ComputePrecision::Policy getPrecision() const { return precision_; }

        /**
         * @brief Gets the configured training mode
         *
         * @return bool True if in training mode, false if in inference mode
         */
        bool isTraining() const { return is_training_; }

        /**
         * @brief Validates the configuration
         *
         * Base implementation validates that the component name is not empty.
         * Derived classes should call this base implementation and add their
         * own validation logic.
         *
         * @throws std::invalid_argument If the configuration is invalid
         */
        virtual void validate() const {
            if ( name_.empty() ) {
                throw std::invalid_argument( "name cannot be empty" );
            }
        }

    protected:
        /** @brief Component name, defaults to "unnamed" if not explicitly set */
        std::string name_ = "unnamed";

        /** @brief Precision policy for computation, defaults to Auto */
        ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;

        /** @brief Training mode flag, defaults to false (inference mode) */
        bool is_training_ = false;
    };
}