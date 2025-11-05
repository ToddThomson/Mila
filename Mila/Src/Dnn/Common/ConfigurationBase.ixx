/**
 * @file ConfigurationBase.ixx
 * @brief Base configuration class for neural network component architecture parameters.
 *
 * Configurations define WHAT a component is (structure, hyperparameters),
 * not HOW it's used (training mode, device placement).
 */

module;
#include <string>
#include <stdexcept>
#include <utility>

export module Dnn.ConfigurationBase;

import Compute.Precision;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Base configuration class for neural network component architecture.
     *
     * ConfigurationBase provides common architectural configuration properties:
     * - Name: Component identifier for logging/debugging
     * - Precision policy: Computation precision strategy (performance vs accuracy)
     *
     * These are CONSTRUCTION-TIME parameters that define the component's structure.
     * Runtime state (training mode, device, etc.) is managed by the Module itself.
     *
     * @note Training mode is NOT part of configuration - it's runtime state
     *       managed by Module::setTraining(). Configuration should be immutable
     *       after construction.
     */
    export class ConfigurationBase
    {
    public:
        virtual ~ConfigurationBase() = default;

        /**
         * @brief Sets the name of the component with fluent interface.
         *
         * @param name The name to assign to this component
         * @return Reference to self for method chaining
         */
        template <typename Self>
        decltype(auto) withName( this Self&& self, std::string name )
        {
            self.name_ = std::move( name );
            return std::forward<Self>( self );
        }

        /**
         * @brief Sets the compute precision policy with fluent interface.
         *
         * @param policy The compute precision policy to use
         * @return Reference to self for method chaining
         */
        template <typename Self>
        decltype(auto) withPrecisionPolicy( this Self&& self, ComputePrecision::Policy policy )
        {
            self.precision_ = policy;
            return std::forward<Self>( self );
        }

        /**
         * @brief Gets the configured component name.
         *
         * @return const std::string& The component name
         */
        const std::string& getName() const
        {
            return name_;
        }

        /**
         * @brief Gets the configured precision policy.
         *
         * @return ComputePrecision::Policy The precision policy
         */
        ComputePrecision::Policy getPrecisionPolicy() const
        {
            return precision_;
        }

        /**
         * @brief Validates the configuration.
         *
         * @throws std::invalid_argument If the configuration is invalid
         */
        virtual void validate() const
        {
            if (name_.empty())
            {
                throw std::invalid_argument( "name cannot be empty" );
            }
        }

    protected:
        /** @brief Component name for identification */
        std::string name_ = "unnamed";

        /** @brief Precision policy for computation */
        ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;
    };
}