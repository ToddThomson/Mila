/**
 * @file ModuleConfig.ixx
 * @brief Base configuration class for neural network component architecture parameters.
 */

module;
#include <string>

export module Dnn.ModuleConfig;


import Compute.Precision;
import nlohmann.json;

namespace Mila::Dnn
{
	using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Base configuration class for neural network component architecture.
     *
     * ModuleConfig provides common architectural configuration properties:
     * - Name: Component identifier for logging/debugging
     * - Precision policy: Computation precision strategy (performance vs accuracy)
     *
     * These are CONSTRUCTION-TIME parameters that define the component's structure.
     * Runtime state (training mode, device, etc.) is managed by the Module itself.
     */
    export class ModuleConfig
    {
    public:
        virtual ~ModuleConfig() = default;

        /**
         * @brief Serialize configuration to JSON
         */
        //virtual json toJson() const = 0;

        /**
         * @brief Deserialize configuration from JSON
         */
        //virtual void fromJson( const json2& j ) = 0;

        /**
         * @brief Sets the name of the component with fluent interface.
         *
         * @param name The name to assign to this component
         * @return Reference to self for method chaining
         */
        template <typename Self>
        Self&& withName( this Self&& self, std::string name )
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
        Self&& withPrecisionPolicy( this Self&& self, ComputePrecision::Policy policy )
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
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Default implementation includes the configured name and a numeric
         * representation of the precision policy. Concrete configuration types
         * should override this method to expose additional, module-specific
         * parameters.
         */
        virtual std::string toString() const = 0;

        /**
         * @brief Validates the configuration.
         *
         * @throws std::invalid_argument If the configuration is invalid
         */
        virtual void validate() const = 0;

    protected:
        
        /** @brief Module name for identification */
        std::string name_;

        /** @brief Precision policy for computation */
        ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;
    };
}