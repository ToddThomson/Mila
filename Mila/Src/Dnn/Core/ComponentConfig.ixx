/**
 * @file ModuleConfig.ixx
 * @brief Base configuration class for neural network component architecture parameters.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

export module Dnn.ComponentConfig;

import Compute.Precision;
//import nlohmann.json;

namespace Mila::Dnn
{
	//using json = nlohmann::json;
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
    export class ComponentConfig
    {
    public:
        
        virtual ~ComponentConfig() = default;

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
        virtual void validate() const
        {
            if (!isIdentifier( name_ ))
            {
                throw std::invalid_argument(
                    "ComponentConfig::validate: name must start with a letter and contain only "
                    "letters, digits, '.', '_', '-' (1..128 chars)" );
            }
        }

    protected:

        /**
         * @brief Protected constructor for derived classes.
         *
         * @param default_name A sensible default name for this component type
         */
        explicit ComponentConfig( std::string default_name )
            : name_( std::move( default_name ) )
        {
        }
        
        /** @brief Module name for identification */
        std::string name_;

        /** @brief Precision policy for computation */
        ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;
        
	private:

        static bool isIdentifier( const std::string& s ) noexcept
        {
            // Simple, deterministic ASCII-only check to avoid regex / compiler issues.
            // Rule: start with A-Za-z, then allow A-Za-z0-9 . _ - ; length 1..128.
            constexpr std::size_t kMinLen = 1;
            constexpr std::size_t kMaxLen = 128;

            if (s.size() < kMinLen || s.size() > kMaxLen)
            {
                return false;
            }

            auto isAsciiAlpha = []( unsigned char c ) noexcept {
                return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                };

            auto isAsciiAlphaNum = []( unsigned char c ) noexcept {
                return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                };

            const unsigned char first = static_cast<unsigned char>(s[0]);
            if (!isAsciiAlpha( first ))
            {
                return false;
            }

            for (unsigned char uc : s)
            {
                if (isAsciiAlphaNum( uc ))
                {
                    continue;
                }

                if (uc == '.' || uc == '_' || uc == '-')
                {
                    continue;
                }

                return false;
            }

            return true;
        }
    };  
}