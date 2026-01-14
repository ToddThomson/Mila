/**
 * @file ComponentConfig.ixx
 * @brief Base configuration interface for DNN components.
 *
 * Provides construction-time configuration primitives shared by all component
 * configuration types (precision policy and serialization/validation hooks).
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

export module Dnn.ComponentConfig;

import Serialization.Metadata;
import Compute.Precision;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base for component configuration objects.
     *
     * ComponentConfig defines the public API common to all configuration
     * objects:
     *  - serialization to/from the framework's metadata abstraction
     *  - configuration validation
     *  - a compact string summary
     *
     * Implementations are expected to override the pure-virtual members and
     * include base-field handling (precision_) when appropriate.
     */
    export class ComponentConfig
    {
    public:

        /**
         * @brief Virtual destructor for polymorphic base.
         */
        virtual ~ComponentConfig() = default;

        /**
         * @brief Convert configuration into a SerializationMetadata object.
         *
         * Implementations should include any fields required to fully reconstruct
         * the configuration via `fromMetadata`.
         *
         * @return SerializationMetadata Metadata representation of the config.
         */
        virtual SerializationMetadata toMetadata() const = 0;

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Implementations should read available keys and leave missing keys
         * at their current/default values to preserve forward/backward
         * compatibility.
         *
         * @param meta Metadata to read configuration values from.
         */
        virtual void fromMetadata( const SerializationMetadata& meta ) = 0;

        /**
         * @brief Validate configuration parameters.
         *
         * Called by callers to ensure the configuration represents a valid,
         * constructible component. Implementations must throw
         * std::invalid_argument (or a derived exception) when validation fails.
         *
         * @throws std::invalid_argument If the configuration is invalid.
         */
        virtual void validate() const = 0;

        /**
         * @brief Fluent setter for the compute precision policy.
         *
         * Sets the precision policy used during component construction and
         * returns the concrete configuration object for chaining.
         *
         * @tparam Self CRTP/self type (deduced via C++23 `this` parameter)
         * @param policy Precision policy to set.
         * @return Self&& Reference to the concrete config for chaining.
         */
        template <typename Self>
        Self&& withPrecisionPolicy( this Self&& self, ComputePrecision::Policy policy )
        {
            self.precision_ = policy;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the configured precision policy.
         *
         * @return ComputePrecision::Policy The configured precision policy.
         */
        ComputePrecision::Policy getPrecisionPolicy() const
        {
            return precision_;
        }

        /**
         * @brief Produce a short, human-readable summary of the configuration.
         *
         * Implementations should return a compact, single-line description
         * suitable for logging and debugging.
         *
         * @return std::string Human-readable summary of the configuration.
         */
        virtual std::string toString() const = 0;

    protected:

        // REVIEW: Should we deprecate the default precision policy?
        // It is not currently used and is an old development concept.

        /** @brief Precision policy used for computation. */
        ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;
    };
}