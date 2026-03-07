/**
 * @file Swiglu.Config.ixx
 * @brief Configuration for the SwiGLU activation component.
 *
 * Design principle (Mila-wide):
 *   - Constructor parameters are structurally required — no sensible default exists.
 *   - Fluent setters are reserved for optional behavioural parameters that have
 *     well-known defaults. There are no fluent overrides for constructor parameters.
 *
 * SwigluConfig has no structurally required parameters — hidden_dim is determined
 * from the input tensor shape at build time. The default constructor is correct.
 *
 * Optional (fluent): inner_gelu_method (default: Tanh).
 *
 * Typical usage:
 * @code
 * // Default — Tanh approximation.
 * auto cfg = SwigluConfig();
 *
 * // Explicit approximation method.
 * auto cfg = SwigluConfig()
 *     .withInnerGeluMethod( ApproximationMethod::Exact );
 * @endcode
 */

module;
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>

export module Dnn.Components.Swiglu:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ApproximationMethod;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class SwigluConfig : public ComponentConfig
    {
    public:

        SwigluConfig() = default;


        // ====================================================================
        // Validation
        // ====================================================================

        /**
         * @brief Validate configuration.
         *
         * @throws std::invalid_argument if an unsupported approximation method is requested.
         *
         * Note: Exact and Sigmoid approximation methods are not yet implemented.
         */
        void validate() const override
        {}

        // ====================================================================
        // Serialization
        // ====================================================================

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "precision", static_cast<int64_t>(precision_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*v);
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "SwigluConfig( "
                << ", precision=" << static_cast<int>(precision_)
                << " )";
            return oss.str();
        }
    };
}
