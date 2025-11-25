/**
 * @file ResidualConfig.ixx
 * @brief Configuration interface for the Residual module in the Mila DNN framework.
 *
 * ResidualConfig provides a type-safe fluent interface to configure Residual
 * connection modules.
 */

module;
#include <stdexcept>
#include <string>
#include <ostream>
#include <sstream>
#include <cstdint>
#include <utility>

export module Dnn.Components.Residual:Config;

import Dnn.ComponentConfig;
import Dnn.ConnectionType;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @brief Configuration class for Residual connection module.
     *
     * ResidualConfig is a lightweight, fluent configuration object consumed by
     * Residual modules and by compute-backend factories.
     */
    export class ResidualConfig : public ComponentConfig
    {
    public:

        /**
         * @brief Default constructor.
         *
         * Leaves the configuration in a valid default state:
         *  - scaling factor = 1.0
         *  - connection type = Addition
         */
        ResidualConfig() = default;

        // ====================================================================
        // Fluent setters
        // ====================================================================

        /**
         * @brief Set the connection type.
         *
         * Currently only Addition is supported.
         *
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withConnectionType( this Self&& self, ConnectionType ct )
        {
            self.connection_type_ = ct;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the scaling factor applied to the residual branch.
         *
         * @param factor Scaling factor (positive)
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withScalingFactor( this Self&& self, float factor )
        {
            self.scaling_factor_ = factor;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief Get the configured connection type.
         *
         * @return ConnectionType The connection type
         */
        ConnectionType getConnectionType() const noexcept
        {
            return connection_type_;
        }

        /**
         * @brief Get the configured scaling factor.
         *
         * @return float Scaling factor
         */
        float getScalingFactor() const noexcept
        {
            return scaling_factor_;
        }

        // ====================================================================
        // Validation
        // ====================================================================

        void validate() const override
        {
            // Validate base properties (name, etc.)

            if (scaling_factor_ <= 0.0f)
            {
                throw std::invalid_argument( "ResidualConfig: scaling_factor must be > 0" );
            }
        }

        // ====================================================================
        // Serialization (ModuleConfig interface)
        // ====================================================================

        json toJson()
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>( precision_ );
            j["scaling_factor"] = scaling_factor_;
            j["connection_type"] = static_cast<int>( connection_type_ );

            return j;
        }

        void fromJson( const json& j )
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_)>( j.at( "precision" ).get<int>() );
            }

            if ( j.contains( "scaling_factor" ) )
            {
                scaling_factor_ = j.at( "scaling_factor" ).get<float>();
            }

            if ( j.contains( "connection_type" ) )
            {
                connection_type_ = static_cast<ConnectionType>( j.at( "connection_type" ).get<int>() );
            }
        }

        // ====================================================================
        // Utilities
        // ====================================================================

        /**
         * @brief Produce a brief human-readable summary of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            //oss << ComponentConfig::toString();
            oss << "; connection=" << connectionTypeToString( connection_type_ );
            oss << "; scaling=" << scaling_factor_;

            // blank line before return per style
            return oss.str();
        }

    private:

        static const char* connectionTypeToString( ConnectionType ct ) noexcept
        {
            switch ( ct )
            {
                case ConnectionType::Addition: return "Addition";
                default:                      return "Unknown";
            }
        }

        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;
    };
}