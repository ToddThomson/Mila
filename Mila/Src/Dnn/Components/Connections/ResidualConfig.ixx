/**
 * @file ResidualConfig.ixx
 * @brief Configuration for the Residual component.
 *
 * Provides fluent setters consumed by Residual components and backend factories.
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
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for Residual connection component.
     *
     * ResidualConfig is a lightweight, fluent configuration object consumed by
     * Residual components and by compute-backend factories.
     */
    export class ResidualConfig : public ComponentConfig
    {
    public:

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
            ComponentConfig::validate();

            if ( scaling_factor_ <= 0.0f )
            {
                throw std::invalid_argument( "ResidualConfig: scaling_factor must be > 0" );
            }
        }

        // ====================================================================
        // Serialization (ComponentConfig interface)
        // ====================================================================

        /**
         * @brief Convert configuration to serialization metadata.
         *
         * Produces a SerializationMetadata object containing the configuration
         * fields suitable for writing into an archive by the caller.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "scaling_factor", static_cast<double>( scaling_factor_ ) )
                .set( "connection_type", static_cast<int64_t>( connection_type_ ) );

            return meta;
        }

        /**
         * @brief Populate configuration from serialization metadata.
         *
         * Reads available fields from the provided metadata and updates the
         * configuration object accordingly.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *prec );
            }

            if ( auto sf = meta.tryGetFloat( "scaling_factor" ) )
            {
                scaling_factor_ = static_cast<float>( *sf );
            }

            if ( auto ct = meta.tryGetInt( "connection_type" ) )
            {
                connection_type_ = static_cast<ConnectionType>( *ct );
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

            oss << "ResidualConfig" << std::endl;
            oss << "scaling factor: " << scaling_factor_ << std::endl;
            oss << "connection type: " << connectionTypeToString( connection_type_ ) << std::endl;

            return oss.str();
        }

    private:

        static const char* connectionTypeToString( ConnectionType ct ) noexcept
        {
            switch ( ct )
            {
                case ConnectionType::Addition:
                    return "Addition";
                default:
                    return "Unknown";
            }
        }

        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;
    };
}