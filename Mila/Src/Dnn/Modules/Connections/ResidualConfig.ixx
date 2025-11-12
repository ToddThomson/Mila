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

export module Dnn.Modules.Residual:Config;

import Dnn.ConfigurationBase;
import Dnn.ConnectionType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Residual connection module.
     *
     * ResidualConfig is a lightweight, fluent configuration object consumed by
     * Residual modules and by compute-backend factories. It validates required
     * constraints in `validate()` and exposes accessors used by the module
     * implementation.
     *
     * Note: Some configuration options are currently disabled and marked for future implementation.
     * The base implementation provides core residual connection functionality with:
     * - Simple addition: y = x + F(x)
     * - Optional scaling factor for the residual branch
     * - No projection (input and output features must match)
     * - No gating (to be added in future versions)
     */
    export class ResidualConfig : public ConfigurationBase
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
        // Active Configuration Options
        // ====================================================================

        /**
         * @brief Set the connection type.
         *
         * Currently only Addition is supported.
         *
         * @param ct ConnectionType value
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withConnectionType( ConnectionType ct )
        {
            connection_type_ = ct;
            return *this;
        }

        /**
         * @brief Get the configured connection type.
         *
         * @return ConnectionType The connection type
         */
        ConnectionType getConnectionType() const
        {
            return connection_type_;
        }

        /**
         * @brief Set the scaling factor applied to the residual branch.
         *
         * Some residual variants apply a learned or fixed scaling to the
         * residual branch; default is 1.0 (no scaling).
         *
         * @param factor Scaling factor (positive)
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withScalingFactor( float factor )
        {
            scaling_factor_ = factor;
            return *this;
        }

        /**
         * @brief Get the configured scaling factor.
         *
         * @return float Scaling factor
         */
        float getScalingFactor() const
        {
            return scaling_factor_;
        }

        
        void validate() const override
        {
            // Validate base properties (name, etc.)
            ConfigurationBase::validate();

            if (scaling_factor_ <= 0.0f)
            {
                throw std::invalid_argument( "ResidualConfig: scaling_factor must be > 0" );
            }
        }

        /**
         * @brief Convenience that formats the configuration into a std::string.
         *
         * Uses `appendTo` under the hood to avoid duplicating formatting logic.
         */
        std::string toString() const
        {
            std::ostringstream oss;

            return oss.str();
        }

    private:

        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;
    };
}