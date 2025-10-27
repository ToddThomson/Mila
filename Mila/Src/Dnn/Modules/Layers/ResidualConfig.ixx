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

export module Dnn.Modules.Residual:Config;

import Dnn.ConfigurationBase;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Residual connection module.
     *
     * ResidualConfig is a lightweight, fluent configuration object consumed by
     * Residual modules and by compute-backend factories. It validates required
     * constraints in `validate()` and exposes accessors used by the module
     * implementation to decide whether to allocate projection/gating tensors.
     */
    export class ResidualConfig : public ConfigurationBase
    {
    public:

        //friend std::ostream& operator<<( std::ostream& os, const ResidualConfig& cfg );

        /**
         * @brief Connection types supported by the residual module.
         *
         * Currently only Addition (x + F(x)) is implemented. Other types may be
         * added in the future; factories should validate support for requested
         * connection types.
         */
        enum class ConnectionType
        {
            Addition    ///< Simple addition (y = x + F(x))
        };

        /**
         * @brief Default constructor.
         *
         * Leaves the configuration in a valid default state:
         *  - scaling factor = 1.0
         *  - connection type = Addition
         *  - projection disabled
         *  - gating disabled
         */
        ResidualConfig() = default;

        /**
         * @brief Set the connection type.
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

        /**
         * @brief Enable or disable projection (used when input and output features differ).
         *
         * If projection is enabled the caller must also provide input and output
         * feature sizes via `withInputFeatures` and `withOutputFeatures`.
         *
         * @param enable True to enable projection, false to disable
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withProjection( bool enable )
        {
            projection_enabled_ = enable;
            return *this;
        }

        /**
         * @brief Query whether projection is enabled.
         *
         * @return true if projection is enabled
         */
        bool useProjection() const
        {
            return projection_enabled_;
        }

        /**
         * @brief Set input feature count used for projection tensor allocation.
         *
         * @param in_features Number of input features
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withInputFeatures( std::size_t in_features )
        {
            input_features_ = in_features;
            return *this;
        }

        /**
         * @brief Set output feature count used for projection tensor allocation.
         *
         * @param out_features Number of output features
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withOutputFeatures( std::size_t out_features )
        {
            output_features_ = out_features;
            return *this;
        }

        /**
         * @brief Get configured input feature count.
         *
         * When projection is enabled, this value must be > 0.
         *
         * @return std::size_t Input feature count
         */
        std::size_t getInputFeatures() const
        {
            return input_features_;
        }

        /**
         * @brief Get configured output feature count.
         *
         * When projection is enabled, this value must be > 0.
         *
         * @return std::size_t Output feature count
         */
        std::size_t getOutputFeatures() const
        {
            return output_features_;
        }

        /**
         * @brief Enable or disable gating on the residual path.
         *
         * When gating is enabled the gate size must be provided via
         * `withGateSize`.
         *
         * @param enable True to enable gating
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withGated( bool enable )
        {
            gated_ = enable;
            return *this;
        }

        /**
         * @brief Set the gated parameter size.
         *
         * Must be > 0 when gating is enabled.
         *
         * @param size Gate parameter vector length
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withGateSize( std::size_t size )
        {
            gate_size_ = size;
            return *this;
        }

        /**
         * @brief Query whether gating is enabled.
         *
         * @return true if gating is enabled
         */
        bool isGated() const
        {
            return gated_;
        }

        /**
         * @brief Get configured gate size.
         *
         * @return std::size_t Gate size (may be zero if gating disabled)
         */
        std::size_t getGateSize() const
        {
            return gate_size_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * Ensures required options are present when dependent features are
         * enabled (for example projection or gating). Throws std::invalid_argument
         * on invalid configurations.
         *
         * @throws std::invalid_argument If the configuration is invalid
         */
        void validate() const
        {
            // Validate base properties (name, etc.)
            ConfigurationBase::validate();

            if (scaling_factor_ <= 0.0f)
            {
                throw std::invalid_argument( "ResidualConfig: scaling_factor must be > 0" );
            }

            if (projection_enabled_)
            {
                if (input_features_ == 0)
                {
                    throw std::invalid_argument( "ResidualConfig: input_features must be specified when projection is enabled" );
                }

                if (output_features_ == 0)
                {
                    throw std::invalid_argument( "ResidualConfig: output_features must be specified when projection is enabled" );
                }
            }

            if (gated_)
            {
                if (gate_size_ == 0)
                {
                    throw std::invalid_argument( "ResidualConfig: gate_size must be > 0 when gating is enabled" );
                }
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
            appendTo( oss );

            return oss.str();
        }

    private:
        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;

        // FUTURE: Projection control
        bool projection_enabled_ = false;
        std::size_t input_features_ = 0;
        std::size_t output_features_ = 0;

        // FUTURE: Gating control
        bool gated_ = false;
        std::size_t gate_size_ = 0;

        /**
         * @brief Append a compact textual representation of this config to an output stream.
         *
         * This method is the single authoritative formatting implementation used
         * by both `operator<<` and `toString()`. It intentionally avoids heap
         * allocations beyond what std::ostream performs.
         */
        void appendTo( std::ostream& os ) const
        {
            os << "ResidualConfig{";
            os << "name=\"" << getName() << "\"";
            os << ", connection=";

            switch (connection_type_)
            {
                case ConnectionType::Addition:
                    os << "Addition";
                    break;
                default:
                    os << "Unknown";
                    break;
            }

            os << ", scaling_factor=" << scaling_factor_;
            os << ", projection=" << (projection_enabled_ ? "enabled" : "disabled");

            if (projection_enabled_)
            {
                os << ", in_features=" << input_features_;
                os << ", out_features=" << output_features_;
            }

            os << ", gated=" << (gated_ ? "enabled" : "disabled");

            if (gated_)
            {
                os << ", gate_size=" << gate_size_;
            }

            os << ", training=" << (isTraining() ? "true" : "false");
            os << " }";
        }
    };

    /**
     * @brief Stream a ResidualConfig to an output stream.
     *
     * Forwards to `ResidualConfig::appendTo` to keep formatting centralized.
     */
    /*inline std::ostream& operator<<( std::ostream& os, const ResidualConfig& cfg )
    {
        cfg.appendTo( os );

        return os;
    }*/
}