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

        // ====================================================================
        // Future Configuration Options (Currently Disabled)
        // ====================================================================

        // TODO: Uncomment when projection is implemented for feature dimension mismatch
        /**
         * @brief Enable or disable projection (used when input and output features differ).
         *
         * If projection is enabled the caller must also provide input and output
         * feature sizes via `withInputFeatures` and `withOutputFeatures`.
         *
         * @param enable True to enable projection, false to disable
         * @return ResidualConfig& Reference to this for method chaining
         */
         // ResidualConfig& withProjection( bool enable )
         // {
         //     projection_enabled_ = enable;
         //     return *this;
         // }

         // TODO: Uncomment when projection is implemented
         /**
          * @brief Query whether projection is enabled.
          *
          * @return true if projection is enabled
          */
          // bool useProjection() const
          // {
          //     return projection_enabled_;
          // }

          // TODO: Uncomment when projection is implemented
          /**
           * @brief Set input feature count used for projection tensor allocation.
           *
           * @param in_features Number of input features
           * @return ResidualConfig& Reference to this for method chaining
           */
           // ResidualConfig& withInputFeatures( int64_t in_features )
           // {
           //     input_features_ = in_features;
           //     return *this;
           // }

           // TODO: Uncomment when projection is implemented
           /**
            * @brief Set output feature count used for projection tensor allocation.
            *
            * @param out_features Number of output features
            * @return ResidualConfig& Reference to this for method chaining
            */
            // ResidualConfig& withOutputFeatures( int64_t out_features )
            // {
            //     output_features_ = out_features;
            //     return *this;
            // }

            // TODO: Uncomment when projection is implemented
            /**
             * @brief Get configured input feature count.
             *
             * When projection is enabled, this value must be > 0.
             *
             * @return int64_t Input feature count
             */
             // int64_t getInputFeatures() const
             // {
             //     return input_features_;
             // }

             // TODO: Uncomment when projection is implemented
             /**
              * @brief Get configured output feature count.
              *
              * When projection is enabled, this value must be > 0.
              *
              * @return int64_t Output feature count
              */
              // int64_t getOutputFeatures() const
              // {
              //     return output_features_;
              // }

              // TODO: Uncomment when gating is implemented
              /**
               * @brief Enable or disable gating on the residual path.
               *
               * When gating is enabled the gate size must be provided via
               * `withGateSize`.
               *
               * @param enable True to enable gating
               * @return ResidualConfig& Reference to this for method chaining
               */
               // ResidualConfig& withGated( bool enable )
               // {
               //     gated_ = enable;
               //     return *this;
               // }

               // TODO: Uncomment when gating is implemented
               /**
                * @brief Set the gated parameter size.
                *
                * Must be > 0 when gating is enabled.
                *
                * @param size Gate parameter vector length
                * @return ResidualConfig& Reference to this for method chaining
                */
                // ResidualConfig& withGateSize( int64_t size )
                // {
                //     gate_size_ = size;
                //     return *this;
                // }

                // TODO: Uncomment when gating is implemented
                /**
                 * @brief Query whether gating is enabled.
                 *
                 * @return true if gating is enabled
                 */
                 // bool isGated() const
                 // {
                 //     return gated_;
                 // }

                 // TODO: Uncomment when gating is implemented
                 /**
                  * @brief Get configured gate size.
                  *
                  * @return int64_t Gate size (may be zero if gating disabled)
                  */
                  // int64_t getGateSize() const
                  // {
                  //     return gate_size_;
                  // }

                  // ====================================================================
                  // Validation
                  // ====================================================================

                  /**
                   * @brief Validate configuration parameters.
                   *
                   * Ensures required options are present and validates the scaling factor.
                   *
                   * @throws std::invalid_argument If the configuration is invalid
                   */
        void validate() const override
        {
            // Validate base properties (name, etc.)
            ConfigurationBase::validate();

            if (scaling_factor_ <= 0.0f)
            {
                throw std::invalid_argument( "ResidualConfig: scaling_factor must be > 0" );
            }

            // TODO: Uncomment when projection is implemented
            // if (projection_enabled_)
            // {
            //     if (input_features_ == 0)
            //     {
            //         throw std::invalid_argument( "ResidualConfig: input_features must be specified when projection is enabled" );
            //     }
            //
            //     if (output_features_ == 0)
            //     {
            //         throw std::invalid_argument( "ResidualConfig: output_features must be specified when projection is enabled" );
            //     }
            // }

            // TODO: Uncomment when gating is implemented
            // if (gated_)
            // {
            //     if (gate_size_ == 0)
            //     {
            //         throw std::invalid_argument( "ResidualConfig: gate_size must be > 0 when gating is enabled" );
            //     }
            // }
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
        // Active configuration
        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;

        // Future configuration options (currently unused)
        // TODO: Uncomment when projection is implemented for feature dimension mismatch
        // Currently: input and output features must match exactly
        // bool projection_enabled_ = false;
        // int64_t input_features_ = 0;
        // int64_t output_features_ = 0;

        // TODO: Uncomment when gating is implemented
        // Currently: no learned gating mechanism
        // bool gated_ = false;
        // int64_t gate_size_ = 0;

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
            os << ", connection=" << connectionTypeToString( connection_type_ );
            os << ", scaling_factor=" << scaling_factor_;

            // TODO: Uncomment when projection is implemented
            // os << ", projection=" << (projection_enabled_ ? "enabled" : "disabled");
            // if (projection_enabled_)
            // {
            //     os << ", in_features=" << input_features_;
            //     os << ", out_features=" << output_features_;
            // }

            // TODO: Uncomment when gating is implemented
            // os << ", gated=" << (gated_ ? "enabled" : "disabled");
            // if (gated_)
            // {
            //     os << ", gate_size=" << gate_size_;
            // }

            os << " }";
        }
    };

    // TODO: Uncomment when stream output is needed
    /**
     * @brief Stream a ResidualConfig to an output stream.
     *
     * Forwards to `ResidualConfig::appendTo` to keep formatting centralized.
     */
     // inline std::ostream& operator<<( std::ostream& os, const ResidualConfig& cfg )
     // {
     //     cfg.appendTo( os );
     //     return os;
     // }
}