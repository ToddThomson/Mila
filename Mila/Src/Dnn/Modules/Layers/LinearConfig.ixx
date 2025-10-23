/**
 * @file LinearConfig.ixx
 * @brief Configuration interface for the Linear module in the Mila DNN framework.
 *
 * Defines the LinearConfig class, providing a type-safe fluent interface for configuring
 * Linear (fully connected) layer modules. Inherits from ConfigurationBase CRTP base and adds
 * Linear-specific options: input/output feature dimensions and bias configuration.
 *
 * Exposed as part of the Linear module via module partitions.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Linear:Config;

import Dnn.Module;
import Dnn.ConfigurationBase;

namespace Mila::Dnn
{
    /**
     * @class LinearConfig
     * @brief Configuration object for a Linear (fully connected) layer.
     *
     * LinearConfig provides a minimal, type-safe fluent interface for describing the
     * parameters required to construct a Linear layer: the number of input features,
     * the number of output features, and whether the layer contains a bias term.
     *
     * Typical usage:
     * @code
     * LinearConfig cfg{128, 64};
     * cfg.withBias(true).validate();
     * @endcode
     *
     * @note Instances are lightweight value objects intended to be passed into module
     *       factories or constructors. Validation should be invoked prior to creating
     *       runtime objects to surface configuration errors early.
     */
    export class LinearConfig : public ConfigurationBase {
    public:
        /**
         * @brief Construct a LinearConfig with required feature dimensions.
         *
         * The constructor initializes the two required dimensions for the Linear layer.
         *
         * @param input_features Number of input features (channels). Must be > 0.
         * @param output_features Number of output features (channels). Must be > 0.
         */
        LinearConfig( size_t input_features, size_t output_features )
            : input_features_( input_features ), output_features_( output_features ) {}

        /**
         * @brief Set whether the Linear layer includes a bias term.
         *
         * This method implements a fluent setter: it modifies this configuration and
         * returns a reference to allow chaining of additional setters.
         *
         * @param has_bias True to include a bias parameter, false to omit it.
         * @return LinearConfig& Reference to this configuration (for chaining).
         */
        LinearConfig& withBias( bool has_bias ) {
            has_bias_ = has_bias;
            return *this;
        }

        /**
         * @brief Get the configured number of input features.
         * @return size_t Number of input features configured.
         */
        size_t getInputFeatures() const { return input_features_; }
        
        /**
         * @brief Get the configured number of output features.
         * @return size_t Number of output features configured.
         */
        size_t getOutputFeatures() const { return output_features_; }
        
        /**
         * @brief Query whether the bias term is enabled.
         * @return bool True if bias is enabled; false otherwise.
         */
        bool hasBias() const { return has_bias_; }

        /**
         * @brief Validate the configuration values.
         *
         * This method performs checks that the configuration is usable for constructing
         * a runtime Linear module. It calls the base class validate implementation and
         * then checks Linear-specific constraints.
         *
         * @throws std::invalid_argument If any required parameter is invalid.
         */
        void validate() const {
            ConfigurationBase::validate();

            if ( input_features_ == 0 || output_features_ == 0 ) {
                throw std::invalid_argument( "Input and output features must be greater than zero" );
            }
        }

    private:
        /**
         * @brief Number of input features (channels) expected by the layer.
         *
         * Must be greater than zero.
         */
        size_t input_features_;

        /**
         * @brief Number of output features (channels) produced by the layer.
         *
         * Must be greater than zero.
         */
        size_t output_features_;

        /**
         * @brief Whether the layer has a bias term.
         *
         * Default is true.
         */
        bool has_bias_ = true;
    };
}