/**
 * @file LinearConfig.ixx
 * @brief Configuration interface for the Linear module in the Mila DNN framework.
 *
 * Defines the LinearConfig class, providing a type-safe fluent interface for configuring
 * Linear (fully connected) layer modules. Inherits from ModuleConfig CRTP base and adds
 * Linear-specific options: input/output feature dimensions and bias configuration.
 *
 * Exposed as part of the Linear module via module partitions.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Linear:Config;

import Dnn.Module;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Linear module.
     *
     * Provides a type-safe fluent interface for configuring Linear modules.
     */
    export class LinearConfig : public ModuleConfig<LinearConfig> {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param input_features The number of input features
         * @param output_features The number of output features
         */
        LinearConfig( size_t input_features, size_t output_features )
            : input_features_( input_features ), output_features_( output_features ) {}

        /**
         * @brief Configure whether the linear layer uses bias.
         *
         * @param has_bias Whether to include bias term
         * @return LinearConfig& Reference to this for method chaining
         */
        LinearConfig& withBias( bool has_bias ) {
            has_bias_ = has_bias;
            return *this;
        }

        size_t getInputFeatures() const { return input_features_; }
        size_t getOutputFeatures() const { return output_features_; }
        bool hasBias() const { return has_bias_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ModuleConfig<LinearConfig>::validate();

            if ( input_features_ == 0 || output_features_ == 0 ) {
                throw std::invalid_argument( "Input and output features must be greater than zero" );
            }
        }

    private:
        size_t input_features_;
        size_t output_features_;
        bool has_bias_ = true;
    };
}