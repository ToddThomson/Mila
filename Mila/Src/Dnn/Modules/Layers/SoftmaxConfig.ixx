/**
 * @file SoftmaxConfig.ixx
 * @brief Configuration interface for the Softmax module in the Mila DNN framework.
 *
 * Defines the SoftmaxConfig class, providing a type-safe fluent interface for configuring
 * Softmax activation function modules.
 */

module;
#include <cstdint>

export module Dnn.Modules.Softmax:Config;

import Dnn.ConfigurationBase;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Softmax module.
     *
     * Provides a type-safe fluent interface for configuring Softmax modules.
     */
    export class SoftmaxConfig : public ConfigurationBase {
    public:
        /**
         * @brief Default constructor.
         */
        SoftmaxConfig() = default;

        /**
         * @brief Set the axis along which to apply the softmax operation.
         *
         * @param axis Dimension for softmax computation (default: -1 for last dimension)
         * @return SoftmaxConfig& Reference to this for method chaining
         */
        SoftmaxConfig& withAxis( int64_t axis ) {
            axis_ = axis;
            return *this;
        }

        /**
         * @brief Get the configured axis value.
         *
         * @return int64_t The axis along which softmax will be computed
         */
        int64_t getAxis() const { return axis_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ConfigurationBase::validate();
            // No additional validation needed for Softmax
        }

    private:
        int64_t axis_ = -1;
    };
}