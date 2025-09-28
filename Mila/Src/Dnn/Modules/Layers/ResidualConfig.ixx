/**
 * @file ResidualConfig.ixx
 * @brief Configuration interface for the Residual module in the Mila DNN framework.
 *
 * Defines the ResidualConfig class, providing a type-safe fluent interface for configuring
 * Residual connection modules. Inherits from ConfigurationBase CRTP base and adds Residual-specific
 * options such as scaling factor and connection type.
 */

module;
#include <stdexcept>
#include <memory>

export module Dnn.Modules.Residual:Config;

import Dnn.Module;
import Dnn.ConfigurationBase;
import Compute.DeviceType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Residual connection module.
     */
    export class ResidualConfig : public ConfigurationBase {
    public:
        enum class ConnectionType {
            Addition,       ///< Simple addition (x + F(x))
        };

        ResidualConfig() = default;

        void validate() const {
            ConfigurationBase::validate();
        }

    private:
        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;
    };
}