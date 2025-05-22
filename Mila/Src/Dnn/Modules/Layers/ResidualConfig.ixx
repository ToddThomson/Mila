/**
 * @file ResidualConfig.ixx
 * @brief Configuration interface for the Residual module in the Mila DNN framework.
 *
 * Defines the ResidualConfig class, providing a type-safe fluent interface for configuring
 * Residual connection modules. Inherits from ModuleConfig CRTP base and adds Residual-specific
 * options such as scaling factor and connection type.
 */

module;
#include <stdexcept>
#include <memory>

export module Dnn.Modules.Residual:Config;

import Dnn.Module;
import Compute.DeviceType;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

    /**
     * @brief Configuration class for Residual connection module.
     */
    export class ResidualConfig : public ModuleConfig<ResidualConfig> {
    public:
        enum class ConnectionType {
            Addition,       ///< Simple addition (x + F(x))
            ScaledAddition, ///< Scaled addition (x + alpha*F(x))
            Gated           ///< Gated connection using learnable parameters
        };

        ResidualConfig() = default;

        /**
         * @brief Set the scaling factor for the residual connection.
         *
         * @param scale Scaling factor (only used for ScaledAddition type)
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withScalingFactor( float scale ) {
            scaling_factor_ = scale;
            return *this;
        }

        /**
         * @brief Set the connection type for the residual.
         *
         * @param type Connection type
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withConnectionType( ConnectionType type ) {
            connection_type_ = type;
            return *this;
        }

        /**
         * @brief Configure whether to include a projection for dimension matching.
         *
         * When input and module output dimensions don't match, a projection linear
         * layer can be automatically added to make dimensions compatible.
         *
         * @param use_projection Whether to use projection when needed
         * @return ResidualConfig& Reference to this for method chaining
         */
        ResidualConfig& withProjection( bool use_projection ) {
            use_projection_ = use_projection;
            return *this;
        }

        /**
         * @brief Set the inner module for the residual connection.
         *
         * The inner module defines the transformation F(x) in the residual formula x + F(x).
         *
         * @param inner_module Shared pointer to the inner module
         * @return ResidualConfig& Reference to this for method chaining
         */
        /* TODO:
        template<DeviceType TDeviceType, typename TInnerInput, typename TInnerOutput>
        ResidualConfig& withInnerModule( std::shared_ptr<Module<TDeviceType, TInnerInput, TInnerOutput>> inner_module ) {
            inner_module_ptr_ = inner_module;
            return *this;
        }*/

        float getScalingFactor() const { return scaling_factor_; }
        ConnectionType getConnectionType() const { return connection_type_; }
        bool useProjection() const { return use_projection_; }

        template<DeviceType TDeviceType, typename TInnerInput, typename TInnerOutput>
        std::shared_ptr<Module<TDeviceType, TInnerInput, TInnerOutput>> getInnerModule() const {
            if ( !inner_module_ptr_ ) {
                throw std::runtime_error( "Inner module not configured for residual connection" );
            }

            auto inner_module = std::dynamic_pointer_cast<Module<TDeviceType, TInnerInput, TInnerOutput>>(
                inner_module_ptr_);

            if ( !inner_module ) {
                throw std::runtime_error( "Inner module type mismatch" );
            }

            return inner_module;
        }

        bool hasInnerModule() const { return inner_module_ptr_ != nullptr; }

        void validate() const {
            ModuleConfig<ResidualConfig>::validate();

            if ( connection_type_ == ConnectionType::ScaledAddition && scaling_factor_ <= 0.0f ) {
                throw std::invalid_argument( "Scaling factor must be positive for scaled addition" );
            }

            if ( !inner_module_ptr_ ) {
                throw std::invalid_argument( "Inner module must be provided for residual connection" );
            }
        }

    private:
        float scaling_factor_ = 1.0f;
        ConnectionType connection_type_ = ConnectionType::Addition;
        bool use_projection_ = true;
        std::shared_ptr<void> inner_module_ptr_ = nullptr;
    };
}