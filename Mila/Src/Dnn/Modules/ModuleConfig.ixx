/**
 * @file ModuleConfig.ixx
 * @brief Configuration system for Mila DNN modules using CRTP pattern
 * @details
 * This file defines the base configuration class template used by all DNN modules
 * in the Mila framework. It implements the Curiously Recurring Template Pattern (CRTP)
 * to enable fluent interface method chaining while maintaining static type safety.
 *
 * The ModuleConfig class provides common configuration options that are applicable
 * to most neural network modules:
 * - Module naming
 * - Device specification (either by name or context)
 * - Compute precision policy
 * - Training mode toggle
 *
 * Each derived configuration class should inherit from ModuleConfig<DerivedClass>
 * and can add its own specific configuration options while maintaining the fluent
 * interface pattern.
 *
 * @note Configuration validation is performed through the validate() method which
 * should be called before using the configuration to initialize a module.
 */

module;
#include <string>
#include <memory>
#include <stdexcept>

export module Dnn.Module:Config;

import Compute.Precision;
import Compute.DeviceContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    
    /**
     * @brief Base configuration class for all modules using CRTP.
     *
     * @tparam TDerived The derived configuration class
     */
    export template<typename TDerived>
        class ModuleConfig {
        public:
            /**
            * @brief Set the name of the module.
            *
            * @param name The name to set
            * @return TDerived& Reference to derived config for method chaining
            */
            TDerived& withName( std::string name ) {
                name_ = std::move( name );
                return static_cast<TDerived&>(*this);
            }

            /**
            * @brief Set the device name for the module.
            *
            * @param device_name The device name to set
            * @return TDerived& Reference to derived config for method chaining
            */
            TDerived& withDeviceName( std::string device_name ) {
                device_name_ = std::move( device_name );
                context_ = nullptr; // Clear context if device name is set
                return static_cast<TDerived&>(*this);
            }

            /**
            * @brief Set the device context for the module.
            *
            * @param context The context to set
            * @return TDerived& Reference to derived config for method chaining
            */
            TDerived& withContext( std::shared_ptr<DeviceContext> context ) {
                context_ = std::move( context );
                device_name_.clear(); // Clear device name if context is set
                return static_cast<TDerived&>(*this);
            }

            /**
            * @brief Set the compute precision policy.
            *
            * @param policy The precision policy to use
            * @return TDerived& Reference to derived config for method chaining
            */
            TDerived& withPrecision( ComputePrecision::Policy policy ) {
                precision_ = policy;
                return static_cast<TDerived&>(*this);
            }

            /**
            * @brief Set the training mode.
            *
            * @param is_training Whether the module is in training mode
            * @return TDerived& Reference to derived config for method chaining
            */
            TDerived& training( bool is_training ) {
                is_training_ = is_training;
                return static_cast<TDerived&>(*this);
            }

            // Getters for configuration values
            const std::string& getName() const { return name_; }
            const std::string& getDeviceName() const { return device_name_; }
            std::shared_ptr<DeviceContext> getContext() const { return context_; }
            ComputePrecision::Policy getPrecision() const { return precision_; }
            bool isTraining() const { return is_training_; }

            // Validation method that ensures configuration is valid before use
            void validate() const {
                if ( name_.empty() ) {
                    throw std::invalid_argument( "Module name cannot be empty" );
                }

                if ( device_name_.empty() && !context_ ) {
                    throw std::invalid_argument( "Either device name or context must be provided" );
                }
            }

        private:
            std::string name_;
            std::string device_name_;
            std::shared_ptr<DeviceContext> context_ = nullptr;
            ComputePrecision::Policy precision_ = ComputePrecision::Policy::Auto;
            bool is_training_ = false;
    };
}