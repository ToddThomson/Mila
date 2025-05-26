/**
 * @file Component.ixx
 * @brief Defines the base Component class for the Mila framework.
 *
 * This file introduces the Component abstraction that serves as the common
 * foundation for both Modules and Operations in the Mila deep neural network framework.
 * Components represent the building blocks of the system that can be configured,
 * have a device context, and maintain a specific compute precision.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <format>

export module Dnn.Component;

import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.Precision;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base class for all components in the Mila framework.
     *
     * The Component class establishes a common interface and behavior for both
     * Modules and Operations. It handles device context management, naming,
     * compute precision, and training mode.
     */
    export class Component {
    public:
        /**
         * @brief Constructor with device name.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param policy The compute precision policy to use (default is Auto).
         */
        explicit Component( const std::string& device_name )
            : device_context_( createContext( device_name ) ) {
            validateDeviceType();
        }

        /**
         * @brief Constructor with a specific device context.
         *
         * @param context The device context to use for this component.
         * @param policy The compute precision policy to use (default is Auto).
         * @throws std::invalid_argument If the provided context is nullptr.
         */
        explicit Component( std::shared_ptr<DeviceContext> context )
            : device_context_( context ) {
            if ( !context ) {
                throw std::invalid_argument( "DeviceContext cannot be nullptr. Please provide a valid DeviceContext." );
            }
            validateDeviceType();
        }

        /**
         * @brief Virtual destructor for proper cleanup in derived classes.
         */
        virtual ~Component() = default;

        /**
         * @brief Get the device context for this component.
         *
         * @return std::shared_ptr<DeviceContext> The device context.
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Set the training mode of this component.
         *
         * @param is_training True if the component is in training mode, false for inference.
         */
        virtual void setTraining( bool is_training ) {
            is_training_ = is_training;
        }

        /**
         * @brief Check if the component is in training mode.
         *
         * @return true If the component is in training mode.
         * @return false If the component is in inference mode.
         */
        bool isTraining() const {
            return is_training_;
        }

        /**
         * @brief Get the name of the component.
         *
         * @return std::string Name of the component.
         */
        std::string getName() const {
            return name_;
        }

        /**
         * @brief Set the name of the component.
         *
         * @param name The name to set. Must not be empty and cannot contain a dot ('.').
         * @throws std::invalid_argument If the name is empty or contains a dot.
         */
        void setName( const std::string& name ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Name must not be empty." );
            }
            name_ = name;
        }

        /**
         * @brief Get the compute precision policy for this component.
         *
         * @return const ComputePrecision& The compute precision policy
         */
        const ComputePrecision& getComputePrecision() const {
            return compute_precision_;
        }

        /**
         * @brief Set the compute precision policy explicitly.
         *
         * @param policy The precision policy to use
         * @return Component& Reference to this component for method chaining
         */
        Component& setComputePrecisionPolicy( ComputePrecision::Policy policy ) {
            compute_precision_.setPolicy( policy );
            return *this;
        }

        /**
         * @brief Convert the component to a string representation.
         *
         * @return std::string String representation of the component.
         */
        virtual std::string toString() const = 0;

        /**
         * @brief Get the device type of the current device context.
         *
         * @return DeviceType The device type (CPU or CUDA).
         */
        DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

    protected:
        /**
         * @brief Validate that the device type matches the derived class requirements.
         *
         * This method should be overridden in derived classes to enforce device type constraints.
         */
        virtual void validateDeviceType() const {}

    private:
        /** @brief The device context used for this component's computations */
        std::shared_ptr<DeviceContext> device_context_;

        /** @brief The compute precision policy for this component */
        ComputePrecision compute_precision_;

        /** @brief The name of the component */
        std::string name_ = "unnamed";

        /** @brief Whether the component is in training mode */
        bool is_training_ = false;

        /**
         * @brief Helper method to create a DeviceContext from a device name.
         *
         * @param device_name Name of the device to create a context for.
         * @return std::shared_ptr<DeviceContext> The created device context.
         */
        static std::shared_ptr<DeviceContext> createContext( const std::string& device_name ) {
            return std::make_shared<DeviceContext>( device_name );
        }
    };
}