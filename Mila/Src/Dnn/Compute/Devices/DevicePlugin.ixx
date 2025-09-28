/**
 * @file DevicePlugin.ixx
 * @brief Abstract device plugin interface for device-agnostic registration and discovery.
 *
 * This module defines the core interface that all device plugins must implement to
 * participate in the device registration system. The abstract API ensures consistency
 * across different device types while allowing each plugin to handle device-specific
 * discovery and registration logic internally.
 */

module;
#include <string>

export module Compute.DevicePlugin;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract base class for device plugins in the device registration system.
     *
     * This interface defines the contract that all device plugins must implement to
     * participate in the unified device discovery and registration framework. Each
     * concrete plugin handles device-specific logic while presenting a consistent
     * interface to the DeviceRegistrar.
     *
     * Key design principles:
     * - Device-specific logic encapsulated in concrete implementations
     * - Graceful degradation when device types are unavailable
     * - Exception-safe device discovery and registration
     * - Clean separation between plugin interface and device API dependencies
     *
     * The plugin system supports heterogeneous compute environments by allowing
     * each device type (CPU, CUDA, Metal, OpenCL, Vulkan) to implement its own
     * discovery and registration logic while maintaining interface consistency.
     */
    export class DevicePlugin {
    public:
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         */
        virtual ~DevicePlugin() = default;

        /**
         * @brief Registers all available devices of this plugin's type with the DeviceRegistry.
         *
         * Concrete implementations should:
         * - Perform runtime availability detection for their device type
         * - Enumerate available devices with capability validation
         * - Register each usable device with appropriate factory functions
         * - Handle unavailability gracefully without throwing exceptions
         * - Use consistent device naming conventions
         *
         * This method should be safe to call even when the underlying device runtime
         * is not available or functional.
         *
         * @note Implementations must not throw exceptions - graceful degradation required
         */
        virtual void registerDevices() = 0;

        /**
         * @brief Checks if devices of this plugin's type are available and functional.
         *
         * Performs lightweight availability checking to determine if the device runtime
         * is present and can be used for computation. This is useful for early detection
         * without performing full device enumeration.
         *
         * @return true if the device runtime is available and functional, false otherwise
         *
         * @note Must not throw exceptions - should return false on any error condition
         */
        virtual bool isAvailable() const = 0;

        /**
         * @brief Gets the number of available devices of this plugin's type.
         *
         * Returns the count of devices that could potentially be registered by this
         * plugin. The actual number of registered devices may be lower due to
         * capability filtering or usability checks.
         *
         * @return Number of available devices, or 0 if the device type is not available
         *
         * @note Must not throw exceptions - should return 0 on any error condition
         */
        virtual int getDeviceCount() const = 0;

        /**
         * @brief Gets the human-readable name identifying this plugin's device type.
         *
         * Returns a descriptive name for the device type handled by this plugin.
         * This name is used for logging, debugging, and user-facing messages.
         *
         * Examples: "CPU", "CUDA", "Metal", "OpenCL", "Vulkan"
         *
         * @return String identifying the device type managed by this plugin
         */
        virtual std::string getPluginName() const = 0;

    protected:
        /**
         * @brief Protected default constructor for derived classes.
         *
         * The constructor is protected to ensure the abstract base class
         * can only be instantiated through concrete derived implementations.
         */
        DevicePlugin() = default;

        /**
         * @brief Copy operations deleted to prevent unintended plugin duplication.
         */
        DevicePlugin(const DevicePlugin&) = delete;
        DevicePlugin& operator=(const DevicePlugin&) = delete;

        /**
         * @brief Move operations defaulted for efficient plugin management.
         */
        DevicePlugin(DevicePlugin&&) = default;
        DevicePlugin& operator=(DevicePlugin&&) = default;
    };
}