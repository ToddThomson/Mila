/**
 * @file DeviceType.ixx
 * @brief Device type definitions and conversion utilities for compute devices.
 */
module;
#include <stdexcept>
#include <algorithm>
#include <string>

export module Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Enumeration of supported compute device types.
     * 
     * Defines the types of compute devices that can be used for
     * tensor operations and neural network operations.
     */
    export enum class DeviceType {
        Cpu,    ///< CPU device type
        Cuda,   ///< CUDA GPU device type
        Metal,  // FUTURE: Add Metal support
        OpenCL, // FUTURE: OpenCL support
        Vulkan  // FUTURE: Vulkan support
    };

    /**
     * @brief Converts a DeviceType to its string representation.
     * 
     * @param device_type The device type to convert.
     * @return std::string The string representation of the device type ("CPU" or "CUDA").
     * @throws std::runtime_error If the device type is invalid.
     */
    export std::string deviceTypeToString(DeviceType device_type) {
        switch (device_type) {
        case DeviceType::Cpu: return "CPU";
        case DeviceType::Cuda: return "CUDA";
        case DeviceType::Metal: return "Metal";
        case DeviceType::OpenCL: return "OpenCL";
        case DeviceType::Vulkan: return "Vulkan";
        default:
            throw std::runtime_error("Unknown DeviceType value");
        }
    };

    /**
     * @brief Converts a string to the corresponding DeviceType.
     * 
     * Performs case-insensitive matching to convert device type strings
     * to the corresponding enum value.
     * 
     * @param device_type The string representation of the device type.
     * @return DeviceType The corresponding device type enum value.
     * @throws std::runtime_error If the string does not represent a valid device type.
     *                            Valid options are: "CPU", "CUDA", "AUTO".
     * @note "AUTO" option is currently commented out in implementation.
     */
    export DeviceType toDeviceType(std::string device_type) {
        std::transform(device_type.begin(), device_type.end(),
            device_type.begin(), ::toupper);

        if (device_type == "CPU") return DeviceType::Cpu;
        if (device_type == "CUDA") return DeviceType::Cuda;
        if (device_type == "METAL") return DeviceType::Metal;
        if (device_type == "OPENCL") return DeviceType::OpenCL;
        if (device_type == "VULKAN") return DeviceType::Vulkan;

        throw std::runtime_error(
            "Invalid device type '" + device_type + "'. Valid options are: CPU, CUDA, Metal, OpenCL, Vulkan"
        );
    }
}