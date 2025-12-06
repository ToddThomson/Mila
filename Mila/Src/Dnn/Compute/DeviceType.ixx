/**
 * @file DeviceType.ixx
 * @brief Device type definitions and conversion utilities for compute devices.
 */
module;
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cctype>
#include <string_view>
#include <format>

export module Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Enumeration of supported compute device types.
     *
     * Defines the types of compute devices that can be used for
     * tensor operations and neural network operations.
     */
    export enum class DeviceType
    {
        Cpu,    ///< CPU device type
        Cuda,   ///< CUDA GPU device type
        Metal,  // FUTURE: Add Metal support
        Rocm,   // FUTURE: Add ROCm support
    };

    /**
     * @brief Converts a DeviceType to its string representation.
     *
     * @param device_type The device type to convert.
     * @return std::string The string representation of the device type (Device::Cpu() or "CUDA").
     * @throws std::invalid_argument If the device type is invalid.
     */
    export std::string deviceTypeToString( DeviceType device_type )
    {
        switch ( device_type )
        {
            case DeviceType::Cpu: return "CPU";
            case DeviceType::Cuda: return "CUDA";
            case DeviceType::Metal: return "Metal";
            case DeviceType::Rocm: return "ROCm";
            default:
                throw std::invalid_argument( "Invalid DeviceType value" );
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
     * @throws std::invalid_argument If the string does not represent a valid device type.
     *                            Valid options are: Device::Cpu(), "CUDA", "AUTO".
     */
    export DeviceType toDeviceType( std::string_view device_type )
    {
        std::string type_str( device_type );
        std::transform( 
            type_str.begin(), type_str.end(),
            type_str.begin(),
            []( unsigned char c ) { return std::toupper( c ); } );

        if ( type_str == "CPU" ) return DeviceType::Cpu;
        if ( type_str == "CUDA" ) return DeviceType::Cuda;
        if ( type_str == "METAL" ) return DeviceType::Metal;
        if ( type_str == "ROCM" ) return DeviceType::Rocm;

        throw std::invalid_argument(
            std::format( "Invalid device type '{}'", type_str )
        );
    }
}