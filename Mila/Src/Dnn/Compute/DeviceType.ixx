module;
#include <stdexcept>
#include <algorithm>
#include <string>

export module Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    export enum class DeviceType {
        Cpu,
        Cuda,
    };

    export std::string deviceToString( DeviceType device_type ) {
        switch ( device_type ) {
            case DeviceType::Cpu: return "CPU";
            case DeviceType::Cuda: return "CUDA";
            default:
                throw std::runtime_error( "Invalid DeviceType." );
        }
    };

    export DeviceType toDeviceType( std::string device_type ) {
        // Convert to uppercase for case-insensitive comparison
        std::transform( device_type.begin(), device_type.end(),
            device_type.begin(), ::toupper );

        if ( device_type == "CPU" ) {
            return DeviceType::Cpu;
        }
        if ( device_type == "CUDA" ) {
            return DeviceType::Cuda;
        }
        
        // Handle "AUTO" by checking CUDA availability
        /*if ( device_type == "AUTO" ) {
            auto devices = list_devices();
            bool has_cuda = std::any_of( devices.begin(), devices.end(),
                []( const auto& device ) {
                return device->getDeviceType() == DeviceType::Cuda;
            } );
            return has_cuda ? DeviceType::Cuda : DeviceType::Cpu;
        }*/

        throw std::runtime_error(
            "Invalid compute type '" + device_type + "'. Valid options are: CPU, CUDA, AUTO"
        );
    }
}