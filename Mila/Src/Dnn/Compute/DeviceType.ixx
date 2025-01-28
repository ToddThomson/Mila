module;
#include <stdexcept>

export module Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    export enum class DeviceType {
        kCpu,
        kCuda,
    };

    export std::string deviceToString( DeviceType device_type ) {
        switch ( device_type ) {
            case DeviceType::kCpu: return "CPU";
            case DeviceType::kCuda: return "CUDA";
            default:
                throw std::runtime_error( "Invalid DeviceType." );
        }
    };
}