module;
#include <stdexcept>

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
}