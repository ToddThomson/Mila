module;
#include <vector>
#include <string>

export module Compute.Device;

import Dnn.Tensor;

namespace Mila::Dnn
{
    enum class DeviceType {
        CPU,
        CUDA,
    };

    export std::string deviceToString( DeviceType device_type ) {
        switch ( device_type ) {
            case DeviceType::CPU: return "CPU";
            case DeviceType::CUDA: return "CUDA";
            default:
                return "Unknown device.";
        }
    };

    export template<typename T>
        class Device {
        public:
            virtual ~Device() = default;
         
    };
}