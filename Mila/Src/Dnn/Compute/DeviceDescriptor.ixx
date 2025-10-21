export module Compute.DeviceDescriptor;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Lightweight descriptor that identifies a device that can be created.
     *
     * Plugins provide a DeviceDescriptor to DeviceRegistry to advertise devices
     * that can be created. The registry uses the descriptor to construct the
     * correct concrete device instance.
     */
    export struct DeviceDescriptor {
        DeviceType device_type;   ///< Device type (Cpu, Cuda, ...)
        int device_id = -1;       ///< Device id for indexed devices (CUDA:0, etc.), -1 for CPU
    };
}