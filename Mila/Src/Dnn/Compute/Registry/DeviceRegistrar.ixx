/**
 * @file DeviceRegistrar.ixx
 * @brief Device-agnostic registrar for automatic device discovery and registration.
 */

module;
#include <string>
#include <memory>
#include <functional>
#include <optional>

export module Compute.DeviceRegistrar;

import Compute.DeviceRegistry;
import Compute.ComputeDevice;
import Compute.CpuDevicePlugin;
import Compute.CudaDevicePlugin;
// FUTURE: import Compute.MetalDevicePlugin;
// FUTURE: import Compute.OpenCLDevicePlugin;
// FUTURE: import Compute.VulkanDevicePlugin;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Device-agnostic registrar for automatic device discovery and registration.
     *
     */
    export class DeviceRegistrar {
    public:
        static DeviceRegistrar& instance() {
            static DeviceRegistrar instance;
            return instance;
        }

        DeviceRegistrar(const DeviceRegistrar&) = delete;
        DeviceRegistrar& operator=(const DeviceRegistrar&) = delete;

    private:
        
        DeviceRegistrar() {
            registerAllDevices();
        }
        
        static void registerAllDevices() {
            auto& registry = DeviceRegistry::instance();

            // Register CPU factory (CPU always available)
            registry.registerDeviceType( CpuDevicePlugin::getPluginName(),
                                        CpuDevicePlugin::getDeviceFactory() );

            // Register CUDA factory if available
            auto cudaFactoryOpt = CudaDevicePlugin::getDeviceFactory();
            
            if (cudaFactoryOpt.has_value()) {
                registry.registerDeviceType( CudaDevicePlugin::getPluginName(),
                                            cudaFactoryOpt.value() );
            }

            // FUTURE: Enable when device plugins are implemented

            // MetalDevicePlugin::registerDeviceFactory();
            // OpenCLDevicePlugin::registerDeviceFactory();
            // VulkanDevicePlugin::registerDeviceFactory();
        }
    };
}