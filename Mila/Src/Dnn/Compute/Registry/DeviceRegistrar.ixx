/**
 * @file DeviceRegistrar.ixx
 * @brief Device-agnostic registrar for automatic device discovery and registration.
 */

export module Compute.DeviceRegistrar;

import Compute.CpuDevice;

import Utils.Logger;

#if defined(MILA_HAS_CUDA)
// TJT: Workaround for MSVC internal compiler error due to perhaps the cuda runtime support
//      in the MSVC 2026 insiders release.
// The direct import of CudaDeviceRegistrar here seems to cause issues.
// The CudaDevice module now provides the CudaDeviceRegistrar class as well.
import Compute.CudaDevice;
//import Compute.CudaDeviceRegistrar;
#endif

#if defined(MILA_HAS_METAL)
import Compute.MetalDevice;
#endif

#if defined(MILA_HAS_ROCM)
import Compute.RocmDevice;
#endif

namespace Mila::Dnn::Compute
{
    /**
     * @brief Device-agnostic registrar for automatic device discovery and registration.
     *
     */
    export class DeviceRegistrar
    {
    public:
        static DeviceRegistrar& instance()
        {
            static DeviceRegistrar instance;
            return instance;
        }

        DeviceRegistrar( const DeviceRegistrar& ) = delete;
        DeviceRegistrar& operator=( const DeviceRegistrar& ) = delete;

    private:

        DeviceRegistrar()
        {
            registerAllDevices();
        }

        static void registerAllDevices()
        {
            CpuDeviceRegistrar::registerDevices();

#if defined(MILA_HAS_CUDA)
            {
                CudaDeviceRegistrar::registerDevices();
            }
#endif

#if defined(MILA_HAS_METAL)
            {
                MetalDeviceRegistrar::registerDevices();
            }
#endif

#if defined(MILA_HAS_ROCM)
            {
                RocmDeviceRegistrar::registerDevices();
            }
#endif
        }
    };
}