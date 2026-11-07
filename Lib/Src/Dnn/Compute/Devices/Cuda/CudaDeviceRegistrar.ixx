/**
 * @file CudaDeviceRegistrar.ixx
 * @brief Finds the CUDA devices Mila can use and registers them with the DeviceRegistry.
 */

module;
#include <cuda_runtime.h>

export module Compute.CudaDeviceRegistrar;

import std;
import Compute.CudaDevice;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;
import Logging.Logger;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA device registrar for device-agnostic registration.
     *
     * Encapsulates CUDA-specific logic for device discovery and registration,
     * providing a clean static interface while handling CUDA runtime API
     * interactions internally.
     */
    export class CudaDeviceRegistrar
    {
    public:

        /**
         * @brief Register CUDA support with the DeviceRegistry.
         *
         * Registers discovered CUDA devices with the global DeviceRegistry.
         * If no devices are present, a warning is emitted and registration
         * is skipped.
         */
        static void registerDevices()
        {
            auto& registry = DeviceRegistry::instance();

            int count = getDeviceCount();

            if ( count == 0 )
            {
                Logging::Logger::warning( "CudaDeviceRegistrar: CUDA not available or no usable devices" );
                return;
            }

            for ( int i = 0; i < count; ++i )
            {
                if ( !isDeviceUsable( i ) )
                {
                    Logging::Logger::warning( std::format( "CudaDeviceRegistrar: CUDA device index '{}' is not useable", i ) );
                    
                    continue;
                }
                
                try
                {
                    DeviceId device_id = Device::Cuda( i );

                    registry.registerDevice( device_id, [device_id]( DeviceConstructionKey key ) -> std::shared_ptr<Device> {
                        return std::make_shared<CudaDevice>( key, device_id );
                        } );
                }
                catch ( const std::exception& ex )
                {
                    Logging::Logger::warning( std::format( "CudaDeviceRegistrar: failed to register CUDA device {}: {}", i, ex.what() ) );
                }
                catch ( ... )
                {
                    Logging::Logger::warning( std::format( "CudaDeviceRegistrar: failed to register CUDA device {}", i ) );
                }
            }
        }

        /**
         * @brief Gets the number of available CUDA devices.
         *
         * @return Number of CUDA devices available, or 0 if CUDA is not available
         */
         static int getDeviceCount() {
             try {
                 int deviceCount = 0;
                 cudaError_t error = cudaGetDeviceCount( &deviceCount );

                 return (error == cudaSuccess) ? deviceCount : 0;
             }
             catch (...) {
                 return 0;
             }
         }

    private:
        /**
         * @brief Checks if a specific CUDA device is usable for computation.
         *
         * Ensures device meets minimal compute capability and memory requirements
         * and that it can be selected via the CUDA runtime.
         */
        static bool isDeviceUsable( int deviceId )
        {
            try
            {
                cudaDeviceProp deviceProp;
                cudaError_t error = cudaGetDeviceProperties( &deviceProp, deviceId );

                if ( error != cudaSuccess )
                {
                    return false;
                }

                if ( deviceProp.major < 3 )
                {
                    return false;
                }

                if ( deviceProp.totalGlobalMem < (1ULL << 30) )
                {
                    return false;
                }

                int computeMode = -1;
                error = cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, deviceId );

                if ( computeMode == cudaComputeModeProhibited )
                {
                    return false;
                }

                // Probing usability must not disturb global state. Every query above takes a
                // device id and binds nothing, so the only reason to touch the current device
                // is to prove it can be selected -- restore it afterwards, or enumeration
                // leaves the LAST probed device current and every context built later
                // inherits the wrong one. cudaDeviceReset() used to follow this and was
                // worse: it destroys the device context, including any stream or allocation
                // already living on it.
                int previous_device = 0;

                if ( cudaGetDevice( &previous_device ) != cudaSuccess )
                {
                    return false;
                }

                cudaError_t setError = cudaSetDevice( deviceId );

                if ( setError != cudaSuccess )
                {
                    return false;
                }

                cudaSetDevice( previous_device );

                return true;
            }
            catch ( ... )
            {
                return false;
            }
        }
    };
}
