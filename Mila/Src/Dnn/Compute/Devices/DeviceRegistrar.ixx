/**
 * @file DeviceRegistrar.ixx
 * @brief Implementation of a class to manage device registration.
 */

module;
#include <string>
#include <memory>
#include <cuda_runtime.h>

export module Compute.DeviceRegistrar;

import Compute.DeviceRegistry;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Class to manage compute device initialization.
     *
     * This class provides a centralized mechanism for registering all available
     * compute devices within the Mila framework. It follows a singleton pattern
     * with lazy initialization to ensure devices are registered only once when needed.
     */
    export class DeviceRegistrar {
    public:
        /**
         * @brief Get the singleton instance of DeviceRegistrar.
         *
         * @return DeviceRegistrar& Reference to the singleton instance.
         */
        static DeviceRegistrar& instance() {
            static DeviceRegistrar instance;

            // Lazy initialization of devices
            if ( !is_initialized_ ) {
                registerDevices();
                is_initialized_ = true;
            }

            return instance;
        }

        // Delete copy constructor and copy assignment operator
        DeviceRegistrar( const DeviceRegistrar& ) = delete;
        DeviceRegistrar& operator=( const DeviceRegistrar& ) = delete;

    private:
        DeviceRegistrar() = default;

        /**
         * @brief Register all available compute devices.
         */
        static void registerDevices() {
            registerCpuDevices();
            registerCudaDevices();
        }

		/**
		 * @brief Register CPU device.
		 */
		static void registerCpuDevices() {
			DeviceRegistry::instance().registerDevice( "CPU", []() {
				return std::make_shared<CpuDevice>();
				} );
		}

		/**
		 * @brief Register all available CUDA devices.
		 */
		static void registerCudaDevices() {
			int deviceCount = 0;
			cudaError_t error = cudaGetDeviceCount( &deviceCount );
			if ( error != cudaSuccess ) {
				// Handle error (e.g., log it)
				return;
			}
			for ( int i = 0; i < deviceCount; i++ ) {
				std::string name = "CUDA:" + std::to_string( i );
				DeviceRegistry::instance().registerDevice( name, [i]() {
					return std::make_shared<CudaDevice>( i );
					} );
			}
		}
		
        /**
		 * @brief Flag indicating whether devices have been initialized.
		 */
        static inline bool is_initialized_ = false;
    };
}