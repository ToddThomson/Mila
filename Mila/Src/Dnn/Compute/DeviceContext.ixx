module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.DeviceContext;

import Compute.ComputeDevice;
import Compute.DeviceRegistry;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
	/**
	* @brief The DeviceContext class manages the device context for module and tensor computations.
	*/
	export class DeviceContext {
	public:
		
		/**
		* @brief Gets the singleton instance of the DeviceContext.
		* @return The singleton instance of the DeviceContext.
		*/
		static DeviceContext& instance() {
			static DeviceContext instance;
			if ( !is_initialized_ ) {
				registerDevices();

				// Set the default device to the highest performance CUDA device or CPU if no CUDA devices are found
				try {
					instance.setDevice( "CUDA:0" );
				}
				catch ( const std::runtime_error& ) {
					instance.setDevice( "CPU" );
				}

				is_initialized_ = true;
			}

			return instance;
		}

		/**
		* @brief Gets the current device.
		* @return A shared pointer to the current device.
		*/
		std::shared_ptr<ComputeDevice> getDevice() const {
			return device_;
		}

		/**
		* @brief Sets the current device by name.
		* @param name The name of the device to set.
		* @throws std::runtime_error if the device name is invalid.
		*/
		void setDevice( const std::string& device_name ) {
			device_ = DeviceRegistry::instance().createDevice( device_name );
			
			if ( !device_ ) {
				throw std::runtime_error( "Invalid device name." );
			}
		}

		DeviceContext( const DeviceContext& ) = delete;
		DeviceContext& operator=( const DeviceContext& ) = delete;

	private:
		DeviceContext() = default;
		std::shared_ptr<ComputeDevice> device_;
		static inline bool is_initialized_ = false;

		static void registerDevices() {
			CpuDevice::registerDevice();
			CudaDevice::registerDevices();
		}
	};
}