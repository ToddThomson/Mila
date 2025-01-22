module;
#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <memory>
#include <mutex>

export module Dnn.ModelContext;

import Compute.DeviceInterface;
import Compute.DeviceRegistry;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	* @brief The Context class manages the device context for computations.
	*/
	export class ModelContext {
	public:
		/**
		* @brief Gets the singleton instance of the Context.
		* @return The singleton instance of the Context.
		*/
		static ModelContext& instance() {
			static ModelContext instance;
			return instance;
		}

		/**
		* @brief Gets the current device.
		* @return A shared pointer to the current device.
		*/
		std::shared_ptr<DeviceInterface> device() const {
			return device_;
		}

		/**
		* @brief Sets the current device by name.
		* @param name The name of the device to set.
		* @throws std::runtime_error if the device name is invalid.
		*/
		void setDevice( const std::string& name ) {
			device_ = DeviceRegistry::instance().createDevice( name );
			if ( !device_ ) {
				throw std::runtime_error( "Invalid device name." );
			}
		}

		ModelContext( const ModelContext& ) = delete;
		ModelContext& operator=( const ModelContext& ) = delete;

	private:
		std::shared_ptr<DeviceInterface> device_;

		/**
		* @brief Constructs the Context, registers devices, and sets the default device.
		*/
		ModelContext() {

			//// Register the CPU device
			//DeviceRegistry::instance().registerDevice( "CPU", []() {
			//	return std::make_shared<Cpu::CpuDevice>();
			//	} );
			//
			//// Register the CUDA devices
			//Cuda::CudaDevice::RegisterDevices();

			// TJT: Revisit this selection. Should we default to CUDA if available?
			// Set the default device
			setDevice( "CPU" );
		}
	};
}