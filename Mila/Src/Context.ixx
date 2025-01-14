module;
#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <memory>
#include <mutex>

export module Mila.Context;

import Compute.DeviceInterface;
import Compute.DeviceRegistry;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila
{
	using namespace Mila::Dnn::Compute;

	/**
  * @brief The Context class manages the device context for computations.
  */
	export class Context {
	public:
		/**
   * @brief Gets the singleton instance of the Context.
   * @return The singleton instance of the Context.
   */
		static Context& instance() {
			static Context instance;
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
		void setDevice(const std::string& name) {
			device_ = DeviceRegistry::instance().createDevice(name);
			if (!device_) {
				throw std::runtime_error("Invalid device name.");
			}
		}

		Context(const Context&) = delete;
		Context& operator=(const Context&) = delete;

	private:
		std::shared_ptr<DeviceInterface> device_;

		/**
   * @brief Constructs the Context, registers devices, and sets the default device.
   */
		Context() {
			// Register the CPU device
			DeviceRegistry::instance().registerDevice("CPU", []() {
				return std::make_shared<Cpu::CpuDevice>();
				});
			// Register the CUDA devices
			Cuda::CudaDevice::RegisterDevices();

			// Set the default device
			setDevice("CPU");
		}
	};
}