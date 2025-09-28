/**
 * @file CpuDevice.ixx
 * @brief Implementation of CPU-based compute device for the Mila framework.
 */

module;
#include <string>

export module Compute.CpuDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing a CPU compute device.
	 *
	 * This class provides an interface to interact with CPU compute resources
	 * within the Mila framework. It defines CPU-specific behaviors and capabilities.
	 * Device registration is handled by the DeviceRegistrar class.
	 */
	export class CpuDevice : public ComputeDevice {
	public:
		
		/**
		 * @brief Gets the type of this compute device.
		 *
		 * @return DeviceType The device type (CPU).
		 */
		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Cpu;
		}

		/**
		 * @brief Gets the name of this CPU device.
		 *
		 * @return std::string The device name ("CPU").
		 */
		std::string getName() const override {
			return "CPU";
		}
	};
}