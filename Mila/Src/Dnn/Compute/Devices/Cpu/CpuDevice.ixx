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
		std::string getDeviceName() const override {
			return "CPU";
		}

		/**
		 * @brief Gets the unique device identifier.
		 *
		 * For CPU devices, this returns -1 to indicate that CPU is not enumerated
		 * like discrete GPU devices. The CPU is treated as a single logical device
		 * without numeric indexing.
		 *
		 * @return int The device identifier (-1 for CPU, indicating no enumeration)
		 */
		constexpr int getDeviceId() const override {
			return -1;
		}
	};
}