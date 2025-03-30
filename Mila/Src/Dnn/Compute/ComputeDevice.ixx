/**
 * @file ComputeDevice.ixx
 * @brief Abstract interface for compute devices in the ML framework.
 */

module;
#include <string>

export module Compute.ComputeDevice;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Abstract interface for compute devices (CPU, CUDA, etc.).
	 *
	 * The ComputeDevice class defines a common interface for different types of
	 * compute devices that can be used for tensor operations and neural network
	 * computations. Each device implementation (CPU, CUDA) inherits from this
	 * class and provides specific implementations of the virtual methods.
	 *
	 * Devices are registered with the DeviceRegistry to allow runtime discovery
	 * and selection of available compute devices.
	 *
	 * @see DeviceType
	 * @see CpuDevice
	 * @see DeviceRegistry
	 */
	export class ComputeDevice {
	public:
		/**
		 * @brief Virtual destructor.
		 *
		 * Ensures proper cleanup of derived device classes.
		 */
		virtual ~ComputeDevice() = default;

		/**
		 * @brief Gets the device type of this compute device.
		 *
		 * @return DeviceType The type of the device (CPU, CUDA, etc.).
		 */
		virtual constexpr DeviceType getDeviceType() const = 0;

		/**
		 * @brief Gets the name of this compute device.
		 *
		 * @return std::string The human-readable name of the device (e.g., "CPU", "NVIDIA RTX 3090").
		 */
		virtual std::string getName() const = 0;
	};
}
