/**
 * @file OpenCLDevice.ixx
 * @brief Minimal OpenCL-based compute device for the Mila framework.
 *
 * Provides a lightweight, portable placeholder implementation deriving from
 * ComputeDevice. Platform-specific OpenCL integration (cl_platform_id,
 * cl_device_id, cl_context, etc.) is intentionally omitted to keep this file
 * buildable on hosts without OpenCL headers while exposing extension points
 * for future integration.
 */

module;
#include <string>
#include <stdexcept>

export module Compute.OpenCLDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing an OpenCL compute device instance.
	 *
	 * Minimal type-safe representation usable by device registration and
	 * execution context code prior to full OpenCL runtime integration.
	 */
	export class OpenCLDevice : public ComputeDevice
	{
	public:
		/**
		 * @brief Constructs an OpenCL device with the specified device index.
		 *
		 * @param device_index The OpenCL device index to initialize (0-based).
		 * @throws std::invalid_argument If device_index is negative.
		 */
		explicit OpenCLDevice( int device_index )
			: device_index_( validateDeviceIndex( device_index ) )
		{
			device_name_ = "OpenCL:" + std::to_string( device_index_ );
		}

		/**
		 * @brief Gets the OpenCL device id (index).
		 *
		 * @return int The device index for this OpenCL device (0-based).
		 */
		constexpr int getDeviceId() const override {
			return device_index_;
		}

		/**
		 * @brief Gets the type of this compute device.
		 *
		 * @return DeviceType The device type (OpenCL).
		 */
		constexpr DeviceType getDeviceType() const override {
			return DeviceType::OpenCL;
		}

		/**
		 * @brief Gets the name of this OpenCL device.
		 *
		 * @return std::string The device name (e.g., "OpenCL:0").
		 */
		std::string getDeviceName() const override {
			return device_name_;
		}

		/**
		 * @brief Returns an opaque native handle placeholder.
		 *
		 * In a full implementation this would return a cl_device_id or a
		 * small wrapper. Returns nullptr until OpenCL integration is added.
		 */
		void* getNativeHandle() const noexcept {
			return nullptr;
		}

		/**
		 * @brief Queries whether the device is likely a discrete GPU.
		 *
		 * Placeholder: returns false until real device discovery is implemented.
		 */
		bool isDiscreteGpu() const noexcept {
			return false;
		}

		/**
		 * @brief Returns an approximate maximum workgroup size.
		 *
		 * Placeholder value; real value should come from CL_DEVICE_MAX_WORK_GROUP_SIZE.
		 */
		size_t getMaxWorkGroupSize() const noexcept {
			return 0;
		}

		/**
		 * @brief Returns an approximate available device memory in bytes.
		 *
		 * Placeholder: returns 0 until integrated with OpenCL memory queries.
		 */
		size_t getDeviceMemorySize() const noexcept {
			return 0;
		}

	private:
		int device_index_;
		std::string device_name_;

		[[nodiscard]] static int validateDeviceIndex( int idx ) {
			if (idx < 0)
			{
				throw std::invalid_argument( "OpenCL device index must be non-negative" );
			}
			return idx;
		}
	};
}