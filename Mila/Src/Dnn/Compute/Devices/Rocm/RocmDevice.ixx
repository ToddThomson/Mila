/**
 * @file VulkanDevice.ixx
 * @brief Minimal Vulkan-based compute device for the Mila framework.
 *
 * This implementation provides a lightweight, cross-platform placeholder
 * for a Vulkan compute device. It derives from ComputeDevice and exposes
 * device identity and simple capability query methods. Platform-specific
 * Vulkan integration (VkInstance, VkPhysicalDevice, VkDevice, etc.) is
 * intentionally omitted to keep this file portable and buildable on hosts
 * without Vulkan headers.
 */

module;
#include <string>
#include <stdexcept>

export module Compute.VulkanDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing a Vulkan compute device instance.
	 *
	 * This class is a minimal, type-safe representation of a Vulkan device
	 * that can be used by higher-level device registration and execution
	 * context code before full Vulkan runtime integration is implemented.
	 */
	export class VulkanDevice : public ComputeDevice
	{
	public:
		/**
		 * @brief Constructs a VulkanDevice with specified device index.
		 *
		 * @param device_index The Vulkan device index to initialize (0-based).
		 * @throws std::invalid_argument If device_index is negative.
		 */
		explicit VulkanDevice( int device_index )
			: device_index_( validateDeviceIndex( device_index ) )
		{
			// device_name_ can be overridden by a discovery/registry step later.
			device_name_ = "Vulkan:" + std::to_string( device_index_ );
		}

		/**
		 * @brief Gets the Vulkan device index.
		 *
		 * @return int The device index for this Vulkan device (0-based).
		 */
		constexpr int getDeviceId() const override {
			return device_index_;
		}

		/**
		 * @brief Gets the type of this compute device.
		 *
		 * @return DeviceType The device type (Vulkan).
		 */
		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Vulkan;
		}

		/**
		 * @brief Gets the name of this Vulkan device.
		 *
		 * @return std::string The device name (e.g., "Vulkan:0").
		 */
		std::string getDeviceName() const override {
			return device_name_;
		}

		/**
		 * @brief Returns an opaque native handle placeholder.
		 *
		 * In a full implementation this would return a pointer to the
		 * underlying VkPhysicalDevice or a small wrapper. Returns nullptr
		 * until Vulkan integration is added.
		 */
		void* getNativeHandle() const noexcept {
			return nullptr;
		}

		/**
		 * @brief Queries whether device is likely discrete.
		 *
		 * Placeholder: returns false until real discovery is implemented.
		 */
		bool isDiscreteGpu() const noexcept {
			return false;
		}

		/**
		 * @brief Returns an approximate maximum workgroup size.
		 *
		 * Placeholder value; real value should come from physical device limits.
		 */
		size_t getMaxWorkGroupSize() const noexcept {
			return 0;
		}

		/**
		 * @brief Returns an approximate available device memory in bytes.
		 *
		 * Placeholder: returns 0 until integrated with Vulkan memory queries.
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
				throw std::invalid_argument( "Vulkan device index must be non-negative" );
			}
			return idx;
		}
	};
}