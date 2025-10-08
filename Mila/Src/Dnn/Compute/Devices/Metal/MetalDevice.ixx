/**
 * @file MetalDevice.ixx
 * @brief Implementation of Metal-based compute device for the Mila framework.
 */

module;
#include <string>
#include <memory>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

export module Compute.MetalDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;
// FUTURE: import Compute.MetalMemoryResource;
// FUTURE: import Compute.MetalManagedMemoryResource;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing a Metal compute device instance.
	 *
	 * This class provides an interface to interact with a specific Apple Metal-capable GPU
	 * within the Mila framework. It handles device properties and capabilities for a single
	 * device instance. Device discovery and registration is handled by MetalDevicePlugin.
	 * Device activation and context management is handled by the DeviceContext class.
	 */
	export class MetalDevice : public ComputeDevice {
	public:
		// FUTURE: Type aliases for Metal memory resources
		// using MR = MetalMemoryResource;
		// using MANAGED_MR = MetalManagedMemoryResource;

		/**
		 * @brief Constructs a MetalDevice with specified device index.
		 *
		 * @param device_index The Metal device index to initialize.
		 */
		explicit MetalDevice(int device_index)
			: device_index_(device_index) {
#ifdef __APPLE__
			@autoreleasepool{
				NSArray<id<MTLDevice>>*devices = MTLCopyAllDevices();
				if (devices && device_index >= 0 && device_index < static_cast<int>(devices.count)) {
					metal_device_ = devices[device_index];
					device_name_ = std::string([metal_device_.name UTF8String]);
				}
			}
#endif
		}

		/**
		 * @brief Gets the Metal device index.
		 *
		 * @return int The device index for this Metal device.
		 */
		int getDeviceIndex() const {
			return device_index_;
		}

		/**
		 * @brief Gets the type of this compute device.
		 *
		 * @return DeviceType The device type (Metal).
		 */
		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Metal;
		}

		/**
		 * @brief Gets the name of this Metal device.
		 *
		 * @return std::string The device name in format "Metal:<device_index>" or actual device name if available.
		 */
		std::string getDeviceName() const override {
			if (!device_name_.empty()) {
				return device_name_;
			}
			return "Metal:" + std::to_string(device_index_);
		}

		/**
		 * @brief Gets the raw Metal device handle.
		 *
		 * @return Native Metal device handle (id<MTLDevice> on Apple platforms, nullptr elsewhere)
		 *
		 * @note This returns a platform-specific handle for advanced Metal operations
		 * @note Returns nullptr on non-Apple platforms
		 */
		void* getNativeDevice() const {
#ifdef __APPLE__
			return (__bridge void*)metal_device_;
#else
			return nullptr;
#endif
		}

		/**
		 * @brief Checks if the device supports Metal Performance Shaders.
		 *
		 * @return bool True if MPS is supported, false otherwise.
		 */
		bool isMPSSupported() const {
#ifdef __APPLE__
			// MPS is available on macOS 10.13+ and iOS 11.0+
			return metal_device_ != nil;
#else
			return false;
#endif
		}

		/**
		 * @brief Checks if the device supports unified memory.
		 *
		 * @return bool True if unified memory is supported, false otherwise.
		 */
		bool isUnifiedMemorySupported() const {
#ifdef __APPLE__
			if (metal_device_) {
				return[metal_device_ hasUnifiedMemory];
			}
#endif
			return false;
		}

		/**
		 * @brief Gets the maximum threads per threadgroup.
		 *
		 * @return size_t Maximum threads per threadgroup, or 0 if unavailable.
		 */
		size_t getMaxThreadsPerThreadgroup() const {
#ifdef __APPLE__
			if (metal_device_) {
				return[metal_device_ maxThreadsPerThreadgroup].width *
					[metal_device_ maxThreadsPerThreadgroup].height *
					[metal_device_ maxThreadsPerThreadgroup].depth;
			}
#endif
			return 0;
		}

		/**
		 * @brief Gets the recommended maximum working set size.
		 *
		 * @return size_t Recommended maximum working set size in bytes, or 0 if unavailable.
		 */
		size_t getRecommendedMaxWorkingSetSize() const {
#ifdef __APPLE__
			if (metal_device_) {
				return[metal_device_ recommendedMaxWorkingSetSize];
			}
#endif
			return 0;
		}

		/**
		 * @brief Checks if the device supports specific Metal GPU family.
		 *
		 * @return bool True if the specified GPU family is supported, false otherwise.
		 */
		bool supportsGPUFamily() const {
#ifdef __APPLE__
			if (metal_device_) {
				// Check for common GPU family support
				return[metal_device_ supportsFamily : MTLGPUFamilyCommon1] ||
					[metal_device_ supportsFamily : MTLGPUFamilyMac2] ||
					[metal_device_ supportsFamily : MTLGPUFamilyApple7];
			}
#endif
			return false;
		}

		/**
		 * @brief Checks if the device supports compute shaders.
		 *
		 * @return bool True if compute shaders are supported, false otherwise.
		 */
		bool supportsComputeShaders() const {
#ifdef __APPLE__
			if (metal_device_) {
				// All modern Metal devices support compute shaders
				return[metal_device_ supportsFamily : MTLGPUFamilyCommon1];
			}
#endif
			return false;
		}

		/**
		 * @brief Gets device memory information.
		 *
		 * @return size_t Available device memory in bytes, or 0 if unavailable.
		 */
		size_t getDeviceMemorySize() const {
#ifdef __APPLE__
			if (metal_device_) {
				if ([metal_device_ hasUnifiedMemory]) {
					// For unified memory, use recommended working set size as approximation
					return[metal_device_ recommendedMaxWorkingSetSize];
				}
				// For discrete GPUs, we'd need additional APIs to get dedicated memory
			}
#endif
			return 0;
		}

	private:
		/** @brief The Metal device index */
		int device_index_;

		/** @brief Cached device name */
		std::string device_name_;

#ifdef __APPLE__
		/** @brief Native Metal device handle */
		id<MTLDevice> metal_device_ = nil;
#endif
	};
}