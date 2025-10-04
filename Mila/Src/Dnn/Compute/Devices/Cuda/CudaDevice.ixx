/**
 * @file CudaDevice.ixx
 * @brief Implementation of CUDA-based compute device for the Mila framework.
 */

module;
#include <cuda_runtime.h>
#include <string>
#include <memory>

export module Compute.CudaDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;
//import Compute.CudaDeviceMemoryResource;
//import Compute.CudaPinnedMemoryResource;
//import Compute.CudaManagedMemoryResource;
import Compute.CudaDeviceProps;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing a CUDA compute device instance.
	 *
	 * This class provides an interface to interact with a specific NVIDIA CUDA-capable GPU
	 * within the Mila framework. It handles device properties and capabilities for a single
	 * device instance. Device discovery and registration is handled by CudaDevicePlugin.
	 * Device activation and context management is handled by the DeviceContext class.
	 */
	export class CudaDevice : public ComputeDevice {
	public:

		/**
		 * @brief Constructs a CudaDevice with specified device ID.
		 *
		 * @param device_id The CUDA device ID to initialize.
		 */
		explicit CudaDevice(int device_id)
			: device_id_(device_id), props_(CudaDeviceProps(device_id_)) {
		}

		/**
		 * @brief Gets the CUDA device ID.
		 *
		 * @return int The device ID for this CUDA device.
		 */
		int getDeviceId() const {
			return device_id_;
		}

		/**
		 * @brief Gets the type of this compute device.
		 *
		 * @return DeviceType The device type (CUDA).
		 */
		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Cuda;
		}

		/**
		 * @brief Gets the name of this CUDA device.
		 *
		 * @return std::string The device name in format "CUDA:<device_id>".
		 */
		std::string getName() const override {
			return "CUDA:" + std::to_string(device_id_);
		}

		/**
		 * @brief Gets the properties of this CUDA device.
		 *
		 * @return const DeviceProps& Reference to the device properties.
		 */
		const CudaDeviceProps& getProperties() const
		{
			return props_;
		}

		/**
		 * @brief Checks if the device supports FP16 precision.
		 *
		 * @return bool True if FP16 is supported, false otherwise.
		 */
		bool isFp16Supported() const {
			int major, minor;
			cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id_);
			cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id_);
			// Ampere architecture or newer is required for FP16
			return (major >= 8);
		}

		/**
		 * @brief Checks if the device supports FP8 precision.
		 *
		 * @return bool True if FP8 is supported, false otherwise.
		 */
		bool isFp8Supported() const {
			int major, minor;
			cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id_);
			cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id_);

			// Hopper architecture or newer is required for FP8
			return (major >= 9);
		}

		/**
		 * @brief Checks if the device supports FP4 precision.
		 *
		 * @return bool True if FP4 is supported, false otherwise.
		 */
		bool isFp4Supported() const {
			int major, minor;
			cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id_);
			cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id_);
			// Hopper architecture or newer is required for FP4
			return (major >= 9);
		}

	private:
		/** @brief The CUDA device ID */
		int device_id_;
		/** @brief Device properties */
		CudaDeviceProps props_;
	};
}