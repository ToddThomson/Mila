/**
 * @file CudaDevice.ixx
 * @brief Implementation of CUDA-based compute device for the Mila framework.
 */

module;
#include <iostream>
#include <string>
#include <cuda_runtime.h>

export module Compute.CudaDevice;

import Compute.ComputeDevice;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.CudaMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;
import Cuda.DeviceProps;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Class representing a CUDA compute device.
	 *
	 * This class provides an interface to interact with NVIDIA CUDA-capable GPUs
	 * within the Mila framework. It handles device properties retrieval and
	 * registration in the DeviceRegistry. Device activation is managed by the
	 * DeviceContext class.
	 */
	export class CudaDevice : public ComputeDevice {
	public:
		/** @brief Type alias for device memory resource */
		using MR = CudaMemoryResource;
		/** @brief Type alias for pinned memory resource */
		using PINNED_MR = CudaPinnedMemoryResource;
		/** @brief Type alias for managed memory resource */
		using MANAGED_MR = CudaManagedMemoryResource;

		/**
		 * @brief Constructs a CudaDevice with specified device ID.
		 *
		 * @param device_id The CUDA device ID to initialize.
		 */
		explicit CudaDevice( int device_id )
			: device_id_( device_id ), props_( DeviceProps( device_id_ ) ) {}

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
			return "CUDA:" + std::to_string( device_id_ );
		}

		/**
		 * @brief Gets the properties of this CUDA device.
		 *
		 * @return const DeviceProps& Reference to the device properties.
		 */
		const DeviceProps& getProperties() const
		{
			return props_;
		}

		/**
		 * @brief Checks if the device supports FP16 precision.
		 *
		 * @return bool True if FP16 is supported, false otherwise.
		 */
		bool isFp16Supported() {
			int major, minor;
			cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device_id_ );
			cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device_id_ );
			// Ampere architecture or newer is required for FP16
			return (major >= 8);
		}

		/**
		 * @brief Checks if the device supports FP8 precision.
		 *
		 * @return bool True if FP8 is supported, false otherwise.
		 */
		bool isFp8Supported() {
			int major, minor;
			cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device_id_ );
			cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device_id_ );

			// Hopper architecture or newer is required for FP8
			return (major >= 9);
		}

		/**
		 * @brief Checks if the device supports FP4 precision.
		 *
		 * @return bool True if FP4 is supported, false otherwise.
		 */
		bool isFp4Supported() {
			int major, minor;
			cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device_id_ );
			cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device_id_ );
			// Hopper architecture or newer is required for FP4
			return (major >= 9);
		}

		/**
		* @brief Registers all available CUDA devices with the DeviceRegistry.
		*
		* This method discovers all CUDA devices in the system and registers them
		* with the DeviceRegistry for later instantiation. It performs the registration
		* only once, even if called multiple times.
		*/
		static void registerDevices() {
			if ( registered_ ) return;

			int deviceCount = 0;

			cudaError_t error = cudaGetDeviceCount( &deviceCount );

			if ( error != cudaSuccess ) {
				// FIXME: Utils::Logger::warning( "Failed to get CUDA device count: {}", cudaGetErrorString( error ) );
				return;
			}

			for ( int i = 0; i < deviceCount; i++ ) {
				std::string name = "CUDA:" + std::to_string( i );
				DeviceRegistry::instance().registerDevice( name, [i]() {
					return std::make_shared<CudaDevice>( i );
					} );
			}

			registered_ = true;
		}

	private:
		/** @brief The CUDA device ID */
		int device_id_;
		/** @brief Device properties */
		DeviceProps props_;
		/** @brief Flag indicating if devices have been registered */
		static inline bool registered_{ false };
	};

	//export bool CudaDevice::registered_ = (CudaDevice::registerDevices(), true);
}
