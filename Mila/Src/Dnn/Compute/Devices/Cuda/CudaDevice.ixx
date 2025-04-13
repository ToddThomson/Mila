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
	 * within the Mila framework. It handles device initialization, properties
	 * retrieval, and registration in the DeviceRegistry.
	 */
	export class CudaDevice : public ComputeDevice {
	public:
		/** @brief Type alias for device memory resource */
		using MR = DeviceMemoryResource;
		/** @brief Type alias for pinned memory resource */
		using PINNED_MR = PinnedMemoryResource;
		/** @brief Type alias for managed memory resource */
		using MANAGED_MR = ManagedMemoryResource;

		/**
		 * @brief Constructs a CudaDevice with specified device ID.
		 *
		 * @param device_id The CUDA device ID to initialize (default: 0).
		 * @throws std::runtime_error If device cannot be set.
		 */
		explicit CudaDevice( int device_id = 0 )
			: device_id_( setDevice( device_id ) ), props_( DeviceProps( device_id_ ) ) {}

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
		 * @brief Registers all available CUDA devices with the DeviceRegistry.
		 *
		 * This method discovers all CUDA devices in the system and registers them
		 * with the DeviceRegistry for later instantiation. It performs the registration
		 * only once, even if called multiple times.
		 */
		static void registerDevices() {
			if ( registered_ ) return;

			int deviceCount = 0;
			cudaGetDeviceCount( &deviceCount );

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

		/**
		 * @brief Sets the current CUDA device.
		 *
		 * @param device_id The CUDA device ID to set.
		 * @return int The device ID that was set.
		 * @throws std::runtime_error If the device cannot be set.
		 */
		int setDevice( int device_id ) {
			cudaError_t error = cudaSetDevice( device_id );
			if ( error != cudaSuccess ) {
				std::string errorMsg = "Failed to set CUDA device " + std::to_string( device_id ) +
					": " + cudaGetErrorString( error );
				throw std::runtime_error( errorMsg );
			}
			return device_id;
		}
	};

	//export bool CudaDevice::registered_ = (CudaDevice::registerDevices(), true);
}
