module;
#include <iostream>
#include <set>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

export module Compute.CudaDevice;

import Compute.ComputeDevice;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Cuda.DeviceProps;
import Compute.OperationType;
import Compute.CudaMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;

namespace Mila::Dnn::Compute
{
	export class CudaDevice : public ComputeDevice {
	public:
		using MR = CudaMemoryResource;
		using PINNED_MR = CudaPinnedMemoryResource;
		using MANAGED_MR = CudaManagedMemoryResource;

		explicit CudaDevice( int device_id = 0 ) 
			: device_id_( setDevice( device_id )), props_( Cuda::DeviceProps( device_id_ ) ) {
		}

		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Cuda;
		}

		std::string getName() const override {
			return "CUDA:" + std::to_string( device_id_ );
		}

		const Cuda::DeviceProps& getProperties() const
		{
			return props_;
		}
		
		static void registerDevices() {
			int deviceCount = 0;
			cudaGetDeviceCount( &deviceCount );

			for ( int i = 0; i < deviceCount; i++ ) {
				std::string name = "CUDA:" + std::to_string( i );
				DeviceRegistry::instance().registerDevice( name, [i]() {
					return std::make_shared<CudaDevice>( i );
					} );
			}
		}

	private:
		int device_id_;
		Cuda::DeviceProps props_;
		static bool registered_;

		int setDevice( int device_id ) {
			if ( cudaSetDevice( device_id ) != cudaSuccess ) {
				throw std::runtime_error( "Failed to set Cuda device." );
			}
			return device_id;
		}
	};

	export bool CudaDevice::registered_ = (CudaDevice::registerDevices(), true);
}