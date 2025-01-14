module;
#include <iostream>
#include <set>
#include <string>
#include <sstream>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

export module Compute.CudaDevice;

import Compute.DeviceInterface;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute::Cuda
{
	export class CudaDevice : public DeviceInterface {
	public:

		explicit CudaDevice( int device_id = 0 ) 
			: device_id_( device_id ) {
			if (cudaSetDevice( device_id_ ) != cudaSuccess) {
				throw std::runtime_error( "Failed to set CUDA device." );
			}
		}

		std::set<Operation> supportedOps() const override {
			return { Operation::LayerNorm, Operation::MatrixMultiply };
		}

		std::string name() const override {
			return "CUDA:" + std::to_string( device_id_ );
		}
		
		static void RegisterDevices() {
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
		static bool registered_;
	};

	export bool CudaDevice::registered_ = (CudaDevice::RegisterDevices(), true);
}