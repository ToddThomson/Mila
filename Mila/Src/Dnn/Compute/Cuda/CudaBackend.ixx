module;
#include <iostream>
#include <set>
#include <string>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

export module Compute.CudaBackend;

import Compute.BackendInterface;
import Compute.BackendRegistry;

namespace Mila::Dnn::Compute::Cuda
{
	export class CudaBackend : public BackendInterface {
	public:

		explicit CudaBackend( int deviceId = 0 ) : deviceId_( deviceId ) {
			if (cudaSetDevice( deviceId ) != cudaSuccess) {
				throw std::runtime_error( "Failed to set CUDA device." );
			}
		}

		std::set<Operation> supportedOperations() const override {
			return { Operation::LayerNorm, Operation::MatrixMultiply };
		}

		std::string name() const override {
			return "CUDA";
		}

	private:
		
		int deviceId_;

	};

	struct CudaBackendRegistration {
		CudaBackendRegistration() {
			BackendRegistry::instance().registerBackend( "CUDA", []() {
				return std::make_unique<CudaBackend>(0);
				} );

			// Register additional devices
			int deviceCount = 0;
			cudaGetDeviceCount( &deviceCount );

			for (int i = 1; i < deviceCount; i++) {
				BackendRegistry::instance().registerBackend( "CUDA", [i]() {
					return std::make_unique<CudaBackend>( i );
					} );
			}
		}
	};
}