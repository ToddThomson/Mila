module;
#include <iostream>
#include <set>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

export module Compute.CudaDevice;

import Compute.DeviceInterface;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Cuda.DeviceProps;
import Compute.OperationType;
//import Compute.CudaMatMulOp;

namespace Mila::Dnn::Compute
{
	export class CudaDevice : public DeviceInterface {
	public:

		explicit CudaDevice( int device_id = 0 ) 
			: device_id_( setDevice( device_id )), props_( Cuda::DeviceProps( device_id_ ) ) {
		}

		std::set<OperationType> supportedOps() const override {
			return { OperationType::kLayerNormOp, OperationType::kMatMulOp };
		}

		constexpr DeviceType getDeviceType() const override {
			return DeviceType::kCuda;
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

		/*static void registerOperations() {
			OperationRegistry::instance().registerOperation( "CUDA", "LayerNormOp", []() {
				return std::make_shared<Cuda::LayerNormOp<float>>();
				} );
			OperationRegistry::instance().registerOperation( "CUDA", "MatMulOp", []() {
				return std::make_shared<Cuda::MatMulOp<float>>();
				} );
		}*/
		
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