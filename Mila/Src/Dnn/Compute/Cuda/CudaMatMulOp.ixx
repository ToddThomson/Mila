module;
#include <math.h>
#include <iostream>

#include "Kernels/Cuda.MatMul.h"

export module Compute.CudaMatMulOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.DeviceMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
	template<typename T>
	class CudaMatMulOp : public OperationBase<T, DeviceMemoryResource> {
	public:

		CudaMatMulOp() : OperationBase<T, DeviceMemoryResource>( DeviceType::Cuda, OperationType::MatMulOp ) {}

		void forward(
			const std::shared_ptr<Tensor<T, DeviceMemoryResource>> input,
			const std::vector<std::shared_ptr<Tensor<T, DeviceMemoryResource>>>& parameters_,
			std::shared_ptr<Tensor<T, DeviceMemoryResource>> output,
			std::vector<std::shared_ptr<Tensor<T, DeviceMemoryResource>>>& output_cache ) const override {

			auto weight = parameters_[ 0 ];
			auto bias = parameters_[ 1 ];

			int B = input->shape()[ 0 ];
			int T = input->shape()[ 1 ];
			int C = input->shape()[ 2 ];
			int OC = weight->shape()[ 0 ];

			cuda_matmul_forward( output->data(), input->data(), weight->data(), bias->data(),  B, T, C, OC );
		}

		static void registerOperation() {
			OperationRegistry<float,DeviceMemoryResource>::instance().registerOperation( DeviceType::Cuda, "Cuda::MatMulOp", []() -> std::unique_ptr<OperationBase<float, DeviceMemoryResource>> {
				return std::make_unique<CudaMatMulOp<float>>();
				} );
		}

		std::string getName() const override {
			return "Cuda::MatMulOp";
		}
	};
}