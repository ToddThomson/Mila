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
			const Tensor<T, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<T, DeviceMemoryResource>>>& parameters,
			Tensor<T, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<T, DeviceMemoryResource>>>& output_cache ) const override {

			auto X = input.data();
			auto Y = output.data();

			auto weight = parameters[ 0 ]->data();
			auto bias = parameters[ 1 ]->data();

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ];
			int OC = output.shape()[ 2 ];

			cuda_matmul_forward( Y, X, weight, bias, B, T, C, OC );
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