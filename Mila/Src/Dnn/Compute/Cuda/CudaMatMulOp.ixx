module;
#include <math.h>
#include <vector>
#include <iostream>

#include "Kernels/Cuda.MatMul.h"

export module Compute.CudaMatMulOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaMatMulOp : public OperationBase<TInput, TOutput, CudaDevice> {
	public:

		CudaMatMulOp() : OperationBase<TInput, TOutput, CudaDevice>( DeviceType::Cuda, OperationType::MatMulOp ) {}

		void forward(
			const Tensor<TInput, CudaMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, CudaMemoryResource>>>& parameters,
			Tensor<TOutput, CudaMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& output_state ) const override {

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
			OperationRegistry<float,float,CudaDevice>::instance().registerOperation( DeviceType::Cuda, "Cuda::MatMulOp", []() -> std::unique_ptr<OperationBase<float, float, CudaDevice>> {
				return std::make_unique<CudaMatMulOp<float>>();
				} );
		}

		std::string getName() const override {
			return "Cuda::MatMulOp";
		}
	};
}