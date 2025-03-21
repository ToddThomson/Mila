module;
#include <vector>
#include <iostream>
#include <memory>

#include "Kernels/Cuda.Ops.h"


export module Compute.CudaSoftmaxOp;

import Dnn.Tensor;

import Compute.OperationBase;
import Compute.UnaryOperation;
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
	class CudaSoftmaxOp : public UnaryOperation<TInput, TOutput, CudaDevice> {
	public:

		CudaSoftmaxOp() : UnaryOperation<TInput, TOutput, CudaDevice>( DeviceType::Cuda, OperationType::SoftmaxOp ) {}

		void forward(
			const Tensor<TInput, CudaMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, CudaMemoryResource>>>& parameters,
			Tensor<TOutput, CudaMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();
			int N = input.size();

			cuda_softmax_forward( Y, X, N );
		}

		static void registerOperation() {
			OperationRegistry<float, float, CudaDevice>::instance().registerOperation( DeviceType::Cuda, "Cuda::SoftmaxOp", []() -> std::unique_ptr<OperationBase<float, float, CudaDevice>> {
				return std::make_unique<CudaSoftmaxOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::GeluOp";
		}
	};
}