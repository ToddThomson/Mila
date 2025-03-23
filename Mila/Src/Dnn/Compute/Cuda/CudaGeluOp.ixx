module;
#include <vector>
#include <iostream>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaGeluOp;

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
	class CudaGeluOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:

		CudaGeluOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::GeluOp ) {}

		void forward(
			const Tensor<TInput, CudaMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, CudaMemoryResource>>>& parameters,
			const OperationProperties& properties,
			Tensor<TOutput, CudaMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();
			int N = input.size();

			cuda_gelu_forward( Y, X, N );
		}

		static void registerOperation() {
			OperationRegistry<float, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::GeluOp", []() -> std::unique_ptr<OperationBase<float, float, DeviceType::Cuda>> {
				return std::make_unique<CudaGeluOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::GeluOp";
		}
	};
}