module;
#include <vector>
#include <iostream>

#include "Kernels/Cuda.Gelu.h"

export module Compute.CudaGeluOp;

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
	class CudaGeluOp : public OperationBase<TInput, TOutput, CudaDevice> {
	public:

		CudaGeluOp() : OperationBase<TInput, TOutput, CudaDevice>( DeviceType::Cuda, OperationType::GeluOp ) {}

		void forward(
			const Tensor<TInput, CudaMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, CudaMemoryResource>>>& parameters,
			Tensor<TOutput, CudaMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();
			int N = input.size();

			cuda_gelu_forward( Y, X, N );
		}

		static void registerOperation() {
			OperationRegistry<float, float, CudaDevice>::instance().registerOperation( DeviceType::Cuda, "Cuda::GeluOp", []() -> std::unique_ptr<OperationBase<float, float, CudaDevice>> {
				return std::make_unique<CudaGeluOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::GeluOp";
		}
	};
}