module;
#include <vector>
#include <iostream>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaResidualOp;

import Dnn.Tensor;

import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
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
	class CudaResidualOp : public BinaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:

		CudaResidualOp() : BinaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::ResidualOp ) {}

		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input1,
			const Tensor<TInput, DeviceMemoryResource>& input2,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TOutput, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, DeviceMemoryResource>>>& output_state ) const override {

			auto X1 = input1.data();
			auto X2 = input2.data();
			auto Y = output.data();
			int N = input1.size();

			cuda_residual_forward( Y, X1, X2, N );
		}

		static void registerOperation() {
			OperationRegistry<float, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::ResidualOp", []() -> std::unique_ptr<OperationBase<float, float, DeviceType::Cuda>> {
				return std::make_unique<CudaResidualOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::ResidualOp";
		}
	};
}