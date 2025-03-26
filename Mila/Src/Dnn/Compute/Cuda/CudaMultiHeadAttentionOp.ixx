module;
#include <vector>
#include <memory>
#include <string>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaMHAOp;

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
	class CudaMultiHeadAttentionOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		CudaMultiHeadAttentionOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::MultiHeadAttentionOp ) {}

		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& attributes,
			Tensor<TOutput, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, DeviceMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();

			auto weight = parameters[ 0 ]->data();
			auto bias = parameters[ 1 ]->data();

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ];
			int OC = output.shape()[ 2 ];

			//cuda_mha_forward( Y, X, weight, bias, B, T, C, OC );
		}

		static void registerOperation() {
			OperationRegistry<float, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::MultiHeadAttentionOp", []() -> std::unique_ptr<OperationBase<float, float, DeviceType::Cuda>> {
				return std::make_unique<CudaMultiHeadAttentionOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::MultiHeadAttentionOp";
		}
	};
}