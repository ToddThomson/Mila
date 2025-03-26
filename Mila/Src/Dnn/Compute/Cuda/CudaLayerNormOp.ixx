module;
#include <vector>
#include <memory>
#include <string>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaLayerNormOp;

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
	class CudaLayerNormOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:

		CudaLayerNormOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::LayerNormOp ) {}

		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& attributes,
			Tensor<TOutput, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, DeviceMemoryResource>>>& output_state ) const override {

			const float* X = input.data();
			float* Y = output.data();

			const float* weight = parameters[ 0 ]->data();
			const float* bias = parameters[ 1 ]->data();

			float* mean = output_state[ 0 ]->data();
			float* rstd = output_state[ 1 ]->data();

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ];

			cuda_layernorm_forward( Y, mean, rstd, X, weight, bias, B, T, C );
		}

		static void registerOperation() {
			OperationRegistry<float, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::LayerNormOp", []() -> std::unique_ptr<OperationBase<float, float, DeviceType::Cuda>> {
				return std::make_unique<CudaLayerNormOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::LayerNormOp";
		}
	};
}