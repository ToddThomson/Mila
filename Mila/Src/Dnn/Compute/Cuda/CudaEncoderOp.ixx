module;
#include <vector>
#include <memory>
#include <string>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaEncoderOp;

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
	template<typename TOutput = float>
	class CudaEncoderOp : public UnaryOperation<int, TOutput, DeviceType::Cuda> {
	public:

		CudaEncoderOp() : UnaryOperation<int, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::EncoderOp ) {}

		void forward(
			const Tensor<int, CudaMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TOutput, CudaMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, CudaMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();

			auto wte = parameters[ 0 ];
			auto wpe = parameters[ 1 ];

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = wte->shape()[ 1 ];

			cuda_encoder_forward( Y, X, wte->data(), wpe->data(), B, T, C);
		}

		static void registerOperation() {
			OperationRegistry<int, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::EncoderOp", []() -> std::unique_ptr<OperationBase<int, float, DeviceType::Cuda>> {
				return std::make_unique<CudaEncoderOp<float>>();
			} );
		}

		std::string getName() const override {
			return "Cuda::EncoderOp";
		}
	};
}