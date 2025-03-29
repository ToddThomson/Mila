module;
#include <vector>
#include <memory>
#include <string>

#include "Kernels/Cuda.Ops.h"
#include <cuda_fp16.h>

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
	template<typename TInput = uint16_t, typename TPrecision = half>
	class CudaEncoderOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
	public:

		CudaEncoderOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( DeviceType::Cuda, OperationType::EncoderOp ) {}

		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TPrecision, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TPrecision, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TPrecision, DeviceMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();

			auto wte = parameters[ 0 ];
			auto wpe = parameters[ 1 ];

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = wte->shape()[ 1 ];

			// FIXME: cuda_encoder_forward( Y, X, wte->data(), wpe->data(), B, T, C);
		}

		static void registerOperation() {
			/*OperationRegistry<int, float, DeviceType::Cuda>::instance().registerOperation( DeviceType::Cuda, "Cuda::EncoderOp", []() -> std::unique_ptr<OperationBase<int, float, DeviceType::Cuda>> {
				return std::make_unique<CudaEncoderOp<float>>();
			} );*/
		}

		std::string getName() const override {
			return "Cuda::EncoderOp";
		}
	};
}