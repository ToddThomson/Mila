module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
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

namespace Mila::Dnn::Compute
{
	using namespace Mila::Dnn;

	export
	template<typename TInput, typename TPrecision = TInput>
	class CudaGeluOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
	public:

		CudaGeluOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( DeviceType::Cuda, OperationType::GeluOp ) {}

		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TPrecision, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TPrecision, DeviceMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();
			int N = input.size();

			cuda_gelu_forward( Y, X, N );
		}

		std::string getName() const override {
			return "Cuda::GeluOp";
		}
	};

	/**
	* @brief Class responsible for registering the CudaGeluOp operation.
	*
	* The CudaGeluOpRegistrar class registers the CudaGeluOp operation with the OperationRegistry.
	* It associates the operation name "Cuda::GeluOp" with a factory function that creates instances of CudaGeluOp.
	*/
	export class CudaGeluOpRegistrar {
	public:
		/**
		* @brief Registers the CudaGeluOp operation with the OperationRegistry.
		*
		* This function registers the CudaGeluOp operation for the CUDA device type
		* with the OperationRegistry. It associates the operation name "Cuda::GeluOp"
		* with a factory function that creates instances of CudaGeluOp.
		*/
		static void registerOperations() {
			const std::string opName = "Cuda::GeluOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaGeluOp<float,float>>();
				}
			);

			/* FIXME: FP16 Precision is not supported yet
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaGeluOp<float, half>>();
				}
			);*/
		}
	};
}