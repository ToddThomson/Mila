/**
 * @file CudaResidualOp.ixx
 * @brief Implementation of the CUDA-based residual operation for neural networks.
 */

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
	/**
	 * @brief CUDA implementation of the residual operation for neural networks.
	 *
	 * This class provides a CUDA-based implementation of the residual operation,
	 * which performs element-wise addition of two input tensors.
	 * It is commonly used in residual connections in neural network architectures
	 * such as ResNet and Transformers to help with gradient flow and mitigate the
	 * vanishing gradient problem. The implementation is optimized for NVIDIA GPUs
	 * using CUDA for high-performance computation.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TOutput The data type of the output tensor elements (defaults to the input type).
	 */
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaResidualOp : public BinaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		/**
		 * @brief Constructs a new CUDA Residual operation.
		 *
		 * Initializes the operation with the CUDA device type and ResidualOp operation type.
		 */
		CudaResidualOp() : BinaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::ResidualOp ) {}

		/**
		 * @brief Performs the forward pass of the residual operation on CUDA.
		 *
		 * Adds two input tensors element-wise and stores the result in the output tensor.
		 * The computation is performed on the GPU using CUDA kernels for optimal performance.
		 *
		 * @param input1 The first input tensor to be added.
		 * @param input2 The second input tensor to be added.
		 * @param parameters Additional parameters (not used in this operation).
		 * @param properties Additional attributes for the operation.
		 * @param output The output tensor where the results will be stored.
		 * @param output_state Cache for intermediate results (not used in this operation).
		 */
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

		/**
		 * @brief Gets the name of this operation.
		 *
		 * @return std::string The name of the operation ("Cuda::ResidualOp").
		 */
		std::string getName() const override {
			return "Cuda::ResidualOp";
		}
	};

	/**
	 * @brief Class responsible for registering the CudaResidualOp operation.
	 *
	 * The CudaResidualOpRegistrar class registers the CudaResidualOp operation with the OperationRegistry.
	 * It associates the operation name "Cuda::ResidualOp" with a factory function that creates
	 * instances of CudaResidualOp.
	 */
	export class CudaResidualOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaResidualOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaResidualOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::ResidualOp"
		 * with a factory function that creates instances of CudaResidualOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::ResidualOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaResidualOp<float, float>>();
				}
			);

			// Add additional precision variants if needed, for example:
			/*
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaResidualOp<float, half>>();
				}
			);
			*/
		}

		/**
		 * @brief Self-registration mechanism that registers the operation during startup.
		 *
		 * This static member ensures the operation is registered when the program starts
		 * without requiring explicit registration calls.
		 */
		static inline bool isRegistered = []() {
			registerOperations();
			return true;
			}();
	};
}
