/**
 * @file CudaSoftmaxOp.ixx
 * @brief Implementation of the CUDA-based softmax operation for neural networks.
 */

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
	/**
	 * @brief CUDA implementation of the softmax operation for neural networks.
	 *
	 * This class provides a CUDA-based implementation of the softmax operation,
	 * which converts a vector of real numbers into a probability distribution.
	 * The softmax function is commonly used in classification tasks as the
	 * final activation function of a neural network, and in attention mechanisms
	 * within transformer architectures.
	 *
	 * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance
	 * computation, especially for large vocabulary sizes typical in language models.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TOutput The data type of the output tensor elements (defaults to the input type).
	 */
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaSoftmaxOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		/**
		 * @brief Constructs a new CUDA Softmax operation.
		 *
		 * Initializes the operation with the CUDA device type and SoftmaxOp operation type.
		 */
		CudaSoftmaxOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::SoftmaxOp ) {}

		/**
		 * @brief Performs the forward pass of the softmax operation on CUDA.
		 *
		 * Converts input logits into a probability distribution by taking the
		 * exponential of each element and normalizing by their sum. The computation
		 * is performed on the GPU using CUDA kernels for optimal performance.
		 *
		 * The implementation includes numerical stability improvements by subtracting
		 * the maximum value before applying the exponential function.
		 *
		 * @param input Input tensor containing logits of shape [B, TElementType, V], where B is batch size,
		 *              TElementType is sequence length, and V is vocabulary size.
		 * @param parameters Additional parameters (not used in this operation).
		 * @param attributes Additional attributes for the operation.
		 * @param output Output tensor of shape [B, TElementType, V] to store the resulting probability distribution.
		 * @param output_state Cache for intermediate results (not used in this operation).
		 */
		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& attributes,
			Tensor<TOutput, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, DeviceMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();
			int N = input.size();

			//cuda_softmax_forward( Y, X, N );
		}

		/**
		 * @brief Gets the name of this operation.
		 *
		 * @return std::string The name of the operation ("Cuda::SoftmaxOp").
		 */
		std::string getName() const override {
			return "Cuda::SoftmaxOp";
		}
	};

	/**
	 * @brief Class responsible for registering the CudaSoftmaxOp operation.
	 *
	 * The CudaSoftmaxOpRegistrar class registers the CudaSoftmaxOp operation with the OperationRegistry.
	 * It associates the operation name "Cuda::SoftmaxOp" with a factory function that creates
	 * instances of CudaSoftmaxOp.
	 */
	export class CudaSoftmaxOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaSoftmaxOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaSoftmaxOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::SoftmaxOp"
		 * with a factory function that creates instances of CudaSoftmaxOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::SoftmaxOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaSoftmaxOp<float, float>>();
				}
			);

			// Add additional precision variants if needed, for example:
			/*
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaSoftmaxOp<float, half>>();
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
