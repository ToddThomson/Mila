/**
 * @file CudaFullyConnectedOp.ixx
 * @brief Implementation of the CUDA-based Fully Connected operation for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <string>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaMatMulOp;

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
	 * @brief CUDA implementation of the Fully Connected operation for neural networks.
	 *
	 * This class provides a CUDA-based implementation of the Fully Connected operation,
	 * which performs a matrix multiplication between the input and a weight matrix,
	 * optionally adds a bias, and produces an output. The operation is accelerated
	 * on NVIDIA GPUs using CUDA for high-performance computation.
	 *
	 * The implementation delegates the actual matrix multiplication to optimized CUDA kernels
	 * that may leverage libraries like cuBLAS or custom implementations for maximum performance.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TOutput The data type of the output tensor elements (defaults to the input type).
	 */
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaFullyConnectedOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		/**
		 * @brief Constructs a new CUDA Fully Connected operation.
		 *
		 * Initializes the operation with the CUDA device type and FullyConnectedOp operation type.
		 */
		CudaFullyConnectedOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::FullyConnectedOp ) {}

		/**
		 * @brief Performs the forward pass of the Fully Connected operation on CUDA.
		 *
		 * Computes the matrix multiplication between input and weights, adds bias if provided,
		 * and stores the result in the output tensor. The computation is performed on the GPU
		 * using CUDA kernels for optimal performance.
		 *
		 * @param input Input tensor of shape [B, TElementType, C] where B is batch size, TElementType is sequence length, and C is input feature dimension.
		 * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias is of shape [OC].
		 * @param properties Additional attributes for the operation.
		 * @param output Output tensor of shape [B, TElementType, OC] where OC is output feature dimension.
		 * @param output_state Cache for intermediate results (not used in this operation).
		 */
		void forward(
			const Tensor<TInput, DeviceMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<TInput, DeviceMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TOutput, DeviceMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, DeviceMemoryResource>>>& output_state ) const override {

			auto X = input.data();
			auto Y = output.data();

			auto weight = parameters[ 0 ]->data();
			float* bias = { nullptr };

			if ( parameters.size() == 2 ) {
				bias = parameters[ 1 ]->data();
			}

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ];
			int OC = output.shape()[ 2 ];

			cuda_matmul_forward( Y, X, weight, bias, B, T, C, OC );
		}

		/**
		 * @brief Gets the name of this operation.
		 *
		 * @return std::string The name of the operation ("Cuda::FullyConnectedOp").
		 */
		std::string getName() const override {
			return "Cuda::FullyConnectedOp";
		}
	};

	/**
	 * @brief Class responsible for registering the CudaFullyConnectedOp operation.
	 *
	 * The CudaFullyConnectedOpRegistrar class registers the CudaFullyConnectedOp operation with the OperationRegistry.
	 * It associates the operation name "Cuda::FullyConnectedOp" with a factory function that creates instances of CudaFullyConnectedOp.
	 */
	export class CudaFullyConnectedOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaFullyConnectedOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaFullyConnectedOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::FullyConnectedOp"
		 * with a factory function that creates instances of CudaFullyConnectedOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::FullyConnectedOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaFullyConnectedOp<float, float>>();
				}
			);

			// Add additional precision variants if needed, for example:
			/*
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaFullyConnectedOp<float, half>>();
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
