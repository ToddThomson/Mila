/**
 * @file CudaMultiHeadAttentionOp.ixx
 * @brief Implementation of the CUDA-based Multi-Head Attention operation for transformer models.
 */

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
	/**
	 * @brief CUDA implementation of the Multi-Head Attention operation for transformer models.
	 *
	 * This class provides a CUDA-based implementation of the Multi-Head Attention operation,
	 * which is a key component of transformer architectures. The operation allows the model to
	 * jointly attend to information from different representation subspaces at different positions.
	 *
	 * Multi-Head Attention consists of several attention mechanisms operating in parallel:
	 * 1. Linear projections of the input into query, key, and value vectors
	 * 2. Scaled dot-product attention computation between queries and keys
	 * 3. Applying attention weights to values
	 * 4. Concatenation of attention outputs from different heads
	 *
	 * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance computation.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TOutput The data type of the output tensor elements (defaults to the input type).
	 */
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaMultiHeadAttentionOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		/**
		 * @brief Constructs a new CUDA Multi-Head Attention operation.
		 *
		 * Initializes the operation with the CUDA device type and MultiHeadAttentionOp operation type.
		 */
		CudaMultiHeadAttentionOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::MultiHeadAttentionOp ) {}

		/**
		 * @brief Performs the forward pass of the Multi-Head Attention operation on CUDA.
		 *
		 * Computes attention scores, applies softmax to get attention weights, and uses these
		 * weights to compute a weighted sum of value vectors. This process is performed in
		 * parallel for multiple attention heads, then outputs are concatenated and projected.
		 *
		 * The computation is performed on the GPU using CUDA kernels for optimal performance.
		 *
		 * @param input Input tensor of shape [B, T, C] containing the input sequence, where B is batch size,
		 *              T is sequence length, and C is the input feature dimension.
		 * @param parameters Vector of parameter tensors [weight, bias], where weight contains the query, key,
		 *                   value projections and output projection, and bias contains the corresponding biases.
		 * @param attributes Additional attributes for the operation, such as number of attention heads.
		 * @param output Output tensor of shape [B, T, OC] containing the attention output, where OC is the
		 *               output feature dimension.
		 * @param output_state Cache for intermediate results like attention scores and weights for
		 *                     potential use in backward pass or visualization.
		 */
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

		/**
		 * @brief Gets the name of this operation.
		 *
		 * @return std::string The name of the operation ("Cuda::MultiHeadAttentionOp").
		 */
		std::string getName() const override {
			return "Cuda::MultiHeadAttentionOp";
		}
	};

	/**
	 * @brief Class responsible for registering the CudaMultiHeadAttentionOp operation.
	 *
	 * The CudaMultiHeadAttentionOpRegistrar class registers the CudaMultiHeadAttentionOp operation
	 * with the OperationRegistry. It associates the operation name "Cuda::MultiHeadAttentionOp"
	 * with a factory function that creates instances of CudaMultiHeadAttentionOp.
	 */
	export class CudaMultiHeadAttentionOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaMultiHeadAttentionOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaMultiHeadAttentionOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::MultiHeadAttentionOp"
		 * with a factory function that creates instances of CudaMultiHeadAttentionOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::MultiHeadAttentionOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaMultiHeadAttentionOp<float, float>>();
				}
			);

			// Add additional precision variants if needed, for example:
			/*
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaMultiHeadAttentionOp<float, half>>();
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
