/**
 * @file CudaLayerNormOp.ixx
 * @brief Implementation of the CUDA-based Layer Normalization operation for neural networks.
 */

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
	/**
	 * @brief CUDA implementation of the Layer Normalization operation for neural networks.
	 *
	 * This class provides a CUDA-based implementation of the Layer Normalization operation,
	 * which normalizes the activations of a layer for each example in a batch, usually applied
	 * before the activation function. Layer normalization helps stabilize the learning process
	 * and reduce the training time required to learn the parameters of neural networks.
	 *
	 * The normalization is applied across the last dimension (feature dimension) and includes
	 * learnable scale (gamma) and shift (beta) parameters. The implementation is optimized for
	 * NVIDIA GPUs using CUDA for high-performance computation.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TOutput The data type of the output tensor elements (defaults to the input type).
	 */
	export
		template<typename TInput, typename TOutput = TInput>
	class CudaLayerNormOp : public UnaryOperation<TInput, TOutput, DeviceType::Cuda> {
	public:
		/**
		 * @brief Constructs a new CUDA Layer Normalization operation.
		 *
		 * Initializes the operation with the CUDA device type and LayerNormOp operation type.
		 */
		CudaLayerNormOp() : UnaryOperation<TInput, TOutput, DeviceType::Cuda>( DeviceType::Cuda, OperationType::LayerNormOp ) {}

		/**
		 * @brief Performs the forward pass of the Layer Normalization operation on CUDA.
		 *
		 * Normalizes the input tensor across the feature dimension (last dimension) by:
		 * 1. Computing the mean and standard deviation of each sample
		 * 2. Normalizing the values using these statistics
		 * 3. Applying learnable scale and shift parameters
		 *
		 * The computation is performed on the GPU using CUDA kernels for optimal performance.
		 *
		 * @param input Input tensor of shape [B, T, C] to be normalized, where B is batch size,
		 *              T is sequence length, and C is feature dimension.
		 * @param parameters Vector of parameter tensors [weight, bias] where weight (gamma) and
		 *                   bias (beta) are both of shape [C].
		 * @param attributes Additional attributes for the operation.
		 * @param output Output tensor of shape [B, T, C] containing the normalized values.
		 * @param output_state Vector containing tensors for intermediate results [mean, rstd],
		 *                     where mean is the mean values and rstd is the reciprocal of standard deviation.
		 */
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

		/**
		 * @brief Gets the name of this operation.
		 *
		 * @return std::string The name of the operation ("Cuda::LayerNormOp").
		 */
		std::string getName() const override {
			return "Cuda::LayerNormOp";
		}
	};

	/**
	 * @brief Class responsible for registering the CudaLayerNormOp operation.
	 *
	 * The CudaLayerNormOpRegistrar class registers the CudaLayerNormOp operation with the OperationRegistry.
	 * It associates the operation name "Cuda::LayerNormOp" with a factory function that creates
	 * instances of CudaLayerNormOp.
	 */
	export class CudaLayerNormOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaLayerNormOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaLayerNormOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::LayerNormOp"
		 * with a factory function that creates instances of CudaLayerNormOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::LayerNormOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
					return std::make_shared<CudaLayerNormOp<float, float>>();
				}
			);

			// Add additional precision variants if needed, for example:
			/*
			OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
					return std::make_shared<CudaLayerNormOp<float, half>>();
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
