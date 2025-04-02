/**
 * @file CudaEncoderOp.ixx
 * @brief Implementation of the CUDA-based Encoder operation for transformer models.
 */

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
	/**
	 * @brief CUDA implementation of the Encoder operation for transformer models.
	 *
	 * This class provides a CUDA-based implementation of the Encoder operation, which performs
	 * token embedding lookups and positional embedding additions. It transforms discrete
	 * token IDs into continuous vector representations by combining:
	 * 1. Token embeddings from a learned vocabulary table (wte)
	 * 2. Positional embeddings that encode sequence position information (wpe)
	 *
	 * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance computation,
	 * supporting both integer and half-precision floating-point operations.
	 *
	 * @tparam TInput The data type of the input tensor elements (typically uint16_t or int for token IDs).
	 * @tparam TPrecision The data type used for computation and output (typically half or float).
	 */
	export
		template<typename TInput = uint16_t, typename TPrecision = half>
		requires (std::is_same_v<TInput, int> || std::is_same_v<TInput, uint16_t>) &&
	(std::is_same_v<TPrecision, half> || std::is_same_v<TPrecision, float>)
		class CudaEncoderOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
		public:
			/**
			 * @brief Constructs a new CUDA Encoder operation.
			 *
			 * Initializes the operation with the CUDA device type and EncoderOp operation type.
			 */
			CudaEncoderOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( DeviceType::Cuda, OperationType::EncoderOp ) {}

			/**
			 * @brief Performs the forward pass of the Encoder operation on CUDA.
			 *
			 * Transforms input token IDs into continuous embeddings by:
			 * 1. Looking up token embeddings from the embedding table (wte)
			 * 2. Adding positional embeddings (wpe) based on token position
			 *
			 * The computation is performed on the GPU using CUDA kernels for optimal performance.
			 *
			 * @param input Input tensor of shape [B, TElementType] containing token IDs, where B is batch size and TElementType is sequence length.
			 * @param parameters Vector of parameter tensors [wte, wpe] where wte is of shape [V, C] (vocabulary size × embedding dimension)
			 *                   and wpe is of shape [maxT, C] (maximum sequence length × embedding dimension).
			 * @param properties Additional attributes for the operation.
			 * @param output Output tensor of shape [B, TElementType, C] containing the resulting embeddings.
			 * @param output_state Cache for intermediate results (not used in this operation).
			 */
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

				// FIXME: cuda_encoder_forward( Y, X, wte->data(), wpe->data(), B, TElementType, C);
			}

			/**
			 * @brief Gets the name of this operation.
			 *
			 * @return std::string The name of the operation ("Cuda::EncoderOp").
			 */
			std::string getName() const override {
				return "Cuda::EncoderOp";
			}
	};

	/**
	 * @brief Class responsible for registering the CudaEncoderOp operation.
	 *
	 * The CudaEncoderOpRegistrar class registers the CudaEncoderOp operation with the OperationRegistry.
	 * It associates the operation name "Cuda::EncoderOp" with a factory function that creates
	 * instances of CudaEncoderOp with appropriate template parameters.
	 */
	export class CudaEncoderOpRegistrar {
	public:
		/**
		 * @brief Registers the CudaEncoderOp operation with the OperationRegistry.
		 *
		 * This function registers the CudaEncoderOp operation for the CUDA device type
		 * with the OperationRegistry. It associates the operation name "Cuda::EncoderOp"
		 * with a factory function that creates instances of CudaEncoderOp.
		 */
		static void registerOperations() {
			const std::string opName = "Cuda::EncoderOp";

			OperationRegistry::instance().registerOperation<uint16_t, float, DeviceType::Cuda>(
				opName,
				[]() -> std::shared_ptr<OperationBase<uint16_t, float, DeviceType::Cuda>> {
					return std::make_shared<CudaEncoderOp<uint16_t, float>>();
				}
			);
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
