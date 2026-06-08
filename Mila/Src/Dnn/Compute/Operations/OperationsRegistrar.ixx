module;
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include <type_traits>
#include <utility>
#include <tuple>
#ifdef MILA_HAS_CUDA
#include <cuda_fp16.h>
#endif

export module Compute.OperationsRegistrar;

import Compute.CpuOperations;
#ifdef MILA_HAS_CUDA
import Compute.CudaOperations;
#endif

namespace Mila::Dnn::Compute
{
	/**
	* @brief Class to manage compute operations initialization.
	*/
	export class OperationsRegistrar {
	public:
		/**
		* @brief Get the singleton instance of OperationsRegistrar.
		* 
		* @return OperationsRegistrar& Reference to the singleton instance.
		*/
		static OperationsRegistrar& instance() {
			static OperationsRegistrar instance;

			// Lazy initialization of operations
			if (!is_initialized_) {
				registerOperations();
				is_initialized_ = true;
			}

			return instance;
		}

		// Delete copy constructor and copy assignment operator
		OperationsRegistrar(const OperationsRegistrar&) = delete;
		OperationsRegistrar& operator=(const OperationsRegistrar&) = delete;

	private:
		OperationsRegistrar() = default;

		/**
		* @brief Initialize the compute operations.
		*/
		static void registerOperations() {
			// TJT: This is rather an ugly way of registering operations but it is all I can think of for now.
			// It's good enough for now. I will revisit

			CpuEncoderOpRegistrar::registerOperations();
			CpuGeluOpRegistrar::registerOperations();
			CpuLayerNormOpRegistrar::registerOperations();
			CpuLinearOpRegistrar::registerOperations();
			CpuAttentionOpRegistrar::registerOperations();
			CpuResidualOpRegistrar::registerOperations();
			CpuSoftmaxOpRegistrar::registerOperations();

#ifdef MILA_HAS_CUDA
            Cuda::TokenEmbedding::CudaTokenEmbeddingOpRegistrar::registerOperations();
			Cuda::Lpe::CudaLpeOpRegistrar::registerOperations();
			Cuda::Gelu::CudaGeluOpRegistrar::registerOperations();
            Cuda::Swiglu::CudaSwigluOpRegistrar::registerOperations();
			Cuda::LayerNorm::CudaLayerNormOpRegistrar::registerOperations();
            Cuda::RmsNorm::CudaRmsNormOpRegistrar::registerOperations();
			// DEPRECATED: Cuda::Linear::CudaLinearOpRegistrar::registerOperations();
            Cuda::Rope::CudaRopeOpRegistrar::registerOperations();
			Cuda::MultiHeadAttention::CudaMultiHeadAttentionOpRegistrar::registerOperations();
			Cuda::Gqa::CudaGroupedQueryAttentionOpRegistrar::registerOperations();
			Cuda::Residual::CudaResidualOpRegistrar::registerOperations();
			Cuda::Softmax::CudaSoftmaxOpRegistrar::registerOperations();
#endif

			//CudaMatMulBiasGeluOpRegistrar::registerOperations();
		}

		static inline bool is_initialized_ = false;
	};
}