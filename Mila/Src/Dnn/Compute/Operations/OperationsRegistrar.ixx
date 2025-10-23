module;
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cuda_fp16.h>

export module Compute.OperationsRegistrar;

import Compute.CpuOperations;
import Compute.CudaOperations;

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

			//CpuEncoderOpRegistrar::registerOperations();
			CpuGeluOpRegistrar::registerOperations();
			CpuLinearOpRegistrar::registerOperations();
			//CpuLayerNormOpRegistrar::registerOperations();
			//CpuMultiHeadAttentionOpRegistrar::registerOperations();
			//CpuResidualOpRegistrar::registerOperations();
			CpuSoftmaxOpRegistrar::registerOperations();

			//CudaEncoderOpRegistrar::registerOperations();
			CudaGeluOpRegistrar::registerOperations();
			CudaLinearOpRegistrar::registerOperations();
			//CudaLayerNormOpRegistrar::registerOperations();
			//CudaMultiHeadAttentionOpRegistrar::registerOperations();
			//CudaResidualOpRegistrar::registerOperations();
			CudaSoftmaxOpRegistrar::registerOperations();
			//CudaMatMulBiasGeluOpRegistrar::registerOperations();
		}

		static inline bool is_initialized_ = false;
	};
}