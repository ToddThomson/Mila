export module Compute.Operations;

import Compute.CpuOperations;
import Compute.CudaOperations;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Singleton class to manage and initialize compute operations.
	*/
	export class Operations {
	public:
		/**
		* @brief Get the singleton instance of Operations.
		* 
		* @return Operations& Reference to the singleton instance.
		*/
		static Operations& instance() {
			static Operations instance;

			// Lazy initialization of operations
			if (!is_initialized_) {
				initializeOperations();
				is_initialized_ = true;
			}

			return instance;
		}

		// Delete copy constructor
		Operations(const Operations&) = delete;
		// Delete copy assignment operator
		Operations& operator=(const Operations&) = delete;

	private:
		// Default constructor
		Operations() = default;

		/**
		* @brief Initialize the compute operations.
		*/
		static void initializeOperations() {
			// CPU operations...
			CpuAttentionOp<float>::registerOperation();
			CpuEncoderOp::registerOperation();
			CpuGeluOp<float>::registerOperation();
			CpuLayerNormOp::registerOperation();
			CpuMatMulOp<float>::registerOperation();
			CpuResidualOp<float>::registerOperation();
			CpuSoftmaxOp<float>::registerOperation();

			// CUDA operations...
			CudaGeluOp<float>::registerOperation();
			CudaMatMulOp<float>::registerOperation();
		}

		// Flag to check if operations are initialized
		static inline bool is_initialized_ = false;
	};
}