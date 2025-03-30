/**
 * @file Gelu.ixx
 * @brief Implementation of the Gaussian Error Linear Unit (GELU) activation function module.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Gelu;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	 * @brief Gaussian Error Linear Unit (GELU) activation function module.
	 *
	 * GELU is an activation function defined as:
	 * GELU(x) = x * ?(x)
	 * where ?(x) is the standard Gaussian cumulative distribution function.
	 *
	 * This activation function is used in many state-of-the-art neural network
	 * architectures, including transformers, as an alternative to ReLU.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TPrecision The data type used for computation and output (defaults to the input type).
	 * @tparam TDeviceType The device type where the computation will be performed (CPU or CUDA).
	 */
	export
		template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class Gelu : public Module<TInput, TPrecision, TDeviceType> {
	public:
		/**
		 * @brief Memory resource type based on the device type.
		 *
		 * This alias resolves to either DeviceMemoryResource for CUDA devices
		 * or HostMemoryResource for CPU devices.
		 */
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		 * @brief Constructs a new Gelu activation module.
		 *
		 * @param name The name of the module for identification purposes.
		 * @param is_training Whether the module is being used in training mode (defaults to false).
		 */
		Gelu( std::string name, bool is_training = false )
		{
			this->setTraining( is_training );
			this->setName( name );
			createOperation();
		}

		/**
		 * @brief Gets the number of trainable parameters in this module.
		 *
		 * The GELU activation function has no trainable parameters.
		 *
		 * @return size_t Always returns 0 for GELU.
		 */
		size_t parameterCount() const override {
			return 0;
		}

		/**
		 * @brief Performs the forward pass of the GELU activation function.
		 *
		 * Applies the GELU activation function element-wise to the input tensor.
		 *
		 * @param input The input tensor to apply the activation function to.
		 * @param output The output tensor where the results will be stored.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TPrecision, MR>& output ) {
			operation_->forward( input, parameters_, properties_, output, output_state_ );
		}

		/**
		 * @brief Saves the module state to a ZIP archive.
		 *
		 * Since GELU has no trainable parameters, this function is mostly a placeholder
		 * for the module interface.
		 *
		 * @param zip The ZIP archive to save the module state to.
		 */
		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& tensor : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		/**
		 * @brief Loads the module state from a ZIP archive.
		 *
		 * Since GELU has no trainable parameters, this function is mostly a placeholder
		 * for the module interface.
		 *
		 * @param zip The ZIP archive to load the module state from.
		 */
		void load( mz_zip_archive& zip ) override {
			for ( const auto& tensor : this->getParameterTensors() ) {
				// Load tensor data from zip archive
			}
		}

		/**
		 * @brief Converts the module information to a human-readable string.
		 *
		 * @return std::string A string representation of the module information.
		 */
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "Gelu: " << this->getName() << std::endl;

			return oss.str();
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		/**
		 * @brief The parameters for the operation.
		 *
		 * The GELU activation has no parameters, so this is initialized to nullptr.
		 */
		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr };

		/**
		 * @brief The output cache.
		 *
		 * Storage for intermediate results that might be needed for the backward pass.
		 * Not used in this module, so initialized to nullptr.
		 */
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{ nullptr };

		/**
		 * @brief The operation properties.
		 *
		 * Additional attributes that might be needed for the operation.
		 */
		OperationAttributes properties_;

		/**
		 * @brief The underlying unary operation that implements the GELU function.
		 */
		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TPrecision, TDeviceType>> operation_{ nullptr };

		/**
		 * @brief Creates the appropriate GELU operation based on the device type.
		 *
		 * This method initializes the operation_ member with the appropriate implementation
		 * of the GELU operation for either CPU or CUDA, as determined by the TDeviceType template parameter.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::GeluOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cpu>>(base_op);
			}
			else {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::GeluOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cuda>>(base_op);
			}
		}
	};
}