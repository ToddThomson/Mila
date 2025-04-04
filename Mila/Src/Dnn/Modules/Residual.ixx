/**
 * @file Residual.ixx
 * @brief Implementation of the Residual connection module for neural networks.
 */

module;
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Residual;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	 * @brief Residual connection module for neural networks.
	 *
	 * The Residual module implements a skip connection that adds the input tensor
	 * to the output of some function, helping mitigate the vanishing gradient problem
	 * in deep neural networks. This is a key component in many modern architectures
	 * including ResNet and Transformers.
	 *
	 * The operation performed is: output = input_a + input_b
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TPrecision The data type used for computation and output (defaults to the input type).
	 * @tparam TDeviceType The device type where the computation will be performed (CPU or CUDA).
	 */
	export
		template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class Residual : public Module<TInput, TPrecision, TDeviceType> {
	public:
		/**
		 * @brief Memory resource type based on the device type.
		 *
		 * This alias resolves to either DeviceMemoryResource for CUDA devices
		 * or HostMemoryResource for CPU devices.
		 */
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		 * @brief Constructs a new Residual connection module.
		 *
		 * @param name The name of the module for identification purposes.
		 * @param is_training Whether the module is being used in training mode (defaults to false).
		 */
		Residual( std::string name, bool is_training = false ) {
			this->setTraining( is_training );
			this->setName( name );
			createOperation();
		}

		/**
		 * @brief Gets the number of trainable parameters in this module.
		 *
		 * The Residual connection has no trainable parameters.
		 *
		 * @return size_t Always returns 0 for Residual connections.
		 */
		size_t parameterCount() const override {
			return 0;
		}

		/**
		 * @brief Performs the forward pass of the Residual connection.
		 *
		 * Adds the two input tensors element-wise and stores the result in the output tensor.
		 *
		 * @param input_a The first input tensor (typically the skip connection).
		 * @param input_b The second input tensor (typically the function output).
		 * @param output The output tensor where the results will be stored.
		 */
		void forward( const Tensor<TInput, MR>& input_a, const Tensor<TInput, MR>& input_b, Tensor<TInput, MR>& output ) {
			operation_->forward( input_a, input_b, parameters_, attributes_, output, output_state_ );
		}

		/**
		 * @brief Saves the module state to a ZIP archive.
		 *
		 * Since Residual has no trainable parameters, this function is mostly a placeholder
		 * for the module interface.
		 *
		 * @param zip The ZIP archive to save the module state to.
		 */
		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		/**
		 * @brief Loads the module state from a ZIP archive.
		 *
		 * Since Residual has no trainable parameters, this function is mostly a placeholder
		 * for the module interface.
		 *
		 * @param zip The ZIP archive to load the module state from.
		 */
		void load( mz_zip_archive& zip ) override {
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
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
			oss << "Residual: " << this->getName() << std::endl;

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
		 * The Residual connection has no parameters, so this is initialized to nullptr.
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
		 * @brief The operation attributes.
		 *
		 * Additional attributes that might be needed for the operation.
		 */
		OperationAttributes attributes_;

		/**
		 * @brief The underlying binary operation that implements the Residual connection.
		 *
		 * For residual connections, this operation performs element-wise addition
		 * between the two input tensors.
		 */
		std::shared_ptr<Dnn::Compute::BinaryOperation<TInput, TPrecision, TDeviceType>> operation_{ nullptr };

		/**
		 * @brief Creates the appropriate Residual operation based on the device type.
		 *
		 * This method initializes the operation_ member with the appropriate implementation
		 * of the Residual operation for either CPU or CUDA, as determined by the TDeviceType
		 * template parameter.
		 *
		 * The operation performs element-wise addition of the two input tensors.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::ResidualOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::BinaryOperation<TInput, TPrecision, DeviceType::Cpu>>(base_op);
			}
			else {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::ResidualOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::BinaryOperation<TInput, TPrecision, DeviceType::Cuda>>(base_op);
			}
		}
	};
}
