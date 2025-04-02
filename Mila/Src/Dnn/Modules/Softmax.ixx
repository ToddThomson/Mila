/**
 * @file Softmax.ixx
 * @brief Implementation of the Softmax activation function module.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.Softmax;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.DeviceType;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationAttributes;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	 * @brief Softmax module for neural networks.
	 *
	 * This class implements the softmax function, which is often used in the final layer of a neural network
	 * to convert raw scores into probabilities. The softmax operation normalizes the input values by applying:
	 *
	 * softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
	 *
	 * where the sum is computed over the specified axis. This normalization ensures all values sum to 1,
	 * allowing them to be interpreted as probabilities for classification tasks.
	 *
	 * @tparam TInput The data type of the input tensor elements.
	 * @tparam TPrecision The data type used for internal precision calculations, defaults to TInput.
	 * @tparam TDeviceType The compute device type (Cpu or Cuda), defaults to Cuda.
	 */
	export
		template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class Softmax final : public Module<TInput, TPrecision, TDeviceType> {
	public:
		/**
		 * @brief Memory resource type based on the device type.
		 *
		 * This alias resolves to either DeviceMemoryResource for CUDA devices
		 * or HostMemoryResource for CPU devices.
		 */
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		 * @brief Construct a new Softmax module.
		 *
		 * @param name The name identifier for the module.
		 * @param axis The dimension along which to apply the softmax operation. Default is -1 (last dimension).
		 * @param is_training Whether the module is in training mode. Default is false.
		 */
		Softmax( std::string name, int64_t axis = -1, bool is_training = false )
			: axis_{ axis } {
			this->setTraining( is_training );
			this->setName( name );
			createOperation();
		}

		/**
		 * @brief Get the number of parameters in the module.
		 *
		 * The Softmax module has no trainable parameters.
		 *
		 * @return size_t Always returns 0 as Softmax has no parameters.
		 */
		size_t parameterCount() const override {
			return 0;
		}

		/**
		 * @brief Perform the forward pass of the softmax operation.
		 *
		 * Computes the softmax of the input tensor along the specified axis and writes the result to the output tensor.
		 * The operation exponentiates each element and then normalizes by the sum of all exponentiated values
		 * along the specified axis.
		 *
		 * @param input The input tensor to apply softmax to.
		 * @param output The tensor where softmax results will be stored.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TInput, MR>& output ) {
			operation_->forward( input, parameters_, attributes_, output, output_state_ );
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

		/**
		 * @brief Save the module's state to a zip archive.
		 *
		 * Serializes the module's state to the provided zip archive. Since Softmax has no trainable parameters,
		 * this primarily preserves the module's configuration.
		 *
		 * @param zip The zip archive where the module state will be saved.
		 */
		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& tensor : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		/**
		 * @brief Load the module's state from a zip archive.
		 *
		 * Deserializes the module's state from the provided zip archive. Since Softmax has no trainable parameters,
		 * this primarily restores the module's configuration.
		 *
		 * @param zip The zip archive from which to load the module state.
		 */
		void load( mz_zip_archive& zip ) override {
			for ( const auto& tensor : this->getParameterTensors() ) {
				// Load tensor data from zip archive
			}
		}

		/**
		 * @brief Convert the module information to string.
		 *
		 * Creates a string representation of the Softmax module, including its name,
		 * the dimension used for softmax computation, and the input shape.
		 *
		 * @return std::string A formatted string describing the module configuration.
		 */
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "Softmax: " << this->getName() << ", Dimension: " << axis_ << std::endl;

			return oss.str();
		}

	private:
		/**
		 * @brief The dimension to perform the softmax operation on.
		 *
		 * Default is -1 for the last dimension. Negative values count backward from the end
		 * (-1 means the last dimension, -2 means the second-to-last dimension, etc.).
		 */
		int64_t axis_{ -1 };

		/**
		 * @brief The parameters for the operation.
		 *
		 * The Softmax activation has no parameters, so this is initialized to nullptr.
		 */
		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{};

		/**
		 * @brief The output state.
		 *
		 * Storage for intermediate results that might be needed for the backward pass.
		 * Not actively used in this module, so initialized to nullptr.
		 */
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{};

		/**
		 * @brief The operation attributes.
		 *
		 * Contains the configuration for the softmax operation, such as the axis along which to compute.
		 */
		OperationAttributes attributes_{ axis_ };

		/**
		 * @brief The underlying unary operation that implements the softmax function.
		 */
		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TPrecision, TDeviceType>> operation_{ nullptr };

		/**
		 * @brief Create the appropriate softmax operation based on device type.
		 *
		 * This method initializes the operation_ member with the appropriate implementation
		 * of the softmax operation for either CPU or CUDA, as determined by the TDeviceType template parameter.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::SoftmaxOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cpu>>(base_op);
			}
			else {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::SoftmaxOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cuda>>(base_op);
			}
		}
	};
}
