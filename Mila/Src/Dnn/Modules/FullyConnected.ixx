/**
 * @file FullyConnected.ixx
 * @brief Implementation of the Fully Connected (Linear) module for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.FullyConnected;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	* @brief A class representing a fully-connected module.
	*
	* The fully-connected module (also known as a linear or dense layer) performs
	* a linear transformation of the input data. The operation is defined as:
	* output = input * weight + bias
	*
	* This is a fundamental building block in neural networks that connects
	* every input neuron to every output neuron with learnable weights.
	*
	* @tparam TInput The data type of the input tensor elements.
	* @tparam TPrecision The data type used for computation and output (defaults to the input type).
	* @tparam TDeviceType The device type where the computation will be performed (CPU or CUDA).
	*/
	export
		template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class FullyConnected : public Module<TInput, TPrecision, TDeviceType> {
	public:
		/**
		 * @brief Memory resource type based on the device type.
		 *
		 * This alias resolves to either DeviceMemoryResource for CUDA devices
		 * or HostMemoryResource for CPU devices.
		 */
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		 * @brief Constructs a new FullyConnected module.
		 *
		 * @param name The name of the module for identification purposes.
		 * @param input_channels The number of input features/channels.
		 * @param output_channels The number of output features/channels.
		 * @param has_bias Whether to include a bias term in the transformation (defaults to true).
		 * @param is_training Whether the module is initially in training mode (defaults to false).
		 */
		FullyConnected(
			std::string name,
			size_t input_channels,
			size_t output_channels,
			bool has_bias = true,
			bool is_training = false )
			: input_channels_{ input_channels }, output_channels_{ output_channels }, has_bias_{ has_bias } {
			this->setTraining( is_training );
			this->setName( name );
			initializeTensors();
			createOperation();
		}

		/**
		 * @brief Gets the number of trainable parameters in this module.
		 *
		 * Counts the total number of trainable parameters, which includes
		 * the weight tensor and, if present, the bias tensor.
		 *
		 * @return size_t The total number of parameters.
		 */
		size_t parameterCount() const override {
			size_t num_params = weight_->size();
			if ( has_bias_ ) {
				num_params += bias_->size();
			}
			return num_params;
		}

		/**
		 * @brief Gets the weight tensor used for the linear transformation.
		 *
		 * The weight tensor is used in the matrix multiplication operation:
		 * output = input * weight + bias
		 *
		 * @return std::shared_ptr<Tensor<float, MR>> Shared pointer to the weight tensor.
		 */
		std::shared_ptr<Tensor<TPrecision, MR>> getWeight() {
			return weight_;
		}

		/**
		 * @brief Gets the bias tensor used after the linear transformation.
		 *
		 * The bias tensor is added after the matrix multiplication with weights.
		 * If the module was configured without bias, returns std::nullopt.
		 *
		 * @return std::optional<std::shared_ptr<Tensor<float, MR>>> Optional containing
		 *         the bias tensor if available, otherwise std::nullopt.
		 */
		std::optional<std::shared_ptr<Tensor<TPrecision, MR>>> getBias() {
			if ( !has_bias_ ) {
				return std::nullopt;
			}
			return bias_;
		}

		/**
		 * @brief Gets whether the module has a bias tensor.
		 *
		 * @return bool True if the module has a bias tensor, false otherwise.
		 */
		bool hasBias() const {
			return has_bias_;
		}

		/**
		 * @brief Performs the forward pass of the FullyConnected operation.
		 *
		 * Applies the linear transformation to the input tensor:
		 * output = input * weight + bias
		 *
		 * @param input The input tensor to be transformed.
		 * @param output The output tensor where the results will be stored.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TPrecision, MR>& output ) {
			operation_->forward( input, parameters_, properties_, output, output_state_ );
		}

		/**
		 * @brief Saves the module state to a ZIP archive.
		 *
		 * Serializes the trainable parameters (weight, bias) to the provided archive.
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
		 * Deserializes the trainable parameters (weight, bias) from the provided archive.
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
		 * Includes detailed information about the module configuration and parameters.
		 *
		 * @return std::string A string representation of the module information.
		 */
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "FullyConnected: " << this->getName();
			oss << ", Input channels: " << input_channels_;
			oss << ", Output channels: " << output_channels_ << std::endl;
			oss << this->parametersToString();

			return oss.str();
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		/**
		 * @brief The number of input features/channels.
		 */
		size_t input_channels_{ 0 };

		/**
		 * @brief The number of output features/channels.
		 */
		size_t output_channels_{ 0 };

		/**
		 * @brief Whether the module has a bias tensor. Default is true.
		 */
		bool has_bias_{ true };

		/**
		 * @brief The weight tensor for the linear transformation.
		 *
		 * Shape is [output_channels_, input_channels_].
		 */
		std::shared_ptr<Tensor<TPrecision, MR>> weight_{ nullptr };

		/**
		 * @brief The bias tensor added after the matrix multiplication.
		 *
		 * Shape is [output_channels_].
		 */
		std::shared_ptr<Tensor<TPrecision, MR>> bias_{ nullptr };

		/**
		 * @brief The trainable parameters for this module (weight and bias).
		 */
		std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_ = {};

		/**
		 * @brief Cache of intermediate tensors needed for backward pass.
		 */
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_ = {};

		/**
		 * @brief The operation attributes, may include additional configuration.
		 */
		OperationAttributes properties_;

		/**
		 * @brief The underlying operation that implements the FullyConnected transformation.
		 */
		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TPrecision, TDeviceType>> operation_{ nullptr };

		/**
		 * @brief Initializes the tensors needed for the FullyConnected operation.
		 *
		 * Creates and initializes:
		 * - weight tensor (initialized with Xavier/Glorot distribution)
		 * - bias tensor (initialized to zeros if has_bias_ is true)
		 */
		void initializeTensors() {
			// Initialize the weight tensor using xavier distribution
			weight_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_, input_channels_ } );
			weight_->setName( this->getName() + ".weight" );
			xavier<float, MR>( *weight_, input_channels_, output_channels_ );
			parameters_.emplace_back( weight_ );
			this->parameter_map_[ "weight" ] = weight_;

			if ( has_bias_ ) {
				// Initialize the bias tensor with zeros
				bias_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_ } );
				bias_->setName( this->getName() + ".bias" );
				parameters_.emplace_back( bias_ );
				this->parameter_map_[ "bias" ] = bias_;
			}
		}

		/**
		 * @brief Creates the appropriate FullyConnected operation based on device type.
		 *
		 * This method initializes the operation_ member with the appropriate implementation
		 * of FullyConnected for either CPU or CUDA, as determined by the TDeviceType
		 * template parameter.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::FullyConnectedOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cpu>>(base_op);
			}
			else {
				auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::FullyConnectedOp" );
				operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, DeviceType::Cuda>>(base_op);
			}
		}
	};
}
