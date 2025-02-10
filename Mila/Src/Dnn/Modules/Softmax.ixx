module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.Softmax;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

export namespace Mila::Dnn::Modules
{
	using namespace Mila::Dnn::Compute;

	/**
	* @brief Softmax module for neural networks.
	* 
	* This class implements the softmax function, which is often used in the final layer of a neural network
	* to convert raw scores into probabilities.
	* 
	* @tparam T The data type of the tensor elements.
	* @tparam MR The memory resource type, either CpuMemoryResource or DeviceMemoryResource.
	*/
	export
	template<typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class Softmax : public Module<T, MR> {
	public:

		using TensorPtr = std::shared_ptr<Tensor<T, MR>>;

		/**
		* @brief Construct a new Softmax module.
		* 
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor.
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		Softmax( std::string name, const std::vector<size_t>& input_shape, bool is_training = false )
			: name_( name ), input_shape_( input_shape ), is_training_( is_training ) {
			createOperation();
		}

		/**
		* @brief Get the number of parameters in the module.
		* 
		* @return size_t The number of parameters.
		*/
		size_t parameters() const override {
			return 0;
		}

		/**
		* @brief Get the name of the module.
		*
		* @return std::string The name of the module.
		*/
		std::string name() const override {
			return name_;
		}

		/**
		* @brief Perform the forward pass.
		*
		* @param input The input tensor.
		* @return std::shared_ptr<Tensor<float>> The output tensor.
		*/
		Tensor<float, MR> forward( const Tensor<float, MR>& input ) {
			auto output = Tensor<float, MR>( input.shape() );
			operation_->forward( input, parameters_, output, output_cache_ );

			return output;
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

		/**
		* @brief Print the module information.
		*/
		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

	private:
		std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.

		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars.module;

		std::shared_ptr<Dnn::Compute::OperationBase<T, MR>> operation_{ nullptr }; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::SoftmaxOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::SoftmaxOp" );
			}
		}
	};
}