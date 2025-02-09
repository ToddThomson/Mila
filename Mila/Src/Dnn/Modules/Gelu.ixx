module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.Gelu;

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
	 * @brief A class representing a linear module.
	 * The module performs the following operation:
	 * output = input * weight + bias
	 *
	 * @tparam T The data type of the module.
	 */
	export
		template<typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class Gelu : public Module<T, MR> {
	public:

		using TensorPtr = std::shared_ptr<Tensor<T, MR>>;

		/**
		* @brief Construct a new Linear object.
		*
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor.
		* @param output_channels The number of output channels/features.
		* @param has_bias Whether the module has a bias tensor.
		* @param is_training Whether the module is in training mode.
		*/
		Gelu( std::string name, const std::vector<size_t>& input_shape, bool is_training = false )
			: name_( name ), input_shape_( input_shape ), is_training_( is_training ) {
			validateAndCreateParameters();
			createOperation();
		}


		/**
		* @brief Get the number of parameters.
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
		std::shared_ptr<Tensor<float, MR>> forward( const std::shared_ptr<Tensor<float, MR>> input ) {
			auto output = std::make_shared<Tensor<float, MR>>( std::vector<size_t>( input->shape() ) );
			operation_->forward( input, parameters_, output, output_attributes_ );

			return output;
		}

		/**
		 * @brief Print the module information.
		 */
		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.

		size_t output_channels_{ 0 }; ///< The number of output channels.

		
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_attributes_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<T, MR>> operation_{ nullptr }; ///< The operation.

		void validateAndCreateParameters() {
			// TODO: For now, we only support 3D input shapes.
			if ( input_shape_.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have 3 dimensions." );
			}

			// The last dimension of the input shape is the number of input channels/features.
			auto input_channels = input_shape_.back();
		}

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::GeluOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::GeluOp" );
			}
		}
	};
}