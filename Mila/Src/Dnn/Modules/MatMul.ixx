module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.MatMul;

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
	 * @brief A class representing a matrix multiplication module.
	 *
	 * @tparam T The data type of the module.
	 */
	export
		template<typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class MatMul : public Module<T, MR> {
	public:
		using TensorPtr = std::shared_ptr<Tensor<T, MR>>;

		/**
		 * @brief Construct a new MatMul object.
		 *
		 * @param name The name of the module.
		 * @param batch_size The batch size.
		 * @param sequence_length The sequence length.
		 * @param channels The number of channels.
		 * @param channels The number of output channels (channels * N)
		 * @param is_training Whether the module is in training mode.
		 */
		MatMul( std::string name, const std::vector<size_t>& input_shape, size_t output_channels, bool is_training = false )
			: name_( name ), input_shape_( input_shape ), output_channels_( output_channels ), is_training_( is_training ) {
			validateAndCreateParameters();
			createOperation();
		}

        /**
        * @brief Get the weight tensor.
        *
        * @return std::shared_ptr<Tensor<float, MR>> The weight tensor.
        */
		std::shared_ptr<Tensor<float, MR>> getWeight() {
			return weight_;
		}

		void setWeight( const Tensor<float, MR>& weight ) {
			if ( weight.shape() != weight_.shape() ) {
				throw std::invalid_argument( "The shape of the new weight tensor must match the current weight tensor." );
			}
			weight_ = weight;
		}

        /**  
        * @brief Get the bias tensor.  
        *  
        * @return std::shared_ptr<Tensor<float, MR>> The bias tensor.  
        */  
		std::shared_ptr<Tensor<float, MR>> getBias() {
			return bias_;
		}

		void setBias( const Tensor<float, MR>& bias ) {
			if ( bias.shape() != bias_.shape() ) {
				throw std::invalid_argument( "The shape of the new bias tensor must match the current weight tensor." );
			}
			bias_ = bias;
		}


		/**
		 * @brief Get the number of parameters.
		 *
		 * @return size_t The number of parameters.
		 */
		size_t parameters() const override {
			return output_channels_ + output_channels_;
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
			auto B = input->shape()[ 0 ];
			auto T = input->shape()[ 1 ];

			auto output = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ B, T, output_channels_ } );
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

		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::string name_{ "MatMul" }; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.

		size_t output_channels_{ 0 }; ///< The number of output channels.

		bool is_training_{ false }; ///< Whether the module is in training mode.

		std::shared_ptr<Tensor<float, MR>> weight_{ nullptr }; // = Tensor<float, MR>( { output_channels_, C_ } ); ///< The weight tensor.
		std::shared_ptr<Tensor<float, MR>> bias_{ nullptr };// = Tensor<float, MR>( { output_channels_ } ); ///< The bias tensor.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_attributes_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<T, MR>> operation_; ///< The operation.

		void validateAndCreateParameters() {
			// TODO: For now, we only support 3D input shapes.
			if ( input_shape_.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have 3 dimensions." );
			}
			
			// The last dimension of the input shape is the number of channels.
			auto input_channels = input_shape_[ 2 ];

			weight_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_, input_channels } );
			bias_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_ } );

			// Initialize the weight tensor. The bias tensor is default initialized to zeros
			initializeWeights();
			
			parameters_.emplace_back( weight_ );
			parameters_.emplace_back( bias_ );
		}
		
		void initializeWeights() {
			xavier<float, MR>( *weight_, input_shape_[ 2 ], output_channels_ );
		}

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::MatMulOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::MatMulOp" );
			}
		}
	};
}