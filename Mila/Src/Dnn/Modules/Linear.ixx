module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Linear;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	 * @brief A class representing a linear or fully-connected module.
	 * The module performs the following operation:
	 * output = input * weight + bias
	 *
	 * @tparam T The data type of the module.
	 */
	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Linear : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;
		
		Linear(
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
		* @brief Get the number of parameters.
		*
		* @return size_t The number of parameters.
		*/
		size_t parameterCount() const override {
			size_t num_params = weight_->size();
			if ( has_bias_ ) {
				num_params += bias_->size();
			}
			return num_params;
		}

		/**
		 * @brief Perform the forward pass.
		 *
		 * @param input The input tensor.
		 * @return std::shared_ptr<Tensor<float>> The output tensor.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) {
			operation_->forward( input, parameters_, output, output_state_ );
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& [name,tensor] : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				// Load tensor data from zip archive
			}
		}

		/**
		* @brief Convert the module information to string.
		*
		* @return std::string Module information as string.
		*/
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "Linear: " << this->getName();
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
		size_t input_channels_{ 0 }; ///< The number of input channels.
		size_t output_channels_{ 0 }; ///< The number of output channels.
		bool has_bias_{ true }; ///< Whether the module has a bias tensor. Default is true.

		std::shared_ptr<Tensor<float, MR>> weight_{ nullptr };  ///< The weight tensor.
		std::shared_ptr<Tensor<float, MR>> bias_{ nullptr }; ///< The bias tensor.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_ = {}; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_ = {}; ///< The output state.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_ = {}; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, TDevice>> operation_{ nullptr }; ///< The operation.

        /**
        * @brief Validate the input shape and create the weight and bias parameter tensors.
        *
        * This function checks if the input shape has 3 dimensions. If the input shape is valid,
        * it creates the weight tensor with dimensions [output_channels, input_channels] and the
        * bias tensor with dimensions [output_channels] if the module has a bias. The weight tensor
        * is initialized using the Xavier initialization method.
        *
        * @throws std::invalid_argument if the input shape does not have 3 dimensions.
        */
		void initializeTensors() {
			// Initialize the weight tensor using xavier distribution and the bias tensor is default initialized to zeros
			weight_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_, input_channels_ } );
			weight_->setName( this->getName() + ".weight" );
			xavier<float, MR>( *weight_, input_channels_, output_channels_ );
			parameters_.emplace_back( weight_ );
			this->parameter_map_[ "weight" ] = weight_;

			if ( has_bias_ ) {
				bias_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ output_channels_ } );
				bias_->setName( this->getName() + ".bias" );
				parameters_.emplace_back( bias_ );
				this->parameter_map_[ "bias" ] = bias_;
			}
		}
		
		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				operation_ = OperationRegistry<float, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::MatMulOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::MatMulOp" );
			}
		}
	};
}