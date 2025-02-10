module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.Encoder;

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

	export
	template<typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class Encoder : public Module<T, MR> {
	public:

		using TensorPtr = std::shared_ptr<Tensor<T, MR>>;
		
		/**
		* @brief Construct a new Encoder object.
		*
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor.
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		Encoder( std::string name, const std::vector<size_t>& input_shape, bool is_training = false )
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
		* @brief Perform the forward pass of the encoder.
		*
		* @param input The input tensor.
		* @return Tensor<float, MR> The output tensor.
		*/
		Tensor<float, MR> forward( const Tensor<float, MR>& input ) {
			// output is (B,T,C). At each position (b,t), a C dimentional vector summarizing token & position
			// input is (B,T) of integers, holding the token ids at each (b,t) position
			auto output = Tensor<float, MR>( input.shape() );
			operation_->forward( input, parameters_, output, output_cache_ );

			return output;
		}

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
		// wte is (V,C) of token embeddings, short for "weight token embeddings"
		std::shared_ptr<Tensor<float, MR>> wte_{ nullptr };

		// wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
		std::shared_ptr<Tensor<float, MR>> wpe_{ nullptr };

		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The Encoder parameters
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars. Not used in this module.

		std::shared_ptr<Dnn::Compute::OperationBase<T, MR>> operation_{ nullptr }; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			auto channels = input_shape_[ 2 ];
			wte_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels, channels } );
			wpe_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels, channels } );

			parameters_.emplace_back( wte_ );
			parameters_.emplace_back( wpe_ );

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::EncoderOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::EncoderOp" );
			}
		}
	};
}