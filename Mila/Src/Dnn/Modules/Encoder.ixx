module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>

export module Dnn.Modules.Encoder;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaDevice;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute>&& std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Encoder : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;

		/**
		* @brief Construct a new Encoder object.
		*
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor. The input shape is (B,T) of integers, holding
		* the token ids at each (b,t) position
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		Encoder( std::string name, size_t channels, size_t max_seq_len, size_t vocab_len, bool is_training = false )
			: name_{ name }, channels_{ channels }, max_seq_len_{ max_seq_len }, vocab_len_{ vocab_len }, is_training_{ is_training } {
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
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) override {
			operation_->forward( input, parameters_, output, output_cache_ );
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& [name, tensor] : this->named_parameters_ ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& [name, tensor] : this->named_parameters_ ) {
				// Load tensor data from zip archive
			}
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
		//std::vector<size_t> input_shape_; ///< The input shape.
		size_t channels_; ///< The number of channels.
		size_t max_seq_len_; ///< The maximum sequence length.
		size_t vocab_len_; ///< The length of the vocabulary.

		// wte is (V,C) of token embeddings, short for "weight token embeddings"
		std::shared_ptr<Tensor<float, MR>> wte_{ nullptr };

		// wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
		std::shared_ptr<Tensor<float, MR>> wpe_{ nullptr };

		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The Encoder parameters
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars. Not used in this module.

		//Tensor<float, MR> output_; ///< The output tensor.

		std::shared_ptr<Dnn::Compute::OperationBase<int, float, TDevice>> operation_{ nullptr }; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			wte_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ vocab_len_, channels_ } );
			wpe_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ max_seq_len_, channels_ } );

			parameters_.emplace_back( wte_ );
			parameters_.emplace_back( wpe_ );

			// REVIEW: I haven't decided on creating the output on the first call to the forward pass...
			// output is (B,T,C). At each position (b,t), a C dimensional vector summarizing token & position
			//auto B = input_shape_[ 0 ];
			//auto T = input_shape_[ 1 ];

			//output_ = Tensor<float, MR>( std::vector<size_t>( { B, T, channels_ } ) );

			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				operation_ = OperationRegistry<int, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::EncoderOp" );
			}
			else {
				operation_ = OperationRegistry<int, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::EncoderOp" );
			}
		}
	};
}