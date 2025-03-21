module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>

export module Dnn.Modules.Encoder;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaDevice;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
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
			: channels_{ channels }, max_seq_len_{ max_seq_len }, vocab_len_{ vocab_len } {
			this->setTraining( is_training );
			this->setName( name );
			initializeTensors();
			createOperation();
		}

		/**
		* @brief Get the number of parameters in the module.
		*
		* @return size_t The number of parameters.
		*/
		size_t parameterCount() const override {
			return wte_->size() + wpe_->size();
		}

		/**
		* @brief Perform the forward pass of the encoder.
		*
		* @param input The input tensor.
		* @return Tensor<float, MR> The output tensor.
		*/
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) {
			operation_->forward( input, parameters_, output, output_state_ );
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				// Load tensor data from zip archive
			}
		}

        /**
        * @brief Get the module information as a string.
        *
        * @return std::string The module information.
        */
		std::string toString() const override {
			std::ostringstream oss;
			oss << "Encoder: " << this->getName();
			oss << ", Channels: " << channels_ << ", Max Sequence Length: " << max_seq_len_;
			oss << ", Vocabulary Length: " << vocab_len_ << std::endl;
			oss << "Parameter Tensors..." << std::endl;
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				oss << tensor->toString();
			}
			oss << "Parameter count: " << parameterCount() << std::endl;

			return oss.str();
		}
		
	private:
		size_t channels_; ///< The number of channels.
		size_t max_seq_len_; ///< The maximum sequence length.
		size_t vocab_len_; ///< The length of the vocabulary.

		// wte is (V,C) of token embeddings, short for "weight token embeddings"
		std::shared_ptr<Tensor<float, MR>> wte_{ nullptr };

		// wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
		std::shared_ptr<Tensor<float, MR>> wpe_{ nullptr };

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The Encoder parameters
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars. Not used in this module.

		std::shared_ptr<Dnn::Compute::UnaryOperation<int, float, TDevice>> operation_{ nullptr }; ///< The operation.

		void initializeTensors() {
			wte_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ vocab_len_, channels_ } );
			wte_->setName( this->getName() + ".wte" );
			xavier<float, MR>( *wte_, vocab_len_, channels_ );
			wpe_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ max_seq_len_, channels_ } );
			xavier<float, MR>( *wpe_, max_seq_len_, channels_ );
			wpe_->setName( this->getName() + ".wpe" );

			parameters_.emplace_back( wte_ );
			parameters_.emplace_back( wpe_ );

			this->parameter_map_[ "wte" ] = wte_;
			this->parameter_map_[ "wpe" ] = wpe_;
		}

		/**
		* @brief Create the operation.
		*/
		void createOperation() {

			// REVIEW: I haven't decided on creating the output on the first call to the forward pass...
			// output is (B,T,C). At each position (b,t), a C dimensional vector summarizing token & position
			//auto B = input_shape_[ 0 ];
			//auto T = input_shape_[ 1 ];

			//output_ = Tensor<float, MR>( std::vector<size_t>( { B, T, channels_ } ) );

			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				auto base_operation = OperationRegistry<int, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::EncoderOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<int, float, CpuDevice>>(base_operation);
			}
			else {
				auto base_operation = OperationRegistry<int, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::EncoderOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<int, float, CudaDevice>>(base_operation);
			}
		}
	};
}