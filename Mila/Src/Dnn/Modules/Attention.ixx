module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.Attention;

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
	class Attention : public Module<T, MR> {
	public:

		using TensorPtr = std::shared_ptr<Tensor<T, MR>>;

		Attention( std::string name, const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
			: name_{ name }, input_shape_{ input_shape }, num_heads_{ num_heads }, is_training_{ is_training } {
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

		Tensor<float, MR> forward( const Tensor<float, MR>& input ) override {
			auto output = Tensor<float, MR>( input.shape() );
			operation_->forward( input, parameters_, output, output_cache_ );

			return output;
		}

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
		size_t num_heads_{ 0 };
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::shared_ptr<Tensor<float, MR>> attention_ = { nullptr };
		std::shared_ptr<Tensor<float, MR>> pre_attention_{ nullptr };

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<T, MR>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			auto batch_size = input_shape_[ 0 ];
			auto sequence_length = input_shape_[ 1 ];

			// preatt, att are( B, NH, T, T ).NH = number of heads, T = sequence length
			attention_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );
			pre_attention_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );

			output_cache_.emplace_back( attention_ );
			output_cache_.emplace_back( pre_attention_ );

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::AttentionOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::AttentionOp" );
			}
		}
	};
}