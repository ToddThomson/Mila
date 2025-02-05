module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>

export module Dnn.Modules.MatMul;

import Dnn.Module;
import Dnn.Tensor;
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
		MatMul( std::string name, int64_t batch_size, int64_t sequence_length, int64_t channels, int64_t output_channels, bool is_training = false )
			: name_( name ), B_( batch_size ), T_( sequence_length ), C_( channels ), OC_( output_channels ), is_training_( is_training ) {
			createOperation();
		}

		/**
		 * @brief Get the weight tensor.
		 *
		 * @return Tensor<float>& The weight tensor.
		 */
		Tensor<float, MR>& Weight() {
			return weight_;
		}

		/**
		 * @brief Get the bias tensor.
		 *
		 * @return Tensor<float>& The bias tensor.
		 */
		Tensor<float, MR>& Bias() {
			return bias_;
		}

		/**
		 * @brief Get the number of parameters.
		 *
		 * @return size_t The number of parameters.
		 */
		size_t parameters() const override {
			return OC_ + OC_;
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
			auto output = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ B_, T_, OC_ } );
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
		size_t B_{ 0 }; ///< The batch size.
		size_t T_{ 0 }; ///< The sequence length.
		size_t C_{ 0 }; ///< The number of channels.
		size_t OC_{ 0 }; ///< The number of output channels.

		bool is_training_{ false }; ///< Whether the module is in training mode.

		Tensor<float, MR> weight_ = Tensor<float, MR>( { OC_, C_ } ); ///< The weight tensor.
		Tensor<float, MR> bias_ = Tensor<float, MR>( { OC_ } ); ///< The bias tensor.

		std::vector<std::shared_ptr<Tensor<T,MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<T,MR>>> output_attributes_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<T,MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<T,MR>> operation_; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			parameters_.emplace_back( std::make_shared<Tensor<float,MR>>( weight_ ) );
			parameters_.emplace_back( std::make_shared<Tensor<float,MR>>( bias_ ) );

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float,MR>::instance().createOperation( DeviceType::Cpu, "Cpu::MatMulOp" );
			}
			else {
				operation_ = OperationRegistry<float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::MatMulOp" );
			}
		}
	};
}