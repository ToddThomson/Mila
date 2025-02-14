module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <ostream>
#include <type_traits>

export module Dnn.Modules.Residual;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
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
	template<typename TInput, typename TCompute = TInput, typename MR = CpuMemoryResource>
		requires ValidTensorTypes<TInput, TCompute> && ( std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource> )
	class Residual : public Module<TInput, TCompute, MR> {
	public:
		Residual( std::string name, const std::vector<size_t>& input_shape, bool is_training = false )
			: name_( name ), input_shape_( input_shape ), is_training_( is_training ) {
			createOperation();
		}

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
		 * @return Tensor<float,MR> The output tensor.
		 */
		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) {
			operation_->forward( input, parameters_, output_, output_attributes_ );

			return std::move( output_ );
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

		Tensor<TCompute, MR> output_; ///< The output tensor.

		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_attributes_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars. Not used in this module.

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, MR>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			output_ = Tensor<float, MR>( input_shape_ );

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::ResidualOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::ResidualOp" );
			}
		}
	};
}