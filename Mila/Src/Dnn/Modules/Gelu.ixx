module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>

export module Dnn.Modules.Gelu;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.ComputeDevice;
import Compute.CpuDevice;

import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;
	
	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute>&& std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Gelu : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;
		
		//Gelu( std::string name, const std::vector<size_t>& input_shape, bool is_training = false )
		//	: name_( name ), input_shape_( input_shape ), is_training_( is_training ) {
		//	createOperation();
		//}

		Gelu( std::string name, /* const std::shared_ptr<Tensor<TInput, MR>>& input, */ bool is_training = false)
			: name_( name ), is_training_( is_training ) {
			createOperation();
		}

		/**
		* @brief Get the number of parameters.
		*
		* @return size_t The number of parameters.
		*/
		size_t parameterCount() const override {
			return 0;
		}

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameters() const override {
			return parameters_;
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
		 * @return The output tensor.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) override {
			operation_->forward( input, parameters_, output, output_state_ );
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& tensor :getParameters() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& tensor : getParameters() ) {
				// Load tensor data from zip archive
			}
		}

		/**
		 * @brief Print the module information.
		 */
		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameter count: " << parameterCount() << std::endl;
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::string name_; ///< The name of the module.
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{ nullptr }; ///< The output cache. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars. Not used in this module.

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, TDevice>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {

			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				operation_ = OperationRegistry<float, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::GeluOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::GeluOp" );
			}
		}
	};
}