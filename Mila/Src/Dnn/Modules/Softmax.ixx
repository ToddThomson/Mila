module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.Softmax;

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

	/**
	* @brief Softmax module for neural networks.
	* 
	* This class implements the softmax function, which is often used in the final layer of a neural network
	* to convert raw scores into probabilities.
	* 
	* @tparam T The data type of the tensor elements.
	* @tparam MR The memory resource type, either CpuMemoryResource or DeviceMemoryResource.
	*/
	export
		template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute>&& std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Softmax : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;

		/**
		* @brief Construct a new Softmax module.
		* 
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor.
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		Softmax( std::string name, int64_t axis = -1, bool is_training = false )
			: name_( name ), axis_{ axis }, is_training_( is_training ) {
			createOperation();
		}

		/**
		* @brief Get the number of parameters in the module.
		* 
		* @return size_t The number of parameters.
		*/
		size_t parameterCount() const override {
			return 0;
		}

		const std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>>& getSubModules() const override {
			return {};
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
		* @return std::shared_ptr<Tensor<float>> The output tensor.
		*/
		void forward( const Tensor<TInput, MR>& input, Tensor<TInput, MR>& output ) {
			operation_->forward( input, parameters_, output, output_cache_ );
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& tensor : getParameters() ) {
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

	private:
		std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.
		int64_t axis_{ -1 }; ///< The dimension to perform the softmax operation on. Default is -1 for the last dimension.
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		//Tensor<float, MR> output_; ///< The output tensor.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_{ nullptr }; ///< The output attributes. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_{ nullptr }; ///< The scalars.module;

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, TDevice>> operation_{ nullptr }; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				operation_ = OperationRegistry<float, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::SoftmaxOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::SoftmaxOp" );
			}
		}
	};
}