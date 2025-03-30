module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
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
import Compute.UnaryOperation;
import Compute.OperationAttributes;
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
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TPrecision The data type of the precision tensor elements.
    * @tparam TDeviceType The device type, either Cpu or Cuda.
    */
	export
	template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class Softmax : public Module<TInput, TPrecision, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		* @brief Construct a new Softmax module.
		* 
		* @param name The name of the module.
		* @param input_shape The shape of the input tensor.
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		Softmax( std::string name, int64_t axis = -1, bool is_training = false )
			: axis_{ axis } {
			this->setTraining( is_training );
			this->setName( name );
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

		/**
		* @brief Perform the forward pass.
		*
		* @param input The input tensor.
		* @return std::shared_ptr<Tensor<float>> The output tensor.
		*/
		void forward( const Tensor<TInput, MR>& input, Tensor<TInput, MR>& output ) {
			operation_->forward( input, parameters_, attributes_, output, output_state_ );
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& tensor : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& tensor : this->getParameterTensors() ) {
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
			oss << "Softmax: " << this->getName() << ", Dimension: " << axis_;
			oss << ", Input shape: (";
			for ( size_t i = 0; i < input_shape_.size(); ++i ) {
				oss << input_shape_[ i ];
				if ( i != input_shape_.size() - 1 ) {
					oss << ",";
				}
			}
			oss << ")" << std::endl;

			return oss.str();
		}

	private:
		std::vector<size_t> input_shape_; ///< The input shape.
		int64_t axis_{ -1 }; ///< The dimension to perform the softmax operation on. Default is -1 for the last dimension.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_ = { nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_ = { nullptr }; ///< The output attributes. Not used in this module.
		OperationAttributes attributes_{ .axis = axis_ }; ///< The operation properties.
		
		std::unique_ptr<Dnn::Compute::UnaryOperation<TInput, TPrecision, TDeviceType>> operation_{ nullptr }; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			/*auto base_operation = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType>(
				(TDeviceType == DeviceType::Cpu) ? "Cpu::SoftmaxOp" : "Cuda::SoftmaxOp"
			);
			operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<TInput, TPrecision, TDeviceType>>(base_operation);*/
		}
	};
}