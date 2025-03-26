module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
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
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;
	
	export
	template<typename TInput, typename TCompute = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TCompute>
	class Gelu : public Module<TInput, TCompute, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;
		
		Gelu( std::string name, bool is_training = false )
		{
			this->setTraining( is_training );
			this->setName( name );
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

		/**
		 * @brief Perform the forward pass.
		 *
		 * @param input The input tensor.
		 * @return The output tensor.
		 */
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) {
			operation_->forward( input, parameters_, properties_,  output, output_state_ );
		}

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
        * @brief Convert the module information to a string.
        */
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "Gelu: " << this->getName() << std::endl;

			return oss.str();
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{ nullptr }; ///< The output cache. Not used in this module.
		OperationAttributes properties_; ///< The operation properties.

		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TCompute, TDeviceType>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cpu>::instance().createOperation( DeviceType::Cpu, "Cpu::GeluOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, DeviceType::Cpu>>(base_operation);
			}
			else {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cuda>::instance().createOperation( DeviceType::Cuda, "Cuda::GeluOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, DeviceType::Cuda>>(base_operation);
			}
		}
	};
}