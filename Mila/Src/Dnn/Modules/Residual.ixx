module;
//#include <miniz.h>
#include <sstream>
//#include <memory>
//#include <vector>
//#include <string>
//#include <iostream>
//#include <ostream>
#include <type_traits>

export module Dnn.Modules.Residual;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.ComputeDevice;
import Compute.CpuDevice;

import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
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
	class Residual : public Module<TInput, TCompute, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;

		Residual( std::string name, bool is_training = false ) {
			this->setTraining( is_training );
			this->setName( name );
			createOperation();
		}

		size_t parameterCount() const override {
			return 0;
		}

		/**
		 * @brief Perform the forward pass.
		 *
		 * @param input The input tensor.
		 * @return Tensor<float,MR> The output tensor.
		 */
		void forward( const Tensor<TInput, MR>& input_a, const Tensor<TInput, MR>& input_b, Tensor<TInput, MR>& output ) {
			operation_->forward( input_a, input_b, parameters_, attributes_, output, output_state_ );
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
		* @brief Convert the module information to string.
		*
		* @return std::string Module information as string.
		*/
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "Residual: " << this->getName() << std::endl;
			
			return oss.str();
		}


		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_{ nullptr }; ///< The parameters. Not used in this module.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_{ nullptr }; ///< The output attributes. Not used in this module.
		OperationAttributes attributes_; ///< The attributes.

		std::shared_ptr<Dnn::Compute::BinaryOperation<TInput, TCompute, TDeviceType>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cpu>::instance().createOperation( DeviceType::Cpu, "Cpu::ResidualOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::BinaryOperation<float, float, DeviceType::Cpu>>(base_operation);
			}
			else {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cuda>::instance().createOperation( DeviceType::Cuda, "Cuda::ResidualOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::BinaryOperation<float, float, DeviceType::Cuda>>(base_operation);
			}
		}
	};
}