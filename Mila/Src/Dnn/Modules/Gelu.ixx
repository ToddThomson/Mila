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

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameterTensors() const override {
			return parameters_;
		}

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getStateTensors() const override {
			return output_state_;
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
			for ( const auto& tensor :getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& tensor : getParameterTensors() ) {
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