module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Attention;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	export
		template<typename TInput, typename TCompute = TInput, typename TDevice = CpuDevice>
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
    class MultiHeadAttention : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;

		MultiHeadAttention( std::string name, const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
			: input_shape_{ input_shape }, num_heads_{ num_heads } {
			this->setTraining( is_training );
			this->setName( name );
			initializeTensors();
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

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output)  {
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
		* @brief Convert the module information to string.
		*
		* @return std::string Module information as string.
		*/
		std::string toString() const override {
			std::ostringstream oss;
			oss << "--------------------" << std::endl;
			oss << "MultiHeadAttention: " << this->getName();
			oss << ", Number of heads: " << num_heads_;
			oss << ", Input shape: (";
			for ( size_t i = 0; i < input_shape_.size(); ++i ) {
				oss << input_shape_[ i ];
				if ( i != input_shape_.size() - 1 ) {
					oss << ",";
				}
			}
			oss << ")" << std::endl;
			oss << this->stateToString();

			return oss.str();
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_{ 0 };

		std::shared_ptr<Tensor<float, MR>> attn_ = { nullptr };
		std::shared_ptr<Tensor<float, MR>> pre_attn_ = { nullptr };

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_ = {}; ///< The parameters. 
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_ = {}; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_ = {}; ///< The scalars.

		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TCompute, TDevice>> operation_{ nullptr }; ///< The operation.

		void initializeTensors() {
			auto batch_size = input_shape_[ 0 ];
			auto sequence_length = input_shape_[ 1 ];

			// preatt, att are( B, NH, T, T ). NH = number of heads, T = sequence length
			pre_attn_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );
			pre_attn_->setName( this->getName() + ".pre_attn" );

			attn_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );
			attn_->setName( this->getName() + ".attn" );

			output_state_.emplace_back( pre_attn_ );
			output_state_.emplace_back( attn_ );
		}
		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				auto base_operation = OperationRegistry<float, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::AttentionOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, CpuDevice>>(base_operation);
			}
			else {
				auto base_operation = OperationRegistry<float, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::AttentionOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, CudaDevice>>(base_operation);
			}
		}
	};
}