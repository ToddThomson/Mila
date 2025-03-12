module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <ostream>
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
    class MultiHeadAttention : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;

		MultiHeadAttention( std::string name, const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
			: input_shape_{ input_shape }, num_heads_{ num_heads } {
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

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameters() const override {
			return parameters_;
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output) override {
			operation_->forward( input, parameters_, output, output_cache_ );
		}

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

		void print() const override {
			std::cout << "Module: " << this->getName() << std::endl;
			std::cout << "Parameter count: " << parameterCount() << std::endl;
		}

		// TODO: Implement the backward pass.
		// 
		//void backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs) const override {
		//    operation_->backward(grad_outputs, grad_inputs);
		//}

	private:
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_{ 0 };

		std::shared_ptr<Tensor<float, MR>> attention_ = { nullptr };
		std::shared_ptr<Tensor<float, MR>> pre_attention_{ nullptr };

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, TDevice>> operation_{ nullptr }; ///< The operation.

		/**
		 * @brief Create the operation.
		 */
		void createOperation() {
			auto batch_size = input_shape_[ 0 ];
			auto sequence_length = input_shape_[ 1 ];

			// preatt, att are( B, NH, T, T ). NH = number of heads, T = sequence length
			attention_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );
			pre_attention_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, num_heads_, sequence_length, sequence_length } );

			output_cache_.emplace_back( attention_ );
			output_cache_.emplace_back( pre_attention_ );

			if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
				operation_ = OperationRegistry<float, float, CpuDevice>::instance().createOperation( DeviceType::Cpu, "Cpu::AttentionOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, CudaDevice>::instance().createOperation( DeviceType::Cuda, "Cuda::AttentionOp" );
			}
		}
	};
}