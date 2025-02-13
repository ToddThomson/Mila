module;
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

export module Dnn.Modules.LayerNorm;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn::Modules
{
	using namespace Mila::Dnn::Compute;

	/**
	* @brief Layer Normalization module.
	*
	* @tparam T Data type of the tensor.
	* @tparam MR Memory resource type (CpuMemoryResource or DeviceMemoryResource).
	*/
	export
	template<typename TInput, typename TCompute = TInput, typename MR = CpuMemoryResource> 
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::MemoryResource, MR>
	class LayerNorm : public Module<TInput, TCompute, MR> {
	public:

		//using TensorPtr = std::shared_ptr<Tensor<T, MR>>;

		/**
		* @brief Construct a new LayerNorm object.
		*
		* @param name Name of the module.
		* @param input_shape Shape of the input tensor.
		* @param has_bias Whether the module has a bias tensor. Default is true.
		* @param is_training Whether the module is in training mode. Default is false.
		*/
		LayerNorm( std::string name, const std::vector<size_t>& input_shape, bool has_bias = true, bool is_training = false )
			: name_{ name }, input_shape_{ input_shape }, has_bias_{ has_bias }, is_training_{ is_training } {
			createOperation();
		}

		/**
		* @brief Get the weight tensor.
		*
		* @return TensorPtr Shared pointer to the weight tensor.
		*/
		std::shared_ptr<Tensor<TInput, MR>> getWeight() {
			return weight_;
		}

		/**
		* @brief Get the bias tensor.
		*
		* @return TensorPtr Shared pointer to the bias tensor.
		*/
		std::shared_ptr<Tensor<TInput, MR>> getBias() {
			return bias_;
		}

		/**
		* @brief Get the number of parameters.
		*
		* @return size_t Number of parameters.
		*/
		size_t parameters() const override {
			return weight_->size() + bias_->size();
		}

		/**
		* @brief Get the name of the module.
		*
		* @return std::string Name of the module.
		*/
		std::string name() const override {
			return name_;
		}

		/**
		* @brief Forward pass of the module.
		*
		* @param input Input tensor.
		* @return TensorPtr Output tensor.
		*/
		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) override {
			operation_->forward( input, parameters_, output_, output_cache_ );

			return std::move( output_ );
		}

		/**
		* @brief Print the module information.
		*/
		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

	private:
		std::string name_{ "LayerNorm" }; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.
		float epsilon_{ 1e-05f }; ///< The epsilon value.

		bool has_bias_{ true }; ///< Whether the module has a bias tensor. Default is true.
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		Tensor<TCompute, MR> output_; ///< The output tensor.

		std::shared_ptr<Tensor<float, MR>> weight_{ nullptr }; ///< The weight tensor.
		std::shared_ptr<Tensor<float, MR>> bias_{ nullptr }; ///< The bias tensor.

		std::shared_ptr<Tensor<float, MR>> mean_ = { nullptr }; ///< The mean.
		std::shared_ptr<Tensor<float, MR>> rstd_{ nullptr }; ///< The reciprocal standard deviation.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_cache_; ///< The output attributes.
		std::vector<std::shared_ptr<Tensor<float, MR>>> scalars_; ///< The scalars.

		std::shared_ptr<Dnn::Compute::OperationBase<TInput, TCompute, MR>> operation_; ///< The operation.

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			auto batch_size = input_shape_[ 0 ];
			auto sequence_length = input_shape_[ 1 ];
			auto channels = input_shape_[ 2 ];

			weight_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels }, 1.0f );

			if ( has_bias_ )
				bias_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels } );

			parameters_.emplace_back( weight_ );
			parameters_.emplace_back( bias_ );

			mean_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, sequence_length } );
			rstd_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, sequence_length } );

			output_cache_.emplace_back( mean_ );
			output_cache_.emplace_back( rstd_ );

			// Create the output tensor. The output tensor has the same shape as the input tensor.
			output_ = Tensor<TCompute, MR>( input_shape_ );

			// TODO: tensor scalars
			//scalars_[ scalar_names_::EPSILON ] = std::make_shared<Tensor>( epsilon_ );*/

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, float, MR>::instance().createOperation( DeviceType::Cpu, "Cpu::LayerNormOp" );
			}
			else {
				operation_ = OperationRegistry<float, float, MR>::instance().createOperation( DeviceType::Cuda, "Cuda::LayerNormOp" );
			}
		}
	};
}