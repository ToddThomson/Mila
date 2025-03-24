module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

export module Dnn.Modules.LayerNorm;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

	/**
	* @brief Layer Normalization module.
	*
	* @tparam T Data type of the tensor.
	* @tparam TDevice Memory resource type (CpuMemoryResource or DeviceMemoryResource).
	*/
	export
	template<typename TInput, typename TCompute = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TCompute>
	class LayerNorm : public Module<TInput, TCompute, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;

        /**
        * @brief Construct a new LayerNorm object.
        *
        * @param name Name of the module.
        * @param input_shape Shape of the input tensor.
        * @param axis Axis for normalization. Default is -1.
        * @param has_bias Whether the module has a bias tensor. Default is true.
        * @param is_training Whether the module is in training mode. Default is false.
        */
		LayerNorm(
			std::string name,
			const std::vector<size_t>& input_shape,
			int64_t axis = -1,
			bool has_bias = true,
			bool is_training = false )
			: input_shape_{ input_shape }, axis_{ axis }, has_bias_{ has_bias } {
			this->setTraining( is_training );
			this->setName( name );
			initializeTensors();
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
		size_t parameterCount() const override {
			return weight_->size() + bias_->size();
		}

		/**
		* @brief Forward pass of the module.
		*
		* @param input Input tensor.
		* @return TensorPtr Output tensor.
		*/
		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) {
			operation_->forward( input, parameters_, properties_, output, output_state_ );
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the parameters
			for ( const auto& [name,tensor] : this->getParameterTensors() ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& [name,tensor] : this->getParameterTensors() ) {
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
			oss << "LayerNorm: " << this->getName();
			oss << ", Normalization Axis: " << axis_;
			oss << ", Input shape: (";
			for ( size_t i = 0; i < input_shape_.size(); ++i ) {
				oss << input_shape_[ i ];
				if ( i != input_shape_.size() - 1 ) {
					oss << ",";
				}
			}
			oss << ")" << std::endl;

			oss << "Parameter Tensors..."  << std::endl;
			for ( const auto& [name, tensor] : this->getParameterTensors() ) {
				oss << tensor->toString();
			}
			oss << "Parameter count: " << parameterCount() << std::endl;

			oss << "State Tensors..." << std::endl;
			for ( const auto& [name, tensor] : this->getStateTensors() ) {
				oss << tensor->toString();
			}

			return oss.str();
		}

	private:
		std::vector<size_t> input_shape_; ///< The normalized shape.
		float epsilon_{ 1e-05f }; ///< The epsilon value.
		int64_t axis_{ -1 }; ///< The axis for normalization. Default is -1 for last dimension.
		bool has_bias_{ true }; ///< Whether the module has a bias tensor. Default is true.

		std::shared_ptr<Tensor<float, MR>> weight_{ nullptr }; ///< The weight tensor.
		std::shared_ptr<Tensor<float, MR>> bias_{ nullptr }; ///< The bias tensor.

		std::shared_ptr<Tensor<float, MR>> mean_ = { nullptr }; ///< The mean.
		std::shared_ptr<Tensor<float, MR>> rstd_{ nullptr }; ///< The reciprocal standard deviation.

		std::vector<std::shared_ptr<Tensor<float, MR>>> parameters_; ///< The parameters.
		std::vector<std::shared_ptr<Tensor<float, MR>>> output_state_; ///< The output attributes.
		OperationAttributes properties_; ///< The operation properties.

		std::shared_ptr<Dnn::Compute::UnaryOperation<TInput, TCompute, TDeviceType>> operation_; ///< The operation.

		void initializeTensors() {
			auto batch_size = input_shape_[ 0 ];
			auto sequence_length = input_shape_[ 1 ];
			auto channels = input_shape_[ 2 ];

			weight_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels }, 1.0f );
			weight_->setName( this->getName() + ".weight");
			parameters_.emplace_back( weight_ );
			this->parameter_map_[ "weight" ] = weight_;

			if ( has_bias_ ) {
				bias_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ channels } );
				bias_->setName( this->getName() + ".bias");
				parameters_.emplace_back( bias_ );
				this->parameter_map_[ "bias" ] = bias_;
			}

			mean_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, sequence_length } );
			mean_->setName( this->getName() + ".mean" );
			rstd_ = std::make_shared<Tensor<float, MR>>( std::vector<size_t>{ batch_size, sequence_length } );
			rstd_->setName( this->getName() + ".rstd" );

			output_state_.emplace_back( mean_ );
			output_state_.emplace_back( rstd_ );

			this->state_map_[ "mean" ] = mean_;
			this->state_map_[ "rstd" ] = rstd_;
		}

		/**
		* @brief Create the operation.
		*/
		void createOperation() {
			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cpu>::instance().createOperation( DeviceType::Cpu, "Cpu::LayerNormOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, DeviceType::Cpu>>(base_operation);
			}
			else {
				auto base_operation = OperationRegistry<float, float, DeviceType::Cuda>::instance().createOperation( DeviceType::Cuda, "Cuda::LayerNormOp" );
				operation_ = std::dynamic_pointer_cast<Dnn::Compute::UnaryOperation<float, float, DeviceType::Cuda>>(base_operation);
			}
		}
	};
}