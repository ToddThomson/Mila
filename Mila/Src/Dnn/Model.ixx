module;
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <unordered_set>
#include <string>
#include <iostream>
#include <stdexcept>

export module Dnn.Model;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn
{
	/**
	* @brief A class representing a neural network model.
	*
	* @tparam T The data type used for the model's parameters and computations.
	* @tparam MemoryResource The memory resource type used for memory management.
	*/
	export
	template<typename TInput, typename TCompute = TInput, typename MR = Compute::CpuMemoryResource>
		requires ValidTensorTypes<TInput, TCompute> && ( std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::DeviceMemoryResource> )
	class Model : public Module<TInput, TCompute, MR> {
	public:

		/**
		* @brief Constructs a new Model object.
		*
		* Initializes CUDA stream if the memory resource is a device memory resource.
		*/
		Model() {
			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				cudaStreamCreate( &stream_ );
			}
		}

		/**
		* @brief Destroys the Model object.
		*
		* Destroys CUDA stream if the memory resource is a device memory resource.
		*/
		~Model() {
			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				cudaStreamDestroy( stream_ );
			}
		}

		/**
		* @brief Adds a module to the model.
		*
		* @tparam ModuleType The type of the module to add.
		* @param module The module to add.
		* @return size_t The index of the added module.
		* @throws std::invalid_argument if a module with the same name already exists.
		*/
		template <typename ModuleType> requires std::derived_from<ModuleType, Module<TInput, TCompute, MR>>
		size_t add( std::shared_ptr<ModuleType> module ) {

			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				module->setStream( stream_ );
			}

			std::string name = module->name();
			if ( std::find( module_names_.begin(), module_names_.end(), name ) != module_names_.end() ) {
				throw std::invalid_argument( "Module with name '" + name + "'" + " already exists." );
			}

			modules_.emplace_back( std::move( module ) );
			module_names_.emplace_back( name );

			// Return the index of the added module
			return modules_.size() - 1;
		}

		/**
		* @brief Performs a forward pass through the model.
		*
		* @param input The input tensor.
		* @return std::shared_ptr<Tensor<T>> The output tensor.
		* @throws std::runtime_error if the model has not been built.
		*/
		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) override {
			if ( !is_built_ ) {
				throw std::runtime_error( "Model has not been built. Call build() before forward()." );
			}

			Tensor<TCompute, MR> out = input;
			for ( const auto& module : modules_ ) {
				out = module->forward( out );
			}

			return std::move( out );
		}

		/**
		* @brief Builds the model.
		*
		* Sets the training mode for all modules and performs any necessary graph validation or optimizations.
		* @throws std::runtime_error if the model has already been built.
		*/
		void build() {
			if ( is_built_ ) {
				throw std::runtime_error( "Model has already been built." );
			}

			for ( auto& op : modules_ ) {
				op->setTrainingMode( is_training_ );
			}

			is_built_ = true;
		}

		/**
		* @brief Performs a backward pass through the model.
		*
		* @throws std::runtime_error if the model has not been built.
		*/
		void backward() {
			if ( !is_built_ ) {
				throw std::runtime_error( "Model has not been built. Call build() before backward()." );
			}

			if ( !is_training_ ) return;

			// Backward pass implementation
		}

		/**
		* @brief Sets the training mode for the model.
		*
		* @param training The training mode to set.
		*/
		void setTrainingMode( bool training ) {
			is_training_ = training;
		}

		/**
		* @brief Accesses a module by its index.
		*
		* @param index The index of the module.
		* @return std::shared_ptr<Module<T>> A shared pointer to the module.
		* @throws std::out_of_range if the index is out of range.
		*/
		std::shared_ptr<Module<TInput,TCompute, MR>> operator[]( size_t index ) const {
			if ( index >= modules_.size() ) {
				throw std::out_of_range( "Index out of range" );
			}
			return modules_[ index ];
		}

		/**
		* @brief Accesses a module by its name.
		*
		* @param name The name of the module.
		* @return std::shared_ptr<Module<T>> A shared pointer to the module.
		* @throws std::out_of_range if no module with the given name is found.
		*/
		std::shared_ptr<Module<TInput,TCompute, MR>> operator[]( const std::string& name ) const {
			auto it = std::find( module_names_.begin(), module_names_.end(), name );
			if ( it == module_names_.end() ) {
				throw std::out_of_range( "No module found with name '" + name + "'." );
			}
			size_t index = std::distance( module_names_.begin(), it );
			return modules_[ index ];
		}

		inline std::shared_ptr<Tensor<TCompute, MR>> tensor( const Tensor<TInput, MR>& tensor ) {
			auto tensor_ptr = std::make_shared<Tensor<TInput, MR>>( tensor );
			inputs_.emplace( tensor_ptr );

			return tensor_ptr;
		}

		/**
		* @brief Calculates the total number of parameters in the model.
		*
		* @return size_t The total number of parameters.
		*/
		size_t parameters() const override {
			size_t total_parameters = 0;
			for ( const auto& module : modules_ ) {
				total_parameters += module->parameters();
			}
			return total_parameters;
		}

		/**
		* @brief Returns the number of modules in the model.
		*
		* @return size_t The number of modules.
		*/
		size_t size() const {
			return modules_.size();
		}

		/**
		* @brief Prints the model's structure and total number of parameters.
		*/
		void print() const override {
			std::cout << "Modules: " << std::endl;
			for ( const auto& module : modules_ ) {
				module->print();
			}
			std::cout << "Total parameters: " << parameters() << std::endl;
		}

	private:
		std::vector<std::shared_ptr<Module<TInput, TCompute, MR>>> modules_; ///< The list of modules in the model.
		std::vector<std::string> module_names_; ///< The list of module names.

		std::unordered_set<std::shared_ptr<Tensor<TInput, MR>>> inputs_; ///< The setof input tensors.

		bool is_built_{ false }; ///< Indicates whether the model has been built.
		bool is_training_{ false }; ///< Indicates whether the model is in training mode.

		cudaStream_t stream_{}; ///< The CUDA stream for device memory resource.
	};
}