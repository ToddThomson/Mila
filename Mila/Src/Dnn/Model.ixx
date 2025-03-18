module;
#include <cuda_runtime.h>
#include "miniz.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <stdexcept>
#include <type_traits>

export module Dnn.Model;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

import Dnn.Modules.LayerNorm;
import Dnn.Modules.FullyConnected;
import Dnn.Modules.Gelu;
import Dnn.Modules.Residual;

namespace Mila::Dnn
{
	/**
	* @brief A class representing a neural network model.
	*
	* @tparam T The data type used for the model's parameters and computations.
	* @tparam MemoryResource The memory resource type used for memory management.
	*/
	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = Compute::CpuDevice>
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Model {
	public:
		using MR = TDevice::MR;

		/**
		* @brief Constructs a new Model object.
		*
		* Initializes CUDA stream if the memory resource is a device memory resource.
		*/
		Model() {
			if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
				cudaStreamCreate( &stream_ );
			}
		}

		/**
		* @brief Destroys the Model object.
		*
		* Destroys CUDA stream if the memory resource is a device memory resource.
		*/
		~Model() {
			if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
				cudaStreamDestroy( stream_ );
			}
		}

		void saveCheckpoint( const std::string& filename ) const {
			mz_zip_archive zip;
			memset( &zip, 0, sizeof( zip ) );
			mz_zip_writer_init_file( &zip, filename.c_str(), 0 );

			for ( const auto& [name, module] : modules_ ) {
				module->save( zip );
			}

			mz_zip_writer_finalize_archive( &zip );
			mz_zip_writer_end( &zip );
		}

		// Load the checkpoint
		void loadCheckpoint( const std::string& filename ) {
			mz_zip_archive zip;
			memset( &zip, 0, sizeof( zip ) );
			mz_zip_reader_init_file( &zip, filename.c_str(), 0 );

			for ( const auto& [name, module] : modules_ ) {
				module->load( zip );
			}

			mz_zip_reader_end( &zip );
		}

		std::shared_ptr<Tensor<TInput, MR>> tensor( std::string name, const Tensor<TInput, MR>& input ) {
			auto tensor = std::make_shared<Tensor<TInput, MR>>( input );
			input_tensor_map_[ tensor->get_uid() ] = tensor;

			return tensor;
		}

		std::shared_ptr<Tensor<TInput, MR>> gelu( std::string name, std::shared_ptr<Tensor<TInput, MR>> input ) {
			/*auto output = std::make_shared<Tensor<TInput, MR>>( std::vector<T>( input->value.size() ) );
			tensor_map[ output->getID() ] = output;*/

			auto node = std::make_shared<Gelu<TInput, TCompute, TDevice>>( input );

			return registerModule( node );
		}

		std::shared_ptr<Tensor<TInput, MR>> layernorm( 
			std::string name, 
			std::shared_ptr<Tensor<TInput, MR>> input,
			std::vector<size_t> normalized_shape ) {

			/*auto output = std::make_shared<Tensor<TInput,MR>>( std::vector<T>() );
			tensor_map[ output->getID() ] = output;*/

			auto node = std::make_shared<LayerNorm<TInput, TCompute, TDevice>>( name, normalized_shape );
			registerModule( node );

			return { nullptr };// node->output();
		}

		std::shared_ptr<Tensor<TInput, MR>> linear( 
			std::string name, std::shared_ptr<Tensor<TInput, MR>> input,
			std::shared_ptr<Tensor<TInput, MR>> weight,
			std::shared_ptr<Tensor<TInput, MR>> bias ) {

			/*auto output = std::make_shared<Tensor<TInput,MR>>( std::vector<T>() );
			tensor_map[ output->getID() ] = output;*/

			auto node = std::make_shared<FullyConnected<TInput, TCompute, TDevice>>( input, weight, bias );
			registerNode( node );

			return { nullptr };// node->output();
		}

		std::shared_ptr<Tensor<TInput, MR>> residual( std::shared_ptr<Tensor<TInput, MR>> input,
			std::shared_ptr<Tensor<TInput, MR>> function_output ) {

			auto output = std::make_shared<Tensor<TInput, MR>>();
			input_tensor_map_[ output->getID() ] = output;

			auto node = std::make_shared<Residual<TInput, TCompute, MR>>( input, function_output, output );
			return registerNode( node );
		}

		/**
		* @brief Adds a module to the model.
		*
		* @tparam ModuleType The type of the module to add.
		* @param module The module to add.
		* @return size_t The index of the added module.
		* @throws std::invalid_argument if a module with the same name already exists.
		*/
		//template <typename ModuleType> requires std::derived_from<ModuleType, Module<TInput, TCompute, MR>>
		//size_t add( std::shared_ptr<ModuleType> module ) {

		//	if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
		//		module->setStream( stream_ );
		//	}

		//	std::string name = module->name();
		//	if ( std::find( module_names_.begin(), module_names_.end(), name ) != module_names_.end() ) {
		//		throw std::invalid_argument( "Module with name '" + name + "'" + " already exists." );
		//	}

		//	modules_.emplace_back( std::move( module ) );
		//	module_names_.emplace_back( name );

		//	// Return the index of the added module
		//	return modules_.size() - 1;
		//}

		// Execute forward pass
		void forward() {
			if ( !is_built_ ) {
				throw std::runtime_error( "Model has not been built. Call build() before forward()." );
			}

			/*for ( auto& node : execution_order ) {
				node->forward( tensor_map );
			}*/
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
		std::shared_ptr<Module<TInput, TCompute, TDevice>> operator[]( size_t index ) const {
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
		std::shared_ptr<Module<TInput, TCompute, TDevice>> operator[]( const std::string& name ) const {
			auto it = std::find( module_names_.begin(), module_names_.end(), name );
			if ( it == module_names_.end() ) {
				throw std::out_of_range( "No module found with name '" + name + "'." );
			}
			size_t index = std::distance( module_names_.begin(), it );
			return modules_[ index ];
		}

		/*inline std::shared_ptr<Tensor<TCompute, MR>> tensor( const Tensor<TInput, MR>& tensor ) {
			auto tensor_ptr = std::make_shared<Tensor<TInput, MR>>( tensor );
			inputs_.emplace( tensor_ptr );

			return tensor_ptr;
		}*/

		/**
		* @brief Calculates the total number of parameters in the model.
		*
		* @return size_t The total number of parameters.
		*/
		size_t parameters() const {// override {
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
		void print() const {//override {
			std::cout << "Modules: " << std::endl;
			for ( const auto& module : modules_ ) {
				module->print();
			}
			std::cout << "Total parameters: " << parameters() << std::endl;
		}

	private:
		std::unordered_set<std::shared_ptr<Tensor<TInput, MR>>> inputs_; ///< The setof input tensors.
		std::unordered_map<std::string, std::shared_ptr<Tensor<TInput, MR>>> input_tensor_map_;
		std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>> modules_; ///< The list of modules in the model.
		std::vector<std::string> module_names_; ///< The list of module names.

		std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>> execution_order;

		cudaGraph_t cuda_graph;
		cudaGraphExec_t cuda_graph_exec;
		bool cuda_graph_initialized = false;

		bool is_built_{ false }; ///< Indicates whether the model has been built.
		bool is_training_{ false }; ///< Indicates whether the model is in training mode.

		cudaStream_t stream_{}; ///< The CUDA stream for device memory resource.

		std::shared_ptr<Tensor<TCompute, MR>> registerModule( std::shared_ptr<Module<TInput, TCompute, TDevice>> module ) {
			modules_.push_back( module );

			return { nullptr };// input_tensor_map_[ module->output_id ];
		}
	};
}