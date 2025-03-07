module;
#include <miniz.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>

export module Dnn.Module;

import Dnn.Tensor;

import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.ComputeResource;
import Compute.CpuComputeResource;
import Compute.CudaComputeResource;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
	/**
	 * @brief Abstract base class for all modules in the DNN framework.
	 *
	 * @tparam T Data type of the tensor elements.
	 * @tparam MR Memory resource type, either CpuMemoryResource or DeviceMemoryResource.
	 */
	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = Compute::CpuDevice>
		requires ValidTensorTypes<TInput,TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class Module {
	public:
		using MR = TDevice::MR;

		virtual ~Module() = default;

		/**
		 * @brief Forward pass of the module.
		 *
		 * @param input Input tensor.
		 * @return Tensor<TCompute, MR> Output tensor.
		 */
		virtual void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) = 0;

		/**
		 * @brief Backward pass of the module.
		 *
		 * @param gradient Gradient tensor.
		 * @return Tensor<T, MR> Gradient with respect to the input.
		 */
		virtual Tensor<TCompute, MR> backward( const Tensor<TInput, MR>& gradient ) {
			// Default to no op
			return {};
		}

		/**
		 * @brief Set the training mode of the module.
		 *
		 * @param training True if the module is in training mode, false otherwise.
		 */
		void setTrainingMode( bool training ) {
			is_training_ = training;
		}

		/**
		 * @brief Check if the module is in training mode.
		 *
		 * @return true If the module is in training mode.
		 * @return false Otherwise.
		 */
		bool isTraining() const {
			return is_training_;
		}

		/**
		 * @brief Get the number of parameters in the module.
		 *
		 * @return size_t Number of parameters.
		 */
		virtual size_t parameters() const = 0;

		/**
		 * @brief Get the name of the module.
		 *
		 * @return std::string Name of the module.
		 */
		virtual std::string name() const = 0;

		/**
		 * @brief Print the module information.
		 */
		virtual void print() const = 0;

		/**
		 * @brief Save the module state to a file.
		 *
		 * @param zip The zip archive to save the state to.
		 */
		virtual void save( mz_zip_archive& zip ) const = 0;

		/**
		 * @brief Load the module state from a file.
		 *
		 * @param zip The zip archive to load the state from.
		 */
		virtual void load( mz_zip_archive& zip ) = 0;

		/**
		 * @brief Register a child module.
		 *
		 * @param name The name of the child module.
		 * @param module The child module to register.
		 */
		void registerModule( const std::string& name, std::shared_ptr<Module<TInput, TCompute, TDevice>> module ) {
			child_modules_[ name ] = module;
		}

		/**
		 * @brief Register a parameter tensor.
		 *
		 * @param name The name of the parameter tensor.
		 * @param tensor The parameter tensor to register.
		 */
		void registerParameter( const std::string& name, std::shared_ptr<Tensor<TCompute, MR>> tensor ) {
			named_parameters_[ name ] = tensor;
		}

	protected:
		std::unordered_map<std::string, std::shared_ptr<Module<TInput, TCompute, TDevice>>> child_modules_;
		std::unordered_map<std::string, std::shared_ptr<Tensor<TCompute, MR>>> named_parameters_;

		//std::vector<std::shared_ptr<INode>> sub_nodes;
		std::unordered_set<std::shared_ptr<Tensor<TCompute, MR>>> outputs_;

		std::shared_ptr<Tensor<TCompute, MR>> makeOutputTensor( std::string const& name ) {
			auto tensor = std::make_shared<Tensor<TInput, MR>>( name );
			// FIXME: tensor->set_name( name ).set_is_virtual( true );
			outputs_.insert( tensor );

			return tensor;
		}

	private:
		bool is_training_{ false }; ///< Training mode flag.
	};
}