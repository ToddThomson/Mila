module;
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <type_traits>

export module Dnn.Module;

import Dnn.Tensor;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn
{
	/**
	 * @brief Abstract base class for all modules in the DNN framework.
	 *
	 * @tparam T Data type of the tensor elements.
	 * @tparam MR Memory resource type, either CpuMemoryResource or DeviceMemoryResource.
	 */
	export
	template<typename TInput, typename TCompute = TInput, typename MR = Compute::CpuMemoryResource>
		requires std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::DeviceMemoryResource>
	class Module {
	public:
		virtual ~Module() = default;

		/**
		 * @brief Forward pass of the module.
		 *
		 * @param input Input tensor.
		 * @return Tensor<TCompute, MR> Output tensor.
		 */
		virtual Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) = 0;

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

	protected:

		//std::vector<std::shared_ptr<INode>> sub_nodes;
		std::unordered_set<std::shared_ptr<Tensor<TCompute, MR>>> outputs_;

		std::shared_ptr<Tensor<TCompute, MR>> output_tensor( std::string const& name ) {
			auto tensor = std::make_shared<Tensor<TInput, MR>>( name );
			// FIXME: tensor->set_name( name ).set_is_virtual( true );
			outputs_.insert( tensor );

			return tensor;
		}

	private:
		bool is_training_{ false }; ///< Training mode flag.
	};
}