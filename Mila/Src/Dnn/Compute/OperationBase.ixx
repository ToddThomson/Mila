module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

export module Compute.OperationBase;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Base class for all operations in the compute module.
	*
	* @tparam T The data type of the tensor elements.
	* @tparam MR The memory resource type, must be derived from MemoryResource.
	*/
	export
	template <typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class OperationBase {
	public:
		/**
		* @brief Constructs an OperationBase object.
		*
		* @param device_type The type of device on which the operation will be executed.
		* @param operation_type The type of the operation.
		*/
		OperationBase( DeviceType device_type, OperationType operation_type )
			: device_type_( device_type ), operation_type_( operation_type ) {}

		/**
		* @brief Virtual destructor for the OperationBase class.
		*/
		virtual ~OperationBase() = default;

		/**
		* @brief Gets the name of the operation.
		*
		* @return The name of the operation.
		*/
		virtual std::string getName() const = 0;

		/**
		* @brief Gets the device type.
		*
		* @return The device type.
		*/
		constexpr DeviceType getDeviceType() const {
			return device_type_;
		}

		/**
		* @brief Gets the operation type.
		*
		* @return The operation type.
		*/
		constexpr OperationType getOperationType() const {
			return operation_type_;
		}

		/**
		* @brief Executes the forward pass of the operation.
		*
		* @param input The input tensor.
		* @param input_attributes Additional attributes for the input tensor.
		* @param output The output tensor.
		* @param output_attributes Additional attributes for the output tensor.
		*/
		virtual void forward(
			const std::shared_ptr<Tensor<T, MR>> input,
			const std::vector<std::shared_ptr<Tensor<T, MR>>>& input_attributes,
			std::shared_ptr<Tensor<T, MR>> output,
			std::vector<std::shared_ptr<Tensor<T, MR>>>& output_attributes ) const = 0;

		/**
		* @brief Executes the backward pass of the operation.
		*
		* @param grad The gradient tensor.
		* @param inputs The input tensors.
		* @param outputGrads The gradients of the output tensors.
		*
		* @throws std::runtime_error If the operation does not support backward pass.
		*/
		virtual void backward(
			const std::shared_ptr<Tensor<T,MR>> grad,
			const std::vector<std::shared_ptr<Tensor<T,MR>>> inputs_attributes,
			std::vector<std::shared_ptr<Tensor<T,MR>>>& outputGrads ) const {
			// Default implementation for backward pass
			throw std::runtime_error( "Operation does not support backward pass." );
		};

	private:
		DeviceType device_type_; ///< The device type.
		OperationType operation_type_; ///< The operation type.
	};
}