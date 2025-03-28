module;
#include <memory>  
#include <vector>  
#include <type_traits>  

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.TensorType;
import Dnn.TensorTraits;

import Compute.DeviceType; 
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.OperationType;  
import Compute.OperationAttributes;
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaMemoryResource;

import Compute.ComputeResource;
import Compute.CpuComputeResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Abstract class for unary operations.
	*
	* @tparam T The data type of the tensor elements.
	* @tparam TMemoryResource The memory resource type, must be derived from MemoryResource.
	*/
	export
		template <typename TInput, typename TCompute, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TCompute>
	class UnaryOperation : public OperationBase<TInput, TCompute, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		using OperationBase<TInput, TCompute, TDeviceType>::OperationBase;

		/**
		* @brief Executes the forward pass of a unary operation.
		*
		* @param input The input tensor.
		* @param parameters The parameters for the operation.
		* @param output The output tensor.
		* @param output_state Cache for the output tensors.
		*/
		virtual void forward(
			const Tensor<TInput, MR>& input,
			const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TCompute, MR>& output,
			std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& output_state ) const = 0;
	};
}