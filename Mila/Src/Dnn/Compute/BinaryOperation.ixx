module;
#include <memory>  
#include <vector>  
#include <type_traits>  

export module Compute.BinaryOperation;

import Dnn.Tensor;
import Dnn.TensorType;
import Dnn.TensorTraits;

import Compute.DeviceType; 
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.OperationType;  
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaMemoryResource;

import Compute.ComputeResource;
import Compute.CpuComputeResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Abstract class for binary operations.
	*
	* @tparam T The data type of the tensor elements.
	* @tparam TMemoryResource The memory resource type, must be derived from MemoryResource.
	*/
	export
	template <typename TInput, typename TCompute, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TCompute>
	class BinaryOperation : public OperationBase<TInput, TCompute, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;

		using OperationBase<TInput, TCompute, TDeviceType>::OperationBase;

		/**
		* @brief Executes the forward pass of a binary operation.
		*
		* @param input1 The first input tensor.
		* @param input2 The second input tensor.
		* @param parameters The parameters for the operation.
		* @param attributes The attributes for the operation (if any).
		* @param output The output tensor.
		* @param output_state Cache for the output tensors.
		*/
		virtual void forward(
			const Tensor<TInput, MR>& input1,
			const Tensor<TInput, MR>& input2,
			const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& parameters,
			const OperationAttributes& attributes,
			Tensor<TCompute, MR>& output,
			std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& output_state ) const = 0;
	};
}