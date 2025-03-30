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

namespace Mila::Dnn::Compute
{
    /**
    * @brief Abstract class for unary operations.
    *
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TPrecision The data type of the output tensor elements.
    * @tparam TDeviceType The type of device where the operation will be executed, defaults to Compute::DeviceType::Cuda.
    */
    export
    template <typename TInput, typename TPrecision, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
    requires ValidTensorTypes<TInput, TPrecision>
    class UnaryOperation : public OperationBase<TInput, TPrecision, TDeviceType> {
	public:
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;
		using OperationBase<TInput, TPrecision, TDeviceType>::OperationBase;

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
			const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
			const OperationAttributes& properties,
			Tensor<TPrecision, MR>& output,
			std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const = 0;
	};
}