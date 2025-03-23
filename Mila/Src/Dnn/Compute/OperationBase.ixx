module;  
#include <string>  
#include <memory>  
#include <vector>  
#include <stdexcept>  
#include <type_traits>  

export module Compute.OperationBase;  

import Dnn.Tensor;
import Dnn.TensorType;
import Dnn.TensorTraits;

import Compute.DeviceType; 
import Compute.ComputeDevice;
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
* @brief Base class for all compute operations.  
*  
* @tparam T The data type of the tensor elements.  
* @tparam TMemoryResource The memory resource type, must be derived from MemoryResource.  
*/  
export  
template <typename TInput, typename TCompute, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
	requires ValidTensorTypes<TInput, TCompute>
class OperationBase {  
public:
	using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;

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

	///**  
	//* @brief Executes the forward pass of the operation.  
	//*  
	//* @param input The input tensor.  
	//* @param parameters The parameters for the operation.  
	//* @param output The output tensor.  
	//* @param output_state Cache for the output tensors.  
	//*/  
	//virtual void forward(  
	//	const Tensor<TInput, MR>& input,  
	//	const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,  
	//	Tensor<TPrecision, MR>& output,  
	//	std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const = 0;  

	/**  
	* @brief Executes the backward pass of the operation.  
	*  
	* @param grad The gradient tensor.  
	* @param inputs The input tensors.  
	* @param output_grads The gradients of the output tensors.  
	*  
	* @throws std::runtime_error If the operation does not support backward pass.  
	*/  
	virtual void backward(  
		const Tensor<TInput, MR>& grad,  
		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& parameters,  
		std::vector<std::shared_ptr<Tensor<TCompute,MR>>>& output_grads ) const {  
		// Default implementation for backward pass  
		throw std::runtime_error( "Operation does not support backward pass." );  
	};  

private:  
	DeviceType device_type_; ///< The device type.  
	OperationType operation_type_; ///< The operation type.  
};  
}