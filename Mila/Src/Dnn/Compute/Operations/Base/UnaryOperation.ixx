/**
 * @file UnaryOperation.ixx
 * @brief Abstract base class for unary operations in the compute framework.
 *
 * This file defines the `UnaryOperation` class, which serves as an interface
 * for operations that take a single input tensor and produce a single output tensor.
 * Derived classes must implement the `forward()` method to define the forward
 * computation for the operation. Optionally, they can override the `backward()` method
 * to define the backward computation.
 *
 * The `UnaryOperation` class is templated to support various data types for
 * input and output tensors, as well as different device types (e.g., CPU, CUDA).
 *
 * @tparam TInput The data type of the input tensor elements.
 * @tparam TPrecision The data type of the output tensor elements.
 * @tparam TDevice The device type (e.g., CPU, CUDA) on which the operation is executed.
 *
 * @see OperationBase
 * @see BinaryOperation
 */

module;
#include <memory>  
#include <vector>
#include <stdexcept>
#include <type_traits>  

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContextHelpers;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
    * @brief Abstract base class for unary operations in the compute framework.
    *
    * The `UnaryOperation` class defines the interface for operations that take a single input tensor
    * and produce a single output tensor. Derived classes must implement the `forward()` method for
    * the forward pass and may optionally override the `backward()` method for the backward pass.
    *
    * Additional methods for shape validation and parameter initialization are provided to ensure
    * correctness and flexibility in derived classes.
    *
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TPrecision The data type of the output tensor elements.
    * @tparam TDevice The device type (e.g., CPU, CUDA) on which the operation is executed.
    */
    export template <typename TInput, typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
    class UnaryOperation : public OperationBase<TInput, TPrecision, TDeviceType> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, DeviceMemoryResource, HostMemoryResource>;

        /**
        * @brief Constructs a UnaryOperation with the specified operation type.
        *
        * @param operation_type The type of the operation.
        */
        UnaryOperation( OperationType operation_type )
            : OperationBase<TInput, TPrecision, TDeviceType>( operation_type, CreateCompatibleContext<TDeviceType>() ) {}

        /**
        * @brief Constructs a UnaryOperation with the specified operation type and device context.
        *
        * @param operation_type The type of the operation.
        * @param context The device context to use for this operation.
        */
        UnaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : OperationBase<TInput, TPrecision, TDeviceType>( operation_type, ValidateContext<TDeviceType>( context ) ) {}

        /**
        * @brief Virtual destructor for proper cleanup of derived classes.
        */
        virtual ~UnaryOperation() = default;

        /**
        * @brief Executes the forward pass of a unary operation.
        *
        * Derived classes must implement this method to define the forward computation.
        *
        * @param input The input tensor.
        * @param parameters The parameters for the operation (e.g., weights, biases).
        * @param properties Additional properties for the operation.
        * @param output The output tensor.
        * @param output_state Cache for intermediate results or output tensors.
        */
        virtual void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const = 0;

        /**
        * @brief Executes the backward pass of a unary operation.
        *
        * Derived classes may override this method to define the backward computation.
        * The default implementation throws an exception, indicating that the operation
        * does not support a backward pass.
        *
        * @param grad The gradient tensor from the next layer.
        * @param parameters The parameters for the operation (e.g., weights, biases).
        * @param output_grads Gradients for the output tensors.
        * @throws std::runtime_error If the operation does not support a backward pass.
        */
        virtual void backward(
            const Tensor<TInput, MR>& grad,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_grads ) const {
            throw std::runtime_error( "Operation does not support backward pass." );
        }
    };
}
