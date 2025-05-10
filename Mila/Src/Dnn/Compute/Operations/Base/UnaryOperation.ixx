/**
 * @file UnaryOperation.ixx
 * @brief Abstract base class for unary operations in the compute framework.
 *
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
    * @tparam TOutput The data type of the output tensor elements.
    * @tparam TInput The data type of the input tensor elements. Defaults to TOutput,
    *         but can be specified differently for operations like Encoder that require
    *         a different input type (e.g., int) than the output type (e.g., float).
    * @tparam TDeviceType The device type (e.g., CPU, CUDA) on which the operation is executed.
    */
    export template <typename TOutput, typename TInput = TOutput, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TOutput>&& ValidTensorType<TInput>
    class UnaryOperation : public OperationBase<TOutput, TInput, TInput, TDeviceType> {
    public:
        /**
        * @brief Memory resource type based on device type.
        *
        * This type alias automatically selects the appropriate memory resource type
        * based on the device type. For CUDA devices, it uses CudaMemoryResource;
        * for CPU devices, it uses HostMemoryResource.
        */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

        /**
        * @brief Constructs a UnaryOperation with the specified operation type.
        *
        * @param operation_type The type of the operation.
        */
        UnaryOperation( OperationType operation_type )
            : OperationBase<TOutput, TInput, TInput, TDeviceType>( operation_type, CreateCompatibleContext<TDeviceType>() ) {}

        /**
        * @brief Constructs a UnaryOperation with the specified operation type and compute precision.
        *
        * @param operation_type The type of the operation.
        * @param compute_precision The precision to use for internal computations.
        */
        UnaryOperation( OperationType operation_type, ComputePrecision compute_precision )
            : OperationBase<TOutput, TInput, TInput, TDeviceType>( operation_type, compute_precision, CreateCompatibleContext<TDeviceType>() ) {}

        /**
        * @brief Constructs a UnaryOperation with the specified operation type and device context.
        *
        * @param operation_type The type of the operation.
        * @param context The device context to use for this operation.
        */
        UnaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : OperationBase<TOutput, TInput, TInput, TDeviceType>( operation_type, ValidateContext<TDeviceType>( context ) ) {}

        /**
        * @brief Constructs a UnaryOperation with the specified operation type, compute precision, and device context.
        *
        * @param operation_type The type of the operation.
        * @param compute_precision The precision to use for internal computations.
        * @param context The device context to use for this operation.
        */
        UnaryOperation( OperationType operation_type, ComputePrecision compute_precision, std::shared_ptr<DeviceContext> context )
            : OperationBase<TOutput, TInput, TInput, TDeviceType>( operation_type, compute_precision, ValidateContext<TDeviceType>( context ) ) {}

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
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const = 0;

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
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_grads ) const {
            throw std::runtime_error( "Operation does not support backward pass." );
        }

        /**
         * @brief Executes the backward pass of a unary operation with full gradient computation.
         *
         * This comprehensive backward method is designed for operations that need to compute:
         * 1. Gradients with respect to inputs (for backpropagation)
         * 2. Gradients with respect to parameters (for optimization)
         *
         * Derived classes may override this method to define the backward computation.
         * The default implementation throws an exception, indicating that the operation
         * does not support this type of backward pass.
         *
         * @param input Input tensor from the forward pass.
         * @param output_grad Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (e.g., weights, biases).
         * @param parameter_grads Output tensor for parameter gradients.
         * @param input_gradient Output tensor for gradients with respect to input.
         * @param properties Additional properties for the operation.
         * @param output_state Cache tensors from forward pass.
         *
         * @throws std::runtime_error If the operation does not support this backward pass.
         */
        virtual void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_grads,
            Tensor<TInput, MR>& input_grad,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {
            throw std::runtime_error( "Operation does not support full gradient backward pass." );
        }
    };
}
