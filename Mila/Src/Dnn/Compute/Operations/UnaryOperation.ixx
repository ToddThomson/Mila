/**
 * @file UnaryOperation.ixx
 * @brief Abstract base class for unary operations in the compute framework.
 */

module;
#include <memory>  
#include <vector>
#include <stdexcept>
#include <type_traits>  

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
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
    * @tparam TOutput The data type of the output tensor elements. Defaults to TInput.
    * @tparam TDeviceType The device type (e.g., CPU, CUDA) on which the operation is executed.
    */
    export template <DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
    class UnaryOperation : public OperationBase<TDeviceType, TInput, TInput, TOutput> {
    public:
        /**
        * @brief Memory resource type based on device type.
        *
        * This type alias automatically selects the appropriate memory resource type
        * based on the device type. For CUDA devices, it uses CudaMemoryResource;
        * for CPU devices, it uses CpuMemoryResource.
        */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

        /**
        * @brief Constructs a UnaryOperation with the specified operation type.
        *
        * Creates a compatible device context automatically based on TDeviceType.
        *
        * @param operation_type The type of the operation.
        * @param precision_policy The compute precision policy to use.
        */
        UnaryOperation( OperationType operation_type )
            : OperationBase<TDeviceType, TInput, TInput, TOutput>(
                operation_type, CreateCompatibleContext<TDeviceType>() ) {}

        /**
        * @brief Constructs a UnaryOperation with the specified operation type and device context.
        *
        * Validates that the provided context is compatible with TDeviceType.
        *
        * @param operation_type The type of the operation.
        * @param context The device context to use for this operation.
        * @param precision_policy The compute precision policy to use.
        */
        UnaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : OperationBase<TDeviceType, TInput, TInput, TOutput>(
                operation_type,
                ValidateContext<TDeviceType>( context ) ) {}

        /**
        * @brief Virtual destructor for proper cleanup of derived classes.
        */
        virtual ~UnaryOperation() = default;

        /**
        * @brief Executes the forward pass of a unary operation.
        *
        * Derived classes must implement this method to define the forward computation.
        *
        * @param input The input tensor to process.
        * @param parameters The parameters for the operation (e.g., weights, biases).
        * @param properties Additional properties that configure the operation's behavior.
        * @param output The output tensor where results will be stored.
        * @param output_state Cache for intermediate results needed in backward pass.
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
        * This is the simplified version of the backward pass that only computes gradients
        * for the parameters, not for the input.
        *
        * Derived classes may override this method to define the backward computation.
        * The default implementation throws an exception, indicating that the operation
        * does not support a backward pass.
        *
        * @param grad The gradient tensor from the next layer in the network.
        * @param parameters The parameters used during the forward pass.
        * @param output_grads Output vector where parameter gradients will be stored.
        * @throws std::runtime_error If the operation does not support backward pass.
        */
        virtual void backward(
            const Tensor<TInput, MR>& grad,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_grads ) const {
            throw std::runtime_error( "Operation does not support backward pass." );
        }

        /**
         * @brief Executes the comprehensive backward pass of a unary operation.
         *
         * This comprehensive backward method computes:
         * 1. Gradients with respect to inputs (for backpropagation)
         * 2. Gradients with respect to parameters (for optimization)
         *
         * This version provides more flexibility for operations that need access to both
         * the original input and the intermediates from the forward pass to compute gradients.
         *
         * @param input The original input tensor from the forward pass.
         * @param output_grad Gradient of the loss with respect to this operation's output.
         * @param parameters The parameters used during the forward pass.
         * @param parameter_grads Output vector where parameter gradients will be stored.
         * @param input_grad Output tensor where input gradients will be stored.
         * @param properties The same properties used during the forward pass.
         * @param output_state Cached tensors from the forward pass.
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