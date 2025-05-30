/**
 * @file BinaryOperation.ixx
 * @brief Abstract base class for binary operations in the Mila neural network framework.
 * @details Provides an interface for operations that take two input tensors and produce
 *          a single output tensor. This serves as the foundation for operations like
 *          addition, multiplication, convolution, and other binary neural network functions.
 *          Supports configurable compute precision policies for mixed-precision computation.
 */

module;
#include <memory>  
#include <vector>  
#include <type_traits>  
#include <stdexcept>

export module Compute.BinaryOperation;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.DeviceType;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract class for binary operations in the neural network framework.
     *
     * @details This class extends OperationBase to provide specialized functionality for
     * operations that take two input tensors and produce a single output tensor. Derived classes
     * must implement the forward() method to define the specific computation for the operation.
     * Examples include element-wise operations (add, multiply), matrix operations (matmul),
     * and more complex operations like convolution.
     *
     * The class supports configurable compute precision policies which control how operations
     * handle mixed precision computations. This allows for optimizing between performance and
     * accuracy based on the specific requirements of the application.
     *
     * @tparam TDeviceType The target device type for the operation, defaults to DeviceType::Cuda.
     * @tparam TInput1 The data type of the first input tensor elements.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TInput2 The data type of the second input tensor elements, defaults to TInput1.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput1.
     *                 Must satisfy ValidFloatTensorType constraint.
     */
    export template <DeviceType TDeviceType = DeviceType::Cuda, typename TInput1 = float, typename TInput2 = TInput1, typename TOutput = TInput1>
        requires ValidTensorTypes<TInput1, TInput2>&& ValidFloatTensorType<TOutput>
    class BinaryOperation : public OperationBase<TDeviceType, TInput1, TInput2, TOutput> {
    public:

        // TODO: Review the need for need for different TInput2 type. 
        //       Binary operations typically use the same type for both inputs.

        /**
         * @brief Memory resource type based on device type.
         *
         * @details This type alias automatically selects the appropriate memory resource type
         * based on the template device type. For CUDA devices, it uses CudaMemoryResource;
         * for CPU devices, it uses HostMemoryResource. This ensures memory allocation is
         * performed correctly for the target device.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

        /**
         * @brief Constructs a BinaryOperation with the specified operation type and precision policy.
         *
         * @details Creates a device context that matches the TDeviceType template parameter
         * using the CreateCompatibleContext helper function. This constructor simplifies
         * creation when a custom device context is not needed. The precision policy controls
         * how the operation handles mixed precision computations.
         *
         * @param operation_type The type of the operation from the OperationType enumeration.
         * @param precision_policy The compute precision policy to use. Controls the balance between
         *                         performance and accuracy for mixed precision operations.
         *                         Defaults to Auto which lets the implementation decide based on hardware.
         */
        BinaryOperation( OperationType operation_type )
            : OperationBase<TDeviceType, TInput1, TInput2, TOutput>(
                operation_type,
                CreateCompatibleContext<TDeviceType>() ) {}

        /**
         * @brief Constructs a BinaryOperation with the specified operation type, device context, and precision policy.
         *
         * @details Validates that the provided context is compatible with the TDeviceType template
         * parameter. This allows for more control over the execution environment by providing a
         * pre-configured device context. The precision policy controls how the operation
         * handles mixed precision computations.
         *
         * @param operation_type The type of the operation from the OperationType enumeration.
         * @param context The device context to use for this operation. Must be compatible with TDeviceType.
         * @param precision_policy The compute precision policy to use. Controls the balance between
         *                         performance and accuracy for mixed precision operations.
         *                         Defaults to Auto which lets the implementation decide based on hardware.
         * @throws std::runtime_error If the provided context is incompatible with TDeviceType.
         */
        BinaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : OperationBase<TDeviceType, TInput1, TInput2, TOutput>(
                operation_type,
                ValidateContext<TDeviceType>( context ) ) {}

        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         *
         * @details Ensures proper cleanup of derived class resources when destroyed through
         * a base class pointer. Default implementation is sufficient for this base class.
         */
        virtual ~BinaryOperation() = default;

        /**
         * @brief Executes the forward pass of a binary operation.
         *
         * @details Performs the computation defined by the specific binary operation,
         * transforming the two input tensors into an output tensor according to the operation's
         * rules. Derived classes must implement this method to define their specific computation.
         * The method also supports additional parameters and operation-specific attributes.
         *
         * The operation's precision policy may affect how the computation is performed,
         * balancing between performance and accuracy based on the policy setting.
         *
         * @param input1 The first input tensor to the operation.
         * @param input2 The second input tensor to the operation.
         * @param parameters Optional operation-specific learnable parameters (e.g., weights, biases).
         * @param attributes Configuration settings that control the operation's behavior.
         * @param output Pre-allocated tensor where the operation results will be stored.
         * @param output_state Optional cache for intermediate values needed during backward pass.
         */
        virtual void forward(
            const Tensor<TInput1, MR>& input1,
            const Tensor<TInput2, MR>& input2,
            const std::vector<std::shared_ptr<Tensor<TInput1, MR>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const = 0;

        /**
         * @brief Executes the backward pass of a binary operation.
         *
         * @details Computes gradients with respect to both inputs and parameters by propagating the output
         * gradient backward through the operation. Derived classes may override this method to define
         * their specific backward computation.
         *
         * The operation's precision policy may affect how the gradient computation is performed,
         * balancing between performance and accuracy based on the policy setting.
         *
         * The default implementation throws an exception indicating that the operation
         * does not support a backward pass.
         *
         * @param input1 First input tensor from the forward pass.
         * @param input2 Second input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass.
         * @param parameter_gradients Output vector where parameter gradients will be stored.
         * @param input1_gradient Output tensor where gradients for the first input will be stored.
         * @param input2_gradient Output tensor where gradients for the second input will be stored.
         * @param attributes Configuration settings that control the operation's behavior.
         * @param output_state Cache tensors from forward pass.
         *
         * @throws std::runtime_error If the operation does not support backward pass.
         */
        virtual void backward(
            const Tensor<TInput1, MR>& input1,
            const Tensor<TInput2, MR>& input2,
            const Tensor<TOutput, MR>& output,
            const Tensor<TOutput, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput1, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_gradients,
            Tensor<TInput1, MR>& input1_gradient,
            Tensor<TInput2, MR>& input2_gradient,
            const OperationAttributes& attributes,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {
            throw std::runtime_error( "Operation does not support backward pass." );
        }
    };
}