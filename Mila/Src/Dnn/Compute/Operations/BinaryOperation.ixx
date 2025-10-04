/**
 * @file BinaryOperation.ixx
 * @brief Abstract base class for binary operations in the Mila neural network framework.
 * @details Provides an interface for operations that take two input tensors and produce
 *          a single output tensor. This serves as the foundation for operations like
 *          addition, multiplication, convolution, and other binary neural network functions.
 *
 * Key features:
 * - Abstract data type system using TensorDataType enumeration
 * - Device-agnostic design (CPU, CUDA)
 * - Support for scalar tensor operations (broadcasting, reductions)
 * - Type-safe operation dispatch
 * - Compile-time type validation
 *
 * The class supports operations on tensors of any rank, including scalars (rank 0).
 * Broadcasting semantics are handled by derived classes based on operation requirements.
 */

module;
#include <memory>
#include <vector>
#include <type_traits>
#include <stdexcept>

export module Compute.BinaryOperation;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.DeviceType;
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract class for binary operations in the neural network framework.
     *
     * @details This class extends OperationBase to provide specialized functionality for
     * operations that take two input tensors and produce a single output tensor. Derived classes
     * must implement the forward() method to define the specific computation for the operation.
     *
     * Binary operations include:
     * - Element-wise operations: Add, Subtract, Multiply, Divide
     * - Matrix operations: MatMul, BatchedMatMul
     * - Broadcasting operations: AddBias, ScalarMultiply
     * - Comparison operations: Equal, Greater, Less
     * - Complex operations: Convolution, CrossCorrelation
     *
     * Scalar tensor support:
     * - Scalar-tensor operations (e.g., scalar + tensor)
     * - Tensor-scalar operations (e.g., tensor * scalar)
     * - Scalar-scalar operations (e.g., scalar * scalar -> scalar)
     *
     * The class uses abstract TensorDataType enumeration for type safety across different
     * precision formats while maintaining compile-time optimization and device independence.
     *
     * @tparam TDeviceType The target device type for the operation, defaults to DeviceType::Cuda.
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example usage:
     * @code
     * // FP32 CUDA matrix multiplication
     * class MatMul : public BinaryOperation<DeviceType::Cuda, TensorDataType::FP32> {
     *     void forward(const TensorType& A, const TensorType& B,
     *                  const Parameters& params, TensorType& output,
     *                  OutputState& state) const override;
     * };
     *
     * // FP16 CPU element-wise addition
     * class Add : public BinaryOperation<DeviceType::Cpu, TensorDataType::FP16> {
     *     void forward(const TensorType& A, const TensorType& B,
     *                  const Parameters& params, TensorType& output,
     *                  OutputState& state) const override;
     * };
     * @endcode
     */
    export template <DeviceType TDeviceType = DeviceType::Cuda, TensorDataType TDataType = TensorDataType::FP32>
        class BinaryOperation : public OperationBase<TDeviceType, TDataType> {
        public:
            /**
             * @brief Memory resource type based on device type.
             *
             * @details This type alias automatically selects the appropriate memory resource type
             * based on the template device type. For CUDA devices, it uses CudaDeviceMemoryResource;
             * for CPU devices, it uses CpuMemoryResource. This ensures memory allocation is
             * performed correctly for the target device.
             */
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

            /**
             * @brief Tensor type used by this operation.
             *
             * Convenient alias for the concrete tensor type with matching data type and memory resource.
             */
            using TensorType = Tensor<TDataType, MR>;

            /**
             * @brief Type alias for operation parameters (learnable weights, biases, etc.)
             *
             * Parameters are stored as type-erased ITensor pointers to support
             * dynamic parameter management across different tensor types.
             */
            using Parameters = std::vector<std::shared_ptr<ITensor>>;

            /**
             * @brief Type alias for operation state (cached values for backward pass)
             *
             * State tensors store intermediate computations needed for gradient calculation.
             * May include scalar tensors for normalization factors or reduction values.
             */
            using OutputState = std::vector<std::shared_ptr<TensorType>>;

            /**
             * @brief Constructs a BinaryOperation with the specified operation type.
             *
             * @details Creates a device context that matches the TDeviceType template parameter
             * using the CreateCompatibleContext helper function. This constructor simplifies
             * creation when a custom device context is not needed.
             *
             * @param operation_type The type of the operation from the OperationType enumeration.
             *
             * @throws std::runtime_error If device context creation fails.
             *
             * Example:
             * @code
             * class Add : public BinaryOperation<DeviceType::Cuda, TensorDataType::FP32> {
             * public:
             *     Add() : BinaryOperation(OperationType::Add) {}
             * };
             * @endcode
             */
            BinaryOperation( OperationType operation_type )
                : OperationBase<TDeviceType, TDataType>(
                    operation_type,
                    CreateCompatibleContext<TDeviceType>() ) {
            }

            /**
             * @brief Constructs a BinaryOperation with the specified operation type and device context.
             *
             * @details Validates that the provided context is compatible with the TDeviceType template
             * parameter. This allows for more control over the execution environment by providing a
             * pre-configured device context (e.g., specific GPU, custom streams).
             *
             * @param operation_type The type of the operation from the OperationType enumeration.
             * @param context The device context to use for this operation. Must be compatible with TDeviceType.
             *
             * @throws std::runtime_error If the provided context is incompatible with TDeviceType.
             *
             * Example:
             * @code
             * auto context = DeviceContext::create("CUDA:1"); // Specific GPU
             * MatMul op(OperationType::MatMul, context);
             * @endcode
             */
            BinaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
                : OperationBase<TDeviceType, TDataType>(
                    operation_type,
                    ValidateContext<TDeviceType>( context ) ) {
            }

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
             *
             * Input tensor requirements:
             * - May have any rank (0 for scalars, 1 for vectors, 2+ for higher dimensions)
             * - Broadcasting semantics depend on the specific operation
             * - Scalar inputs are fully supported (e.g., scalar + tensor, scalar * scalar)
             *
             * Output tensor:
             * - Will be resized by the operation if needed
             * - Shape determined by operation semantics and broadcasting rules
             * - May be scalar (rank 0) for reduction operations
             *
             * @param inputA The first input tensor to the operation (may be scalar).
             * @param inputB The second input tensor to the operation (may be scalar).
             * @param parameters Optional operation-specific learnable parameters (e.g., weights, biases).
             *                   May include scalar parameters (e.g., bias values).
             * @param output Pre-allocated or unallocated tensor where operation results will be stored.
             *               Will be resized if necessary.
             * @param output_state Optional cache for intermediate values needed during backward pass.
             *                     May include scalar tensors (e.g., normalization factors).
             *
             * @throws std::runtime_error If input shapes are incompatible for the operation.
             * @throws std::runtime_error If device context operations fail.
             *
             * Example implementation:
             * @code
             * void forward(const TensorType& inputA, const TensorType& inputB,
             *              const Parameters& parameters,
             *              TensorType& output, OutputState& output_state) const override {
             *     // Element-wise addition with broadcasting
             *     if (inputA.isScalar() && inputB.isScalar()) {
             *         // Scalar + scalar -> scalar
             *         output.reshape({});
             *     } else if (inputA.isScalar()) {
             *         // Scalar + tensor -> tensor (broadcast scalar)
             *         output.reshape(inputB.shape());
             *     } else if (inputB.isScalar()) {
             *         // Tensor + scalar -> tensor (broadcast scalar)
             *         output.reshape(inputA.shape());
             *     } else {
             *         // Tensor + tensor -> validate shapes and compute
             *         validateBroadcastable(inputA.shape(), inputB.shape());
             *         output.reshape(computeOutputShape(inputA.shape(), inputB.shape()));
             *     }
             *     // Perform the actual computation...
             * }
             * @endcode
             */
            virtual void forward(
                const TensorType& inputA,
                const TensorType& inputB,
                const Parameters& parameters,
                TensorType& output,
                OutputState& output_state ) const = 0;

            /**
             * @brief Executes the backward pass of a binary operation.
             *
             * @details Computes gradients with respect to both inputs and parameters by propagating the output
             * gradient backward through the operation. Derived classes may override this method to define
             * their specific backward computation.
             *
             * Gradient computation:
             * - Propagates output gradient to both input tensors
             * - Computes parameter gradients if operation has learnable parameters
             * - Handles broadcasting by summing gradients over broadcast dimensions
             * - Supports scalar gradients (e.g., scalar loss value)
             *
             * Scalar gradient handling:
             * - If output is scalar, output_gradient is also scalar (rank 0)
             * - If inputs are scalars, corresponding gradients will be scalars
             * - Broadcasting requires gradient reduction (sum) over expanded dimensions
             *
             * The default implementation throws an exception indicating that the operation
             * does not support a backward pass (e.g., inference-only operations).
             *
             * @param inputA First input tensor from the forward pass (may be scalar).
             * @param inputB Second input tensor from the forward pass (may be scalar).
             * @param output Output tensor from the forward pass (may be scalar).
             * @param output_gradient Gradient of the loss with respect to the output (may be scalar).
             * @param parameters Parameters tensor from forward pass (may include scalars).
             * @param parameter_gradients Output vector where parameter gradients will be stored (may include scalars).
             * @param inputA_gradient Output tensor where gradients for the first input will be stored (may be scalar).
             * @param inputB_gradient Output tensor where gradients for the second input will be stored (may be scalar).
             * @param output_state Cache tensors from forward pass (may include scalars).
             *
             * @throws std::runtime_error If the operation does not support backward pass.
             * @throws std::runtime_error If gradient shapes are incompatible.
             *
             * Example implementation:
             * @code
             * void backward(const TensorType& inputA, const TensorType& inputB,
             *               const TensorType& output, const TensorType& output_gradient,
             *               const std::shared_ptr<ITensor>& parameters,
             *               std::vector<std::shared_ptr<TensorType>>& parameter_gradients,
             *               TensorType& inputA_gradient, TensorType& inputB_gradient,
             *               const OutputState& output_state) const override {
             *     // Element-wise addition: gradient flows equally to both inputs
             *     if (inputA.isScalar() && inputB.isScalar()) {
             *         // Scalar gradient -> scalar gradients
             *         inputA_gradient.reshape({});
             *         inputB_gradient.reshape({});
             *         inputA_gradient.item() = output_gradient.item();
             *         inputB_gradient.item() = output_gradient.item();
             *     } else if (inputA.isScalar()) {
             *         // Reduce gradient for scalar input
             *         inputA_gradient.reshape({});
             *         inputA_gradient.item() = sum(output_gradient);
             *         inputB_gradient = output_gradient;
             *     } else {
             *         // Handle broadcasting and compute gradients...
             *     }
             * }
             * @endcode
             */
            virtual void backward(
                const TensorType& inputA,
                const TensorType& inputB,
                const TensorType& output,
                const TensorType& output_gradient,
                const std::shared_ptr<ITensor>& parameters,
                std::vector<std::shared_ptr<TensorType>>& parameter_gradients,
                TensorType& inputA_gradient,
                TensorType& inputB_gradient,
                const OutputState& output_state ) const {
                throw std::runtime_error( "Operation does not support backward pass." );
            }
    };

    /**
     * @brief Type alias for CPU-based binary operations with customizable data type.
     *
     * Convenient alias for creating CPU binary operations with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * class CpuAdd : public CpuBinaryOperation<TensorDataType::FP32> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
        using CpuBinaryOperation = BinaryOperation<DeviceType::Cpu, TDataType>;

    /**
     * @brief Type alias for CUDA-based binary operations with customizable data type.
     *
     * Convenient alias for creating CUDA binary operations with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * class CudaMatMul : public CudaBinaryOperation<TensorDataType::FP16> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
        using CudaBinaryOperation = BinaryOperation<DeviceType::Cuda, TDataType>;
}