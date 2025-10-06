/**
 * @file UnaryOperation.ixx
 * @brief Device-agnostic unary operation interface using abstract tensor data types
 *
 * This module provides a framework for implementing unary operations (single input,
 * single output) across heterogeneous compute environments using abstract TensorDataType
 * enumeration. The system enables seamless operation across different devices (CPU, CUDA)
 * without exposing device-specific concrete types to host compilation.
 *
 * Key architectural features:
 * - Abstract data type system prevents device-specific compilation dependencies
 * - Device-agnostic operation interface with compile-time device validation
 * - Type-safe parameter handling with automatic compatibility checking
 * - Support for scalar tensor operations (activations, reductions, etc.)
 * - Extensible design for various neural network operations
 */

module;
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Device-agnostic unary operation interface using abstract tensor data types
     *
     * Framework for implementing unary operations (single input, single output) across
     * heterogeneous compute environments using abstract TensorDataType enumeration.
     * Provides seamless operation execution on different devices without exposing
     * device-specific concrete types to host compilation.
     *
     * Core architectural principles:
     * - Abstract data types prevent device-specific compilation issues
     * - Device-agnostic interface with compile-time device and type validation
     * - Type-safe parameter handling with automatic compatibility verification
     * - Support for scalar tensor operations (rank 0)
     * - Extensible design supporting various neural network operations
     *
     * Common unary operations:
     * - Activation functions: ReLU, GELU, Sigmoid, Tanh, Softmax
     * - Normalization: BatchNorm, LayerNorm, InstanceNorm
     * - Element-wise: Square, Sqrt, Exp, Log, Abs, Negate
     * - Reductions: Sum, Mean, Max, Min (tensor -> scalar)
     * - Shape operations: Reshape, Transpose, Flatten
     *
     * Scalar tensor operations:
     * - Scalar activations (e.g., scalar -> scalar)
     * - Reductions (e.g., tensor -> scalar)
     * - Broadcasting (e.g., scalar -> tensor)
     *
     * The operation framework supports both simple forward-only computations and
     * comprehensive forward/backward passes for gradient-based optimization.
     *
     * @tparam TDeviceType The target device type for the operation (CPU or CUDA).
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
     *
     * @note Aligns with updated OperationBase using single data type parameter
     * @note Derived classes must implement device-appropriate computation kernels
     * @note Mixed-precision operations should be implemented as separate operations
     *
     * @see TensorDataType for supported abstract data type enumeration
     * @see TensorDataTypeTraits for compile-time data type characteristics
     * @see OperationBase for fundamental operation infrastructure
     * @see DeviceContext for device management and resource allocation
     *
     * Example usage:
     * @code
     * // CUDA GELU activation
     * class CudaGeluOp : public UnaryOperation<DeviceType::Cuda, TensorDataType::FP32> {
     * public:
     *     void forward(const TensorType& input, const Parameters& params,
     *                  TensorType& output, OutputState& state) const override {
     *         // CUDA GELU implementation
     *         if (input.isScalar()) {
     *             output.reshape({});
     *             output.item() = gelu_scalar(input.item());
     *         } else {
     *             output.reshape(input.shape());
     *             launchGeluKernel(input, output);
     *         }
     *     }
     * };
     *
     * // CPU reduction operation (tensor -> scalar)
     * class CpuSumOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32> {
     * public:
     *     void forward(const TensorType& input, const Parameters& params,
     *                  TensorType& output, OutputState& state) const override {
     *         output.reshape({}); // Scalar output
     *         output.item() = std::reduce(input.data(), input.data() + input.size(), 0.0f);
     *     }
     * };
     * @endcode
     */
    export template <DeviceType TDeviceType = DeviceType::Cuda, TensorDataType TDataType = TensorDataType::FP32>
        class UnaryOperation : public OperationBase<TDeviceType, TDataType> {
        public:
            // ====================================================================
            // Type Aliases and Compile-Time Properties
            // ====================================================================

            /**
             * @brief Memory resource type based on device type.
             *
             * Automatically selects CudaDeviceMemoryResource for CUDA or CpuMemoryResource for CPU.
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

            // ====================================================================
            // Construction and Destruction
            // ====================================================================

            /**
             * @brief Constructs unary operation with automatic device context creation
             *
             * Creates a compatible device context automatically based on the device type,
             * simplifying operation instantiation for common use cases.
             *
             * @param operation_type Type identifier for the specific operation
             *
             * @throws std::runtime_error If compatible device context creation fails
             *
             * @note Device context is automatically selected based on TDeviceType
             *
             * Example:
             * @code
             * class ReLU : public UnaryOperation<DeviceType::Cuda, TensorDataType::FP32> {
             * public:
             *     ReLU() : UnaryOperation(OperationType::Activation) {}
             * };
             * @endcode
             */
            explicit UnaryOperation( OperationType operation_type )
                : OperationBase<TDeviceType, TDataType>(
                    operation_type,
                    CreateCompatibleContext<TDeviceType>() ) {
            }

            /**
             * @brief Constructs unary operation with explicit device context
             *
             * Uses a pre-existing device context instance, enabling fine-grained control
             * over device configuration and resource sharing across multiple operations.
             *
             * @param operation_type Type identifier for the specific operation
             * @param context Device context instance to use for this operation
             *
             * @throws std::runtime_error If context is incompatible with TDeviceType
             * @throws std::invalid_argument If context is null
             *
             * Example:
             * @code
             * auto context = DeviceContext::create("CUDA:1"); // Specific GPU
             * ReLU op(OperationType::Activation, context);
             * @endcode
             */
            UnaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
                : OperationBase<TDeviceType, TDataType>(
                    operation_type,
                    ValidateContext<TDeviceType>( context ) ) {
            }

            /**
             * @brief Virtual destructor ensuring proper cleanup in derived classes
             */
            virtual ~UnaryOperation() = default;

            /**
             * @brief Copy operations explicitly deleted for performance safety
             */
            UnaryOperation( const UnaryOperation& ) = delete;
            UnaryOperation& operator=( const UnaryOperation& ) = delete;

            /**
             * @brief Move operations for efficient ownership transfer
             */
            UnaryOperation( UnaryOperation&& ) = default;
            UnaryOperation& operator=( UnaryOperation&& ) = default;

            // ====================================================================
            // Core Operation Interface
            // ====================================================================

            /**
             * @brief Executes the forward pass of the unary operation
             *
             * Derived classes must implement this method to define the forward computation
             * specific to their operation type and target device. The implementation should
             * handle device-specific optimizations, memory access patterns, and numerical
             * precision requirements.
             *
             * Input/Output tensor requirements:
             * - May have any rank (0 for scalars, 1+ for higher dimensions)
             * - Scalar inputs/outputs fully supported
             * - Output shape determined by operation semantics
             *
             * Common patterns:
             * - Element-wise: output shape = input shape (including scalars)
             * - Reductions: output is scalar (rank 0)
             * - Shape transformations: output shape differs from input
             *
             * @param input Input tensor to process (may be scalar).
             * @param parameters Vector of operation parameters (weights, biases, config).
             *                   May include scalar parameters.
             * @param output Output tensor where results will be stored (may be scalar).
             *               Will be resized if necessary.
             * @param output_state Cache for intermediate results needed during backward pass.
             *                     May include scalar tensors.
             *
             * @throws std::runtime_error If computation fails or resources are insufficient
             * @throws std::invalid_argument If tensor shapes or data types are incompatible
             *
             * @note Output tensor must be properly allocated/resized by the implementation
             * @note Supports scalar operations (rank 0 tensors)
             *
             * Example implementation:
             * @code
             * void forward(const TensorType& input, const Parameters& parameters,
             *              TensorType& output, OutputState& output_state) const override {
             *     // ReLU activation: max(0, x)
             *     if (input.isScalar()) {
             *         output.reshape({});
             *         output.item() = std::max(0.0f, input.item());
             *     } else {
             *         output.reshape(input.shape());
             *         auto input_data = input.data();
             *         auto output_data = output.data();
             *         for (size_t i = 0; i < input.size(); ++i) {
             *             output_data[i] = std::max(0.0f, input_data[i]);
             *         }
             *     }
             * }
             * @endcode
             */
            virtual void forward(
                const TensorType& input,
                const Parameters& parameters,
                TensorType& output,
                OutputState& output_state ) const = 0;

            /**
             * @brief Executes simplified backward pass computing parameter gradients only
             *
             * Simplified backward pass implementation that computes gradients with respect
             * to operation parameters but not input tensors. Suitable for operations that
             * are terminal in the computation graph or when input gradients are not required.
             *
             * @param output_grad Gradient tensor from subsequent operations (may be scalar).
             * @param parameters Original parameters used during the forward pass.
             * @param parameter_grads Output vector where computed parameter gradients will be stored.
             *                        May include scalar gradients.
             *
             * @throws std::runtime_error If operation does not support backward computation
             * @throws std::invalid_argument If gradient shapes are incompatible with parameters
             *
             * @note Default implementation throws exception indicating no backward support
             * @note Override to implement parameter-only gradient computation
             * @note Suitable for operations at computation graph boundaries
             */
            virtual void backward(
                const TensorType& output_grad,
                const Parameters& parameters,
                std::vector<std::shared_ptr<TensorType>>& parameter_grads ) const {
                throw std::runtime_error(
                    "Operation '" + this->getName() + "' does not support simplified backward pass." );
            }

            /**
             * @brief Executes comprehensive backward pass computing all gradients
             *
             * Comprehensive backward pass implementation that computes gradients with respect
             * to both input tensors (for backpropagation) and operation parameters (for optimization).
             * Essential for operations within the interior of neural network computation graphs.
             *
             * Gradient computation patterns:
             * - Element-wise ops: gradient shape matches input/output shapes (including scalars)
             * - Reductions: output gradient is scalar, input gradient matches input shape
             * - Broadcasting: gradient reduction needed for broadcast dimensions
             *
             * @param input Original input tensor from the forward pass (may be scalar).
             * @param output_grad Gradient of loss with respect to operation output (may be scalar).
             * @param parameters Original parameters used during forward computation.
             * @param parameter_grads Output vector where parameter gradients will be stored.
             * @param input_grad Output tensor where input gradients will be stored for backpropagation.
             * @param output_state Cached intermediate results from the forward pass.
             *
             * @throws std::runtime_error If operation does not support comprehensive gradient computation
             * @throws std::invalid_argument If tensor shapes or types are incompatible
             *
             * @note Default implementation throws exception indicating no backward support
             * @note Override to implement full gradient computation for training
             * @note Essential for operations requiring input gradient backpropagation
             * @note Should leverage cached forward pass results for computational efficiency
             *
             * Example implementation:
             * @code
             * void backward(const TensorType& input, const TensorType& output_grad,
             *               const Parameters& parameters,
             *               std::vector<std::shared_ptr<TensorType>>& parameter_grads,
             *               TensorType& input_grad,
             *               const OutputState& output_state) const override {
             *     // ReLU gradient: grad * (input > 0)
             *     if (input.isScalar()) {
             *         input_grad.reshape({});
             *         input_grad.item() = (input.item() > 0.0f) ? output_grad.item() : 0.0f;
             *     } else {
             *         input_grad.reshape(input.shape());
             *         auto input_data = input.data();
             *         auto grad_out = output_grad.data();
             *         auto grad_in = input_grad.data();
             *         for (size_t i = 0; i < input.size(); ++i) {
             *             grad_in[i] = (input_data[i] > 0.0f) ? grad_out[i] : 0.0f;
             *         }
             *     }
             * }
             * @endcode
             */
            virtual void backward(
                const TensorType& input,
                const TensorType& output_grad,
                const Parameters& parameters,
                std::vector<std::shared_ptr<TensorType>>& parameter_grads,
                TensorType& input_grad,
                const OutputState& output_state ) const {
                throw std::runtime_error(
                    "Operation '" + this->getName() + "' does not support comprehensive backward pass." );
            }

            // ====================================================================
            // Type-Safe Parameter Access
            // ====================================================================

            /**
             * @brief Provides type-safe access to operation parameters
             *
             * Safely casts type-erased parameter tensors to specific data types with
             * automatic validation and error reporting.
             *
             * @tparam TParamDataType Expected abstract data type of the parameter tensor
             * @param param Type-erased parameter tensor to access
             * @return Const reference to typed tensor with validated data type
             *
             * @throws std::runtime_error If parameter data type doesn't match expected type
             * @throws std::invalid_argument If parameter is null or invalid
             *
             * @note Performs runtime type checking to ensure parameter safety
             * @note Works with scalar parameter tensors (rank 0)
             *
             * Example usage:
             * @code
             * auto& weight = getTypedParameter<TensorDataType::FP32>(parameters[0]);
             * auto& bias = getTypedParameter<TensorDataType::FP32>(parameters[1]); // May be scalar
             * if (bias.isScalar()) {
             *     float bias_value = bias.item();
             * }
             * @endcode
             */
            template<TensorDataType TParamDataType>
            static const Tensor<TParamDataType, MR>& getTypedParameter(
                const std::shared_ptr<ITensor>& param ) {

                if (!param) {
                    throw std::invalid_argument( "Parameter tensor is null" );
                }

                if (param->getDataType() != TParamDataType) {
                    throw std::runtime_error(
                        "Parameter type mismatch - expected " +
                        std::string( TensorDataTypeTraits<TParamDataType>::type_name ) +
                        " but got " + param->getDataTypeName() );
                }

                return static_cast<const Tensor<TParamDataType, MR>&>(*param);
            }

            /**
             * @brief Provides mutable type-safe access to operation parameters
             *
             * Safely casts type-erased parameter tensors to specific data types with
             * mutable access for in-place parameter modifications during training.
             *
             * @tparam TParamDataType Expected abstract data type of the parameter tensor
             * @param param Type-erased parameter tensor to access
             * @return Mutable reference to typed tensor with validated data type
             *
             * @throws std::runtime_error If parameter data type doesn't match expected type
             * @throws std::invalid_argument If parameter is null or invalid
             *
             * @note Enables in-place parameter updates during optimization
             * @note Use with caution to maintain computational graph integrity
             */
            template<TensorDataType TParamDataType>
            static Tensor<TParamDataType, MR>& getMutableTypedParameter(
                const std::shared_ptr<ITensor>& param ) {

                if (!param) {
                    throw std::invalid_argument( "Parameter tensor is null" );
                }

                if (param->getDataType() != TParamDataType) {
                    throw std::runtime_error(
                        "Parameter type mismatch - expected " +
                        std::string( TensorDataTypeTraits<TParamDataType>::type_name ) +
                        " but got " + param->getDataTypeName() );
                }

                return static_cast<Tensor<TParamDataType, MR>&>(*param);
            }

            // ====================================================================
            // Operation Properties and Introspection
            // ====================================================================

            /**
             * @brief Returns operation configuration information
             *
             * Provides detailed information about the operation's configuration
             * including data type, memory resources, and device characteristics
             * for debugging and optimization analysis.
             *
             * @return String containing comprehensive operation configuration details
             */
            virtual std::string getOperationInfo() const {
                return this->getName() +
                    " (DataType: " + std::string( this->getDataTypeName() ) +
                    ", Device: " + (TDeviceType == DeviceType::Cuda ? "CUDA" : "CPU") +
                    ", Memory: " + (MR::is_host_accessible ? "Host" : "Device") + ")";
            }

            /**
             * @brief Validates operation tensor compatibility
             *
             * Performs comprehensive validation of input/output tensor compatibility
             * with operation requirements including shape constraints, data type
             * compatibility, and memory resource alignment.
             *
             * @param input Input tensor to validate
             * @param output Output tensor to validate
             * @return true if tensors are compatible with operation requirements
             *
             * @note Override in derived classes to implement operation-specific validation
             * @note Default implementation performs basic compatibility checks
             * @note Handles scalar tensors correctly
             */
            virtual bool validateTensors( const TensorType& input, const TensorType& output ) const {
                return input.getDataType() == TDataType &&
                    output.getDataType() == TDataType;
            }
    };

    // ====================================================================
    // Type Aliases for Common Operation Configurations
    // ====================================================================

    /**
     * @brief CPU unary operation with specified precision
     *
     * Convenient alias for unary operations using CPU memory,
     * suitable for CPU-only computation and development workflows.
     *
     * @tparam TDataType Tensor data type (defaults to FP32)
     *
     * Example:
     * @code
     * class CpuReLU : public CpuUnaryOperation<TensorDataType::FP32> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
        using CpuUnaryOperation = UnaryOperation<DeviceType::Cpu, TDataType>;

    /**
     * @brief CUDA unary operation with specified precision
     *
     * Convenient alias for unary operations using CUDA device memory,
     * optimized for GPU computation with support for advanced precision formats.
     *
     * @tparam TDataType Tensor data type (defaults to FP32)
     *
     * Example:
     * @code
     * class CudaGELU : public CudaUnaryOperation<TensorDataType::FP16> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
        using CudaUnaryOperation = UnaryOperation<DeviceType::Cuda, TDataType>;
}