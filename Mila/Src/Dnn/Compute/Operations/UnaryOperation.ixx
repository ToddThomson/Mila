/**
 * @file UnaryOperation.ixx
 * @brief Device-agnostic unary operation interface using abstract tensor data types
 *
 * This module provides a sophisticated framework for implementing unary operations
 * (single input, single output) across heterogeneous compute environments using
 * abstract TensorDataType enumeration. The system enables seamless operation across
 * different devices (CPU, CUDA, Metal, OpenCL, Vulkan) without exposing device-specific
 * concrete types to host compilation.
 *
 * Key architectural features:
 * - Abstract data type system prevents device-specific compilation dependencies
 * - Support for mixed-precision operations with different input/output data types
 * - Device-agnostic operation interface with compile-time device validation
 * - Type-safe parameter handling with automatic compatibility checking
 * - Extensible design for various neural network operations and custom kernels
 */

module;
#include <memory>  
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.TensorData;
import Dnn.TensorDataType;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;
//import Compute.TensorDataTypeCompatibility;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Device-agnostic unary operation interface using abstract tensor data types
     *
     * Advanced framework for implementing unary operations (single input, single output)
     * across heterogeneous compute environments using abstract TensorDataType enumeration.
     * Provides seamless operation execution on different devices without exposing
     * device-specific concrete types to host compilation.
     *
     * Core architectural principles:
     * - Abstract data types prevent device-specific compilation issues
     * - Support for mixed-precision workflows with different input/output types
     * - Device-agnostic interface with compile-time device and type validation
     * - Type-safe parameter handling with automatic compatibility verification
     * - Extensible design supporting various neural network operations and custom kernels
     *
     * The operation framework supports both simple forward-only computations and
     * comprehensive forward/backward passes for gradient-based optimization, enabling
     * efficient implementation of various neural network layers and activation functions.
     *
     * @tparam TInputDataType Abstract input tensor data type from TensorDataType enumeration
     * @tparam TOutputDataType Abstract output tensor data type from TensorDataType enumeration
     * @tparam TMemoryResource Memory resource type determining allocation strategy and device targeting
     *
     * @note Input and output data types must be compatible with the specified memory resource
     * @note Device-specific implementations are selected through compile-time dispatch
     * @note Derived classes must implement device-appropriate computation kernels
     *
     * @see TensorDataType for supported abstract data type enumeration
     * @see TensorDataTypeTraits for compile-time data type characteristics
     * @see OperationBase for fundamental operation infrastructure
     * @see DeviceContext for device management and resource allocation
     *
     * Example usage:
     * @code
     * // CUDA activation operation with mixed precision
     * class CudaGeluOp : public UnaryOperation<TensorDataType::FP16, TensorDataType::FP16, CudaMemoryResource> {
     *     void forward(const InputTensor& input, OutputTensor& output, ...) const override {
     *         // CUDA-specific GELU implementation
     *     }
     * };
     *
     * // CPU normalization operation
     * class CpuLayerNormOp : public UnaryOperation<TensorDataType::FP32, TensorDataType::FP32, CpuMemoryResource> {
     *     // CPU-specific layer normalization implementation
     * };
     * @endcode
     */
    export template <TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TOutputDataType = TInputDataType,
        typename TMemoryResource = CudaMemoryResource>
        requires isValidTensorConfiguration<TInputDataType, TMemoryResource>&&
    isValidTensorConfiguration<TOutputDataType, TMemoryResource>
        class UnaryOperation : public OperationBase<TInputDataType, TOutputDataType, TMemoryResource> {
        public:
            // ====================================================================
            // Type Aliases and Compile-Time Properties
            // ====================================================================

            using InputDataType = TensorDataType;                                      ///< Input tensor abstract data type
            using OutputDataType = TensorDataType;                                     ///< Output tensor abstract data type
            using MemoryResource = TMemoryResource;                                    ///< Memory resource type for tensor allocation
            using InputTensor = Tensor<TInputDataType, TMemoryResource>;               ///< Input tensor type alias
            using OutputTensor = Tensor<TOutputDataType, TMemoryResource>;             ///< Output tensor type alias
            using ParameterTensor = std::shared_ptr<ITensorData>;                      ///< Type-erased parameter tensor
            using GradientTensor = std::shared_ptr<OutputTensor>;                      ///< Gradient tensor type alias

            static constexpr TensorDataType input_data_type = TInputDataType;          ///< Compile-time input data type constant
            static constexpr TensorDataType output_data_type = TOutputDataType;        ///< Compile-time output data type constant
            static constexpr bool is_mixed_precision = (TInputDataType != TOutputDataType); ///< Mixed-precision operation detection
            static constexpr bool is_host_accessible = TMemoryResource::is_host_accessible; ///< Host accessibility detection
            static constexpr bool is_device_accessible = TMemoryResource::is_device_accessible; ///< Device accessibility detection

            // ====================================================================
            // Construction and Destruction
            // ====================================================================

            /**
             * @brief Constructs unary operation with automatic device context creation
             *
             * Creates a compatible device context automatically based on the memory
             * resource type, simplifying operation instantiation for common use cases.
             * The context is optimized for the target memory resource characteristics.
             *
             * @param operation_type Type identifier for the specific operation
             *
             * @throws std::runtime_error If compatible device context creation fails
             *
             * @note Device context is automatically selected based on memory resource type
             * @note Context configuration is optimized for the operation's requirements
             */
            explicit UnaryOperation( OperationType operation_type )
                : OperationBase<TInputDataType, TOutputDataType, TMemoryResource>(
                    operation_type, CreateCompatibleContext<TMemoryResource>() ) {}

            /**
             * @brief Constructs unary operation with explicit device context
             *
             * Uses a pre-existing device context instance, enabling fine-grained control
             * over device configuration and resource sharing across multiple operations.
             * Validates context compatibility with the memory resource type.
             *
             * @param operation_type Type identifier for the specific operation
             * @param context Device context instance to use for this operation
             *
             * @throws std::runtime_error If context is incompatible with memory resource
             * @throws std::invalid_argument If context is null
             *
             * @note Context must be compatible with the specified memory resource type
             * @note Enables advanced device management and resource optimization
             */
            UnaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
                : OperationBase<TInputDataType, TOutputDataType, TMemoryResource>(
                    operation_type, ValidateContext<TMemoryResource>( context ) ) {}

            /**
             * @brief Virtual destructor ensuring proper cleanup in derived classes
             *
             * Provides proper resource cleanup for polymorphic destruction,
             * enabling safe use of base class pointers to derived operation instances.
             */
            virtual ~UnaryOperation() = default;

            /**
             * @brief Copy operations explicitly deleted for performance safety
             *
             * Prevents accidental expensive copy operations involving complex
             * device state and operation configurations.
             */
            UnaryOperation( const UnaryOperation& ) = delete;
            UnaryOperation& operator=( const UnaryOperation& ) = delete;

            /**
             * @brief Move operations for efficient ownership transfer
             *
             * Enables efficient transfer of operation instances without
             * copying complex internal state or device resources.
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
             * @param input Input tensor to process with appropriate data type and memory layout
             * @param parameters Vector of operation parameters (weights, biases, configuration tensors)
             * @param output Output tensor where computation results will be stored
             * @param output_state Cache for intermediate results needed during backward pass
             *
             * @throws std::runtime_error If computation fails or resources are insufficient
             * @throws std::invalid_argument If tensor shapes or data types are incompatible
             *
             * @note Implementation must handle packed data types and device-specific memory layouts
             * @note Output tensor must be properly allocated and shaped before calling
             * @note Output state should store minimal information required for gradient computation
             * @note Device-specific implementations should leverage hardware-specific optimizations
             */
            virtual void forward(
                const InputTensor& input,
                const std::vector<ParameterTensor>& parameters,
                OutputTensor& output,
                std::vector<GradientTensor>& output_state ) const = 0;

            /**
             * @brief Executes simplified backward pass computing parameter gradients only
             *
             * Simplified backward pass implementation that computes gradients with respect
             * to operation parameters but not input tensors. Suitable for operations that
             * are terminal in the computation graph or when input gradients are not required.
             *
             * @param output_grad Gradient tensor from subsequent operations in the computation graph
             * @param parameters Original parameters used during the forward pass
             * @param parameter_grads Output vector where computed parameter gradients will be stored
             *
             * @throws std::runtime_error If operation does not support backward computation
             * @throws std::invalid_argument If gradient shapes are incompatible with parameters
             *
             * @note Default implementation throws exception indicating no backward support
             * @note Override to implement parameter-only gradient computation
             * @note Suitable for operations at computation graph boundaries
             */
            virtual void backward(
                const OutputTensor& output_grad,
                const std::vector<ParameterTensor>& parameters,
                std::vector<GradientTensor>& parameter_grads ) const {
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
             * @param input Original input tensor from the forward pass
             * @param output_grad Gradient of loss with respect to operation output
             * @param parameters Original parameters used during forward computation
             * @param parameter_grads Output vector where parameter gradients will be stored
             * @param input_grad Output tensor where input gradients will be stored for backpropagation
             * @param output_state Cached intermediate results from the forward pass
             *
             * @throws std::runtime_error If operation does not support comprehensive gradient computation
             * @throws std::invalid_argument If tensor shapes or types are incompatible
             *
             * @note Default implementation throws exception indicating no backward support
             * @note Override to implement full gradient computation for training
             * @note Essential for operations requiring input gradient backpropagation
             * @note Should leverage cached forward pass results for computational efficiency
             */
            virtual void backward(
                const InputTensor& input,
                const OutputTensor& output_grad,
                const std::vector<ParameterTensor>& parameters,
                std::vector<GradientTensor>& parameter_grads,
                InputTensor& input_grad,
                const std::vector<GradientTensor>& output_state ) const {
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
             * automatic validation and error reporting. Essential for accessing
             * parameters with known data types in device-specific implementations.
             *
             * @tparam TParamDataType Expected abstract data type of the parameter tensor
             * @param param Type-erased parameter tensor to access
             * @return Const reference to typed tensor with validated data type
             *
             * @throws std::runtime_error If parameter data type doesn't match expected type
             * @throws std::invalid_argument If parameter is null or invalid
             *
             * @note Performs runtime type checking to ensure parameter safety
             * @note Essential for accessing typed parameters in mixed-precision workflows
             * @note Enables compile-time optimization of parameter access patterns
             *
             * Example usage:
             * @code
             * auto& weight = getTypedParameter<TensorDataType::FP16>(parameters[0]);
             * auto& bias = getTypedParameter<TensorDataType::FP32>(parameters[1]);
             * @endcode
             */
            template<TensorDataType TParamDataType>
            static const Tensor<TParamDataType, TMemoryResource>& getTypedParameter(
                const ParameterTensor& param ) {

                if ( !param ) {
                    throw std::invalid_argument( "Parameter tensor is null" );
                }

                if ( param->getDataType() != TParamDataType ) {
                    throw std::runtime_error(
                        "Parameter type mismatch - expected " +
                        std::string( TensorDataTypeTraits<TParamDataType>::type_name ) +
                        " but got " + param->getDataTypeName() );
                }

                return static_cast<const Tensor<TParamDataType, TMemoryResource>&>(*param);
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
            static Tensor<TParamDataType, TMemoryResource>& getMutableTypedParameter(
                const ParameterTensor& param ) {

                if ( !param ) {
                    throw std::invalid_argument( "Parameter tensor is null" );
                }

                if ( param->getDataType() != TParamDataType ) {
                    throw std::runtime_error(
                        "Parameter type mismatch - expected " +
                        std::string( TensorDataTypeTraits<TParamDataType>::type_name ) +
                        " but got " + param->getDataTypeName() );
                }

                return static_cast<Tensor<TParamDataType, TMemoryResource>&>(*param);
            }

            // ====================================================================
            // Operation Properties and Introspection
            // ====================================================================

            /**
             * @brief Checks if operation supports mixed-precision computation
             *
             * Compile-time detection of whether the operation uses different data types
             * for inputs and outputs, enabling mixed-precision training optimizations
             * and specialized kernel dispatch.
             *
             * @return true if input and output use different data types, false otherwise
             */
            static constexpr bool supportsMixedPrecision() noexcept {
                return is_mixed_precision;
            }

            /**
             * @brief Checks if operation tensors are host-accessible
             *
             * Compile-time detection of host accessibility, determining whether
             * operation data can be directly accessed from CPU code without
             * explicit memory transfers.
             *
             * @return true if tensors are host-accessible, false for device-only memory
             */
            static constexpr bool isHostAccessible() noexcept {
                return is_host_accessible;
            }

            /**
             * @brief Checks if operation tensors are device-accessible
             *
             * Compile-time detection of device accessibility, determining whether
             * operation data can be accessed from device kernels and GPU operations.
             *
             * @return true if tensors are device-accessible, false for host-only memory
             */
            static constexpr bool isDeviceAccessible() noexcept {
                return is_device_accessible;
            }

            /**
             * @brief Returns operation configuration information
             *
             * Provides detailed information about the operation's configuration
             * including data types, memory resources, and precision characteristics
             * for debugging and optimization analysis.
             *
             * @return String containing comprehensive operation configuration details
             */
            virtual std::string getOperationInfo() const {
                return this->getName() +
                    " (Input: " + std::string( TensorDataTypeTraits<TInputDataType>::type_name ) +
                    ", Output: " + std::string( TensorDataTypeTraits<TOutputDataType>::type_name ) +
                    ", Memory: " + (is_host_accessible ? "Host" : "Device") +
                    (is_mixed_precision ? ", Mixed Precision" : "") + ")";
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
             */
            virtual bool validateTensors( const InputTensor& input, const OutputTensor& output ) const {
                return !input.empty() && !output.empty() &&
                    input.getDataType() == input_data_type &&
                    output.getDataType() == output_data_type;
            }
    };

    // ====================================================================
    // Type Aliases for Common Operation Configurations
    // ====================================================================

    /**
     * @brief CPU unary operation with single precision
     *
     * Convenient alias for unary operations using CPU memory with FP32 data types,
     * suitable for CPU-only computation and development workflows.
     *
     * @tparam TInputDataType Input tensor data type (defaults to FP32)
     * @tparam TOutputDataType Output tensor data type (defaults to input type)
     */
    export template<TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TOutputDataType = TInputDataType>
        using CpuUnaryOperation = UnaryOperation<TInputDataType, TOutputDataType, CpuMemoryResource>;

    /**
     * @brief CUDA unary operation with mixed precision support
     *
     * Convenient alias for unary operations using CUDA device memory,
     * optimized for GPU computation with support for advanced precision formats.
     *
     * @tparam TInputDataType Input tensor data type (defaults to FP32)
     * @tparam TOutputDataType Output tensor data type (defaults to input type)
     */
    export template<TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TOutputDataType = TInputDataType>
        using CudaUnaryOperation = UnaryOperation<TInputDataType, TOutputDataType, CudaMemoryResource>;

    /**
     * @brief Mixed-precision unary operation
     *
     * Convenient alias for operations using different precision for inputs
     * and outputs, enabling advanced mixed-precision training strategies.
     */
    export template<TensorDataType TInputDataType, TensorDataType TOutputDataType, typename TMemoryResource>
        using MixedPrecisionUnaryOperation = UnaryOperation<TInputDataType, TOutputDataType, TMemoryResource>;
}