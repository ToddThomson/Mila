/**
 * @file OperationBase.ixx
 * @brief Core abstraction for neural network operations in the Mila framework
 * @details
 * Defines the fundamental OperationBase template class which serves as the foundation
 * for all computational operations in the Mila neural network architecture. This class
 * establishes the contract that concrete operations must fulfill while providing common
 * functionality related to device context management, operation identification, and
 * type safety.
 *
 * The template parameters enable operations to work across different computation devices
 * (CPU, CUDA) with various precision formats using abstract TensorDataType enumeration.
 * This ensures type safety and enables compile-time optimization while supporting modern
 * precision formats including FP8, FP16, BF16, and integer types.
 *
 * Key features:
 * - Abstract data type system using TensorDataType enumeration
 * - Device-agnostic design (CPU, CUDA)
 * - Support for scalar tensor operations (reductions, loss values, etc.)
 * - Type-safe operation dispatch
 * - Compile-time type validation
 *
 * This abstraction is extended by specialized operation classes like UnaryOperation
 * and BinaryOperation, which implement specific computational patterns.
 *
 * This module is designed for C++23 using class modules and employs tensor type constraints
 * to enforce safety and correctness at compile time.
 *
 * @see UnaryOperation
 * @see BinaryOperation
 * @see DeviceContext
 * @see OperationType
 * @see TensorDataType
 */

module;
#include <string>
#include <memory>
#include <stdexcept>

export module Compute.OperationBase;

import Dnn.TensorDataType;
import Dnn.TensorTypeTraits;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.Precision;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Base class for all compute operations in the Mila neural network framework.
     *
     * @details This abstract base class defines the common interface for all operations that can be
     * performed in the neural network computation graph, regardless of the device type
     * (CPU, CUDA, etc). Specific operations inherit from this class and implement their
     * specialized behavior while adhering to a consistent interface.
     *
     * Uses abstract TensorDataType enumeration to provide type-safe operations across
     * different precision formats while maintaining compile-time optimization and
     * device independence. Supports operations on scalar tensors (rank 0) for reductions,
     * loss values, and other zero-dimensional operations.
     *
     * Type system design:
     * - Abstract data types via TensorDataType enumeration
     * - Compile-time type validation and dispatch
     * - No device-specific types in host code
     * - Support for modern precision formats (FP8, FP16, BF16)
     *
     * @tparam TDeviceType The target device type for the operation, defaults to DeviceType::Cuda.
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example usage:
     * @code
     * // FP32 CUDA operation
     * class MyOperation : public OperationBase<DeviceType::Cuda, TensorDataType::FP32> {
     *     // Implementation
     * };
     *
     * // FP16 CPU operation
     * class MyOperation : public OperationBase<DeviceType::Cpu, TensorDataType::FP16> {
     *     // Implementation
     * };
     * @endcode
     */
    export template <DeviceType TDeviceType = DeviceType::Cuda, TensorDataType TDataType = TensorDataType::FP32>
    class OperationBase {
    public:
        /**
         * @brief Compile-time data type traits for this operation.
         *
         * Provides access to data type characteristics at compile time,
         * enabling type-safe operation dispatch and validation.
         */
        using DataTypeTraits = TensorDataTypeTraits<TDataType>;

        /**
         * @brief The abstract data type used by this operation.
         *
         * Compile-time constant enabling static introspection of the
         * operation's data type without runtime overhead.
         */
        static constexpr TensorDataType data_type = TDataType;

        /**
         * @brief The device type used by this operation.
         *
         * Compile-time constant enabling static introspection of the
         * operation's target device without runtime overhead.
         */
        static constexpr DeviceType device_type = TDeviceType;

    protected:
        /**
         * @brief Constructs an OperationBase object with a specific operation type and device context.
         *
         * @details Initializes the operation with the specified operation type and device context.
         * The context defines the execution environment for this operation. Validates that
         * the device context is not null and optionally that it matches the template device type.
         *
         * @param operation_type The type of the operation (from OperationType enum).
         * @param context The device context to use for this operation. Must not be null.
         * @throw std::invalid_argument If context is null.
         * @throw std::runtime_error If context device type doesn't match TDeviceType (optional validation).
         *
         * @note Protected constructor - only derived classes can instantiate
         * @note Device context validation ensures type safety at runtime
         */
        OperationBase(OperationType operation_type, std::shared_ptr<DeviceContext> context)
            : operation_type_(operation_type), device_context_(context) {
            if (!device_context_) {
                throw std::invalid_argument("Device context must not be null.");
            }

            // Optional: Validate device context matches template parameter
            if (device_context_->getDevice()->getDeviceType() != TDeviceType) {
                throw std::runtime_error("Device context type mismatch with template parameter.");
            }
        }

    public:
        /**
         * @brief Virtual destructor for the OperationBase class.
         *
         * @details Ensures proper cleanup of derived class resources when destroyed through
         * a base class pointer. Default implementation is sufficient for this base class.
         */
        virtual ~OperationBase() = default;

        /**
         * @brief Gets the name of the operation.
         *
         * @details This pure virtual function must be implemented by derived classes to return
         * a unique identifier string for the specific operation type. The name should be
         * descriptive and consistent across framework components.
         *
         * @return std::string A unique name identifying this operation type.
         *
         * Example:
         * @code
         * std::string getName() const override {
         *     return "MatrixMultiply";
         * }
         * @endcode
         */
        virtual std::string getName() const = 0;

        /**
         * @brief Gets the device context associated with this operation.
         *
         * @details The device context contains information about the execution environment,
         * including the device, streams, and memory resources. This context is used for
         * all device interactions performed by this operation.
         *
         * @return std::shared_ptr<DeviceContext> The device context for this operation (never null).
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Gets the device type for this operation.
         *
         * @details This is a convenience method that retrieves the device type from the
         * associated device context. It delegates to the device context's device to
         * determine the actual hardware target.
         *
         * @return DeviceType The type of device (CPU, CUDA, etc.) for this operation.
         *
         * @note This is the runtime device type; compile-time type is available via device_type constant
         */
        DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

        /**
         * @brief Gets the operation type enumeration value.
         *
         * @details Returns the operation type that was specified during construction.
         * This identifies the category of neural network operation being performed
         * (e.g., Convolution, Pooling, Activation).
         *
         * @return OperationType The enumeration value identifying this operation's category.
         */
        OperationType getOperationType() const {
            return operation_type_;
        }

        /**
         * @brief Gets the abstract data type used by this operation.
         *
         * @details Returns the TensorDataType enumeration value for compile-time and
         * runtime type introspection. Useful for validation, debugging, and dynamic
         * operation dispatch.
         *
         * @return TensorDataType The abstract data type enumeration value.
         *
         * Example:
         * @code
         * auto op = createOperation(...);
         * if (op.getDataType() == TensorDataType::FP16) {
         *     // Handle FP16-specific logic
         * }
         * @endcode
         */
        static constexpr TensorDataType getDataType() {
            return TDataType;
        }

        /**
         * @brief Gets the human-readable name of the data type.
         *
         * @details Returns a string representation of the abstract data type for
         * debugging, logging, and user interface purposes.
         *
         * @return std::string The data type name (e.g., "FP32", "FP16", "INT8").
         *
         * Example:
         * @code
         * std::cout << "Operation data type: " << op.getDataTypeName() << std::endl;
         * // Output: Operation data type: FP32
         * @endcode
         */
        static constexpr std::string_view getDataTypeName() {
            return DataTypeTraits::type_name;
        }

        /**
         * @brief Checks if the operation uses a floating-point data type.
         *
         * @details Compile-time constant indicating whether the operation's data type
         * is a floating-point format (FP32, FP16, BF16, FP8, etc.).
         *
         * @return true if data type is floating-point, false for integer types.
         */
        static constexpr bool isFloatType() {
            return DataTypeTraits::is_float_type;
        }

        /**
         * @brief Checks if the operation uses an integer data type.
         *
         * @details Compile-time constant indicating whether the operation's data type
         * is an integer format (INT8, INT32, UINT8, etc.).
         *
         * @return true if data type is integer, false for floating-point types.
         */
        static constexpr bool isIntegerType() {
            return DataTypeTraits::is_integer_type;
        }

        /**
         * @brief Gets the size in bytes of each data element.
         *
         * @details Returns the storage size per element for the operation's data type.
         * Useful for memory allocation and transfer size calculations.
         *
         * @return size_t Size in bytes per element.
         *
         * Example:
         * @code
         * auto bytes_needed = num_elements * op.getElementSize();
         * @endcode
         */
        static constexpr size_t getElementSize() {
            return DataTypeTraits::size_in_bytes;
        }

    private:
        OperationType operation_type_;                      ///< The operation type identifier.
        std::shared_ptr<DeviceContext> device_context_;     ///< The device context for execution (never null).
    };

    /**
     * @brief Type alias for CPU-based operations with customizable data type.
     *
     * Convenient alias for creating CPU operations with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * class MyCpuOp : public CpuOperationBase<TensorDataType::FP32> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
    using CpuOperationBase = OperationBase<DeviceType::Cpu, TDataType>;

    /**
     * @brief Type alias for CUDA-based operations with customizable data type.
     *
     * Convenient alias for creating CUDA operations with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * class MyCudaOp : public CudaOperationBase<TensorDataType::FP16> {
     *     // Implementation
     * };
     * @endcode
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
    using CudaOperationBase = OperationBase<DeviceType::Cuda, TDataType>;
}