/**
 * @file OperationRegistry.ixx
 * @brief Comprehensive registry for neural network operations using abstract tensor data types
 *
 * This file implements a registry system that manages neural network operations across
 * different device types (CPU, CUDA) using abstract TensorDataType enumeration. It enables:
 *
 * - Type-safe registration of operation implementations using abstract data types
 * - Creation of operations based on abstract type information and device contexts
 * - Support for operation variants with automatic fallback to default implementations
 * - Registration and discovery of fused operations for performance optimization
 * - Compile-time validation of operation compatibility
 *
 * Key architectural features:
 * - Abstract data type system prevents device-specific compilation issues
 * - Single data type per operation (simplified from separate input/output types)
 * - Device-agnostic operation lookup and creation
 * - Type-safe operation dispatch with runtime validation
 *
 * The OperationRegistry serves as the central hub for operation management in the compute
 * framework, allowing the system to dynamically select the appropriate implementation
 * based on the current context and requirements.
 */

module;
#include <string>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

export module Compute.OperationRegistry;

import Dnn.TensorDataType;
import Dnn.TensorTypeTraits;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.BinaryOperation;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.CudaDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Metadata for fused operations in the neural network.
     *
     * This structure contains information about fused operations, which combine
     * multiple standard operations into a single optimized implementation.
     * Fused operations typically provide better performance by reducing memory
     * traffic and enabling more aggressive compiler optimizations.
     */
    struct FusedOpMeta {
        std::vector<std::string> module_types;   ///< List of module types that make up this fused operation
        std::string fused_op_name;               ///< Name of the fused operation
        TensorDataType data_type;                ///< Abstract data type used by the fused operation
        std::string variant;                     ///< Variant of the operation implementation
        DeviceType device_type;                  ///< Target device type for the fused operation
    };

    /**
     * @brief A registry for operations using abstract tensor data types
     *
     * This singleton class manages the registration and creation of neural network operations
     * in the Mila framework using abstract TensorDataType enumeration. It provides a unified
     * interface to access different operation implementations across various device types
     * (CPU, CUDA) with type-safe operation lookup and creation.
     *
     * The registry supports:
     * - Type-safe registration and lookup using abstract data types
     * - Operation variants for specialized implementations
     * - Fused operations for performance optimization
     * - Automatic device-specific operation selection
     * - Compile-time type validation
     *
     * Design changes from previous version:
     * - Uses TensorDataType enumeration instead of concrete C++ types
     * - Single data type per operation (operations use same type for input/output)
     * - Simplified TypeID structure using abstract types
     * - Aligned with updated UnaryOperation and BinaryOperation templates
     */
    export class OperationRegistry {
    public:
        /**
         * @brief Type ID structure to uniquely identify operations using abstract types
         *
         * This structure serves as a composite key for the operation registry, enabling
         * precise lookup of operations based on their abstract data type and target device.
         * Uses abstract TensorDataType enumeration instead of std::type_index for better
         * type safety and device independence.
         */
        struct TypeID {
            TensorDataType data_type;           ///< Abstract tensor data type
            DeviceType device_type;             ///< Target device type for the operation

            /**
             * @brief Equality comparison operator for TypeID.
             *
             * @param other The TypeID to compare with
             * @return bool True if all fields match, false otherwise
             */
            bool operator==( const TypeID& other ) const {
                return data_type == other.data_type &&
                    device_type == other.device_type;
            }
        };

        /**
         * @brief Hash function for TypeID to use in unordered_map.
         */
        struct TypeIDHash {
            /**
             * @brief Calculate a hash value for a TypeID.
             *
             * @param id The TypeID to hash
             * @return std::size_t The calculated hash value
             */
            std::size_t operator()( const TypeID& id ) const {
                std::size_t h1 = std::hash<TensorDataType>{}(id.data_type);
                std::size_t h2 = std::hash<DeviceType>{}(id.device_type);
                return h1 ^ (h2 << 1);
            }
        };

        /**
         * @brief Get the singleton instance of the OperationRegistry.
         *
         * @return OperationRegistry& The singleton instance.
         */
        static OperationRegistry& instance() {
            static OperationRegistry registry;
            return registry;
        }

        /**
         * @brief Register a unary operation creator for a specific device type and data type
         *
         * Registers a factory function that creates unary operations with the specified
         * device type and abstract data type. The operation can be later created using
         * the createUnaryOperation method with matching type parameters.
         *
         * @tparam TDeviceType The device type for the operation (CPU or CUDA).
         * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
         * @param operation_name The name of the operation (e.g., "ReLU", "GELU").
         * @param creator The function that creates the unary operation.
         *
         * @note Operations use single data type for both input and output
         * @note Creator function receives DeviceContext and ConfigurationBase
         *
         * Example:
         * @code
         * registry.registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
         *     "ReLU",
         *     [](auto ctx, auto& config) {
         *         return std::make_shared<CudaReLUOp<TensorDataType::FP32>>(ctx, config);
         *     }
         * );
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        void registerUnaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<UnaryOperation<TDeviceType, TDataType>>(
                std::shared_ptr<DeviceContext>,
                const ConfigurationBase& )> creator ) {

            TypeID type_id{
                TDataType,
                TDeviceType
            };

            auto genericCreator = [creator](
                std::shared_ptr<DeviceContext> context,
                const ConfigurationBase& config ) -> std::shared_ptr<void> {
                    return creator( context, config );
                };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Register a binary operation creator for a specific device type and data type
         *
         * Registers a factory function that creates binary operations with the specified
         * device type and abstract data type. The operation can be later created using
         * the createBinaryOperation method with matching type parameters.
         *
         * @tparam TDeviceType The device type for the operation (CPU or CUDA).
         * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
         * @param operation_name The name of the operation (e.g., "Add", "MatMul").
         * @param creator The function that creates the binary operation.
         *
         * @note Operations use single data type for both inputs and output
         * @note Creator function receives DeviceContext and ConfigurationBase
         *
         * Example:
         * @code
         * registry.registerBinaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
         *     "MatMul",
         *     [](auto ctx, auto& config) {
         *         return std::make_shared<CudaMatMulOp<TensorDataType::FP16>>(ctx, config);
         *     }
         * );
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        void registerBinaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<BinaryOperation<TDeviceType, TDataType>>(
                std::shared_ptr<DeviceContext>,
                const ConfigurationBase& )> creator ) {

            TypeID type_id{
                TDataType,
                TDeviceType
            };

            // Convert BinaryOperation creator to GenericCreator
            auto genericCreator = [creator](
                std::shared_ptr<DeviceContext> context,
                const ConfigurationBase& config ) -> std::shared_ptr<void> {
                    return creator( context, config );
                };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Register a fused operation.
         *
         * Registers metadata for a fused operation that combines multiple standard
         * operations into a single optimized implementation. Fused operations can
         * be discovered using findFusedMatch based on the sequence of operations.
         *
         * @param operation_types The sequence of operation types to fuse.
         * @param fused_op_name The name of the fused operation.
         * @param data_type The abstract data type used by the fused operation.
         * @param device_type The target device type for the fused operation.
         * @param variant The variant of the operation (defaults to "Default").
         *
         * Example:
         * @code
         * registry.registerFusedOperation(
         *     {OperationType::MatMul, OperationType::Add, OperationType::ReLU},
         *     "FusedLinearReLU",
         *     TensorDataType::FP16,
         *     DeviceType::Cuda,
         *     "Optimized"
         * );
         * @endcode
         */
        void registerFusedOperation(
            const std::vector<OperationType>& operation_types,
            const std::string& fused_op_name,
            TensorDataType data_type,
            DeviceType device_type,
            const std::string& variant = "Default" ) {

            // Convert OperationType values to strings
            std::vector<std::string> module_types;
            for (const auto& op_type : operation_types) {
                // Map operation types to module names
                std::string module_name = operationTypeToString( op_type );

                // Convert "Op" suffix to "Module" suffix if needed
                if (module_name.length() > 2 && module_name.substr( module_name.length() - 2 ) == "Op") {
                    module_name = module_name.substr( 0, module_name.length() - 2 ) + "Module";
                }

                module_types.push_back( module_name );
            }

            FusedOpMeta meta{
                module_types,
                fused_op_name,
                data_type,
                variant,
                device_type
            };

            fused_ops_.push_back( meta );
        }

        /**
         * @brief Create a unary operation based on abstract type information
         *
         * Creates a unary operation instance using the registry's factory functions.
         * The operation is looked up based on the abstract data type, device type,
         * and operation name. Validates that all type information matches registered
         * operation.
         *
         * @tparam TDeviceType The device type for the operation (CPU or CUDA).
         * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param config Configuration for the operation.
         * @return std::shared_ptr<UnaryOperation<TDeviceType, TDataType>> The created unary operation.
         *
         * @throws std::runtime_error If no operation is registered for the type combination.
         * @throws std::runtime_error If the operation name is not found.
         * @throws std::invalid_argument If the context is null.
         *
         * Example:
         * @code
         * auto relu = registry.createUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
         *     "ReLU",
         *     cuda_context,
         *     config
         * );
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::shared_ptr<UnaryOperation<TDeviceType, TDataType>> createUnaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const ConfigurationBase& config ) const {

            TypeID type_id{
                TDataType,
                TDeviceType
            };

            // Find the operation in the registry
            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: No operations registered for DataType: {}, Device: {}",
                    TensorDataTypeTraits<TDataType>::type_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if (op_it == type_it->second.end()) {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: Operation '{}' not found for DataType: {}, Device: {}",
                    operation_name,
                    TensorDataTypeTraits<TDataType>::type_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
                ) );
            }

            if (!context) {
                throw std::invalid_argument( "DeviceContext cannot be null when creating an operation" );
            }

            auto op = std::static_pointer_cast<UnaryOperation<TDeviceType, TDataType>>(
                op_it->second( context, config ));

            return op;
        }

        /**
         * @brief Create a binary operation based on abstract type information
         *
         * Creates a binary operation instance using the registry's factory functions.
         * The operation is looked up based on the abstract data type, device type,
         * and operation name. Validates that all type information matches registered
         * operation.
         *
         * @tparam TDeviceType The device type for the operation (CPU or CUDA).
         * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param config Configuration for the operation.
         * @return std::shared_ptr<BinaryOperation<TDeviceType, TDataType>> The created binary operation.
         *
         * @throws std::runtime_error If no operation is registered for the type combination.
         * @throws std::runtime_error If the operation name is not found.
         * @throws std::invalid_argument If the context is null.
         *
         * Example:
         * @code
         * auto matmul = registry.createBinaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
         *     "MatMul",
         *     cuda_context,
         *     config
         * );
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::shared_ptr<BinaryOperation<TDeviceType, TDataType>> createBinaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const ConfigurationBase& config ) const {

            TypeID type_id{
                TDataType,
                TDeviceType
            };

            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: No operations registered for DataType: {}, Device: {}",
                    TensorDataTypeTraits<TDataType>::type_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if (op_it == type_it->second.end()) {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: Operation '{}' not found for DataType: {}, Device: {}",
                    operation_name,
                    TensorDataTypeTraits<TDataType>::type_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
                ) );
            }

            if (!context) {
                throw std::invalid_argument( "DeviceContext cannot be null when creating an operation" );
            }

            auto op = std::static_pointer_cast<BinaryOperation<TDeviceType, TDataType>>(
                op_it->second( context, config ));

            return op;
        }

        /**
         * @brief Find a fused operation match for a sequence of module types
         *
         * Searches the registered fused operations for a match with the given sequence
         * of module types. Returns the fused operation metadata if a match is found,
         * allowing the caller to use the optimized fused implementation instead of
         * individual operations.
         *
         * @param child_types The sequence of module types to match.
         * @param device_type The target device type.
         * @param data_type The abstract data type.
         * @param variant The variant of the operation (defaults to "Default").
         * @return std::optional<FusedOpMeta> The matched fused operation metadata if found, or nullopt.
         *
         * @note Supports partial matches (fused op can be shorter than child_types)
         * @note Prefers exact variant matches but falls back to "Default" variant
         * @note Uses sliding window search through child_types sequence
         *
         * Example:
         * @code
         * auto fused = registry.findFusedMatch(
         *     {"MatMulModule", "AddModule", "ReLUModule"},
         *     DeviceType::Cuda,
         *     TensorDataType::FP16
         * );
         * if (fused) {
         *     // Use fused operation: fused->fused_op_name
         * }
         * @endcode
         */
        std::optional<FusedOpMeta> findFusedMatch(
            const std::vector<std::string>& child_types,
            DeviceType device_type,
            TensorDataType data_type,
            const std::string& variant = "Default" ) {

            for (const auto& entry : fused_ops_) {
                if (entry.data_type != data_type) continue;
                if (entry.device_type != device_type) continue;
                if (entry.variant != variant && entry.variant != "Default") continue;
                if (entry.module_types.size() > child_types.size()) continue;

                // Sliding window search for matching sequence
                for (size_t i = 0; i <= child_types.size() - entry.module_types.size(); ++i) {
                    if (std::equal(
                        entry.module_types.begin(), entry.module_types.end(),
                        child_types.begin() + i )) {
                        return entry;  // match found
                    }
                }
            }
            return std::nullopt;
        }

        /**
         * @brief Get list of all registered operation names for a given type configuration
         *
         * Returns all operation names registered for the specified device type and
         * data type. Useful for introspection, debugging, and validation.
         *
         * @tparam TDeviceType The device type to query.
         * @tparam TDataType The abstract data type to query.
         * @return std::vector<std::string> List of registered operation names.
         *
         * Example:
         * @code
         * auto ops = registry.getRegisteredOperations<DeviceType::Cuda, TensorDataType::FP32>();
         * for (const auto& op_name : ops) {
         *     std::cout << "Registered: " << op_name << std::endl;
         * }
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::vector<std::string> getRegisteredOperations() const {
            TypeID type_id{
                TDataType,
                TDeviceType
            };

            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) {
                return {};
            }

            std::vector<std::string> operations;
            operations.reserve( type_it->second.size() );
            for (const auto& [name, _] : type_it->second) {
                operations.push_back( name );
            }
            return operations;
        }

        /**
         * @brief Check if an operation is registered for given type configuration
         *
         * Queries whether a specific operation is available for the given device type,
         * data type, and operation name. Useful for feature detection and validation.
         *
         * @tparam TDeviceType The device type to check.
         * @tparam TDataType The abstract data type to check.
         * @param operation_name The name of the operation to check.
         * @return bool True if operation is registered, false otherwise.
         *
         * Example:
         * @code
         * if (registry.isOperationRegistered<DeviceType::Cuda, TensorDataType::FP16>("GELU")) {
         *     // Use FP16 GELU
         * } else {
         *     // Fallback to FP32 or different implementation
         * }
         * @endcode
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        bool isOperationRegistered( const std::string& operation_name ) const {
            TypeID type_id{
                TDataType,
                TDeviceType
            };

            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) {
                return false;
            }

            return type_it->second.find( operation_name ) != type_it->second.end();
        }

    private:
        /**
         * @brief Type alias for a generic operation creator function.
         *
         * Uses type erasure to store operation creators of different types
         * in the same registry map. Creator returns void* which is cast to
         * the appropriate operation type during creation.
         */
        using GenericCreator = std::function<std::shared_ptr<void>(
            std::shared_ptr<DeviceContext>,
            const ConfigurationBase& )>;

        /**
         * @brief List of registered fused operations.
         */
        std::vector<FusedOpMeta> fused_ops_;

        /**
         * @brief Registry mapping type info to operations.
         *
         * Maps TypeID (data type + device type) to a map of operation names
         * and their creator functions. This enables efficient lookup of
         * operations based on type requirements.
         */
        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        /**
         * @brief Private constructor for singleton pattern.
         */
        OperationRegistry() = default;

        /**
         * @brief Deleted copy constructor.
         */
        OperationRegistry( const OperationRegistry& ) = delete;

        /**
         * @brief Deleted assignment operator.
         */
        OperationRegistry& operator=( const OperationRegistry& ) = delete;
    };
}