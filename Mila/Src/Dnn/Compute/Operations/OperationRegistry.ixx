/**
 * @file OperationRegistry.ixx
 * @brief Provides a registry for neural network operations in the Mila framework.
 *
 * This file implements a comprehensive registry system that manages neural network operations
 * across different device types (CPU, CUDA) and with various data types. It enables:
 *
 * - Registration of operation implementations for specific device types and data types
 * - Creation of operations based on type information and device contexts
 * - Support for operation variants with automatic fallback to default implementations
 * - Registration and discovery of fused operations for performance optimization
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
#include <typeindex>
#include <vector>

export module Compute.OperationRegistry;

import Dnn.TensorTraits;
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
import Compute.CudaMemoryResource;

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
        std::type_index precision_type;          ///< Precision type used by the fused operation
        std::string variant;                     ///< Variant name of the fused operation implementation
    };

    /**
     * @brief A registry for operations that can be created based on operation names, type information, and device type.
     *
     * This singleton class manages the registration and creation of neural network operations
     * in the Mila framework. It provides a unified interface to access different operation
     * implementations across various device types (CPU, CUDA) and with different data types.
     *
     * The registry supports:
     * - Type-safe registration and lookup of operations
     * - Operation variants for specialized implementations
     * - Fused operations for performance optimization
     * - Automatic device-specific operation selection
     */
    export class OperationRegistry {
    public:
        /**
        * @brief Type ID structure to uniquely identify operations based on precision type, input types, and device type.
        *
        * This structure serves as a composite key for the operation registry, enabling
        * precise lookup of operations based on their type parameters and target device.
        */
        struct TypeID {
            std::type_index precision_type;  ///< Type index of the precision/output tensor element type
            std::type_index input1_type;     ///< Type index of the first input tensor element type
            std::type_index input2_type;     ///< Type index of the second input tensor element type (for binary operations)
            DeviceType device_type;          ///< Target device type for the operation
            std::string variant;             ///< Variant name for specialized implementations

            /**
             * @brief Equality comparison operator for TypeID.
             *
             * @param other The TypeID to compare with
             * @return bool True if all fields match, false otherwise
             */
            bool operator==( const TypeID& other ) const {
                return precision_type == other.precision_type &&
                    input1_type == other.input1_type &&
                    input2_type == other.input2_type &&
                    device_type == other.device_type &&
                    variant == other.variant;
            }
        };

        /**
 * @brief Hash function for TypeID to use in unordered_map.
 *
 * This functor provides a hash function for TypeID instances, enabling their
 * use as keys in unordered_map containers. The hash combines the hashes of
 * each field with bit shifts to reduce collisions.
 */
        struct TypeIDHash {
            /**
             * @brief Calculate a hash value for a TypeID.
             *
             * @param id The TypeID to hash
             * @return std::size_t The calculated hash value
             */
            std::size_t operator()( const TypeID& id ) const {
                std::size_t h1 = std::hash<std::type_index>{}(id.precision_type);
                std::size_t h2 = std::hash<std::type_index>{}(id.input1_type);
                std::size_t h3 = std::hash<std::type_index>{}(id.input2_type);
                std::size_t h4 = std::hash<DeviceType>{}(id.device_type);
                std::size_t h5 = std::hash<std::string>{}(id.variant);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
            }
        };

        /**
         * @brief Get the singleton instance of the OperationRegistry.
         *
         * This method provides access to the singleton instance of the OperationRegistry.
         * It ensures that only one registry exists throughout the application's lifecycle.
         *
         * @return OperationRegistry& The singleton instance.
         */
        static OperationRegistry& instance() {
            static OperationRegistry registry;
            return registry;
        }

        /**
         * @brief Register a unary operation creator for a specific device type.
         *
         * This method registers a factory function for unary operations with specific
         * type parameters and for a specific device type.
         *
         * @tparam TOutput The precision used for computation (typically float or half).
         * @tparam TInput The input tensor element type (defaults to TOutput).
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @param operation_name The name of the operation.
         * @param variant The implementation variant name.
         * @param creator The function that creates the unary operation.
         */
        template<typename TOutput, typename TInput = TOutput, DeviceType TDeviceType = DeviceType::Cuda>
            requires ValidFloatTensorType<TOutput> && ValidTensorType<TInput>
        void registerUnaryOperation(
            const std::string& operation_name,
            const std::string& variant,
            std::function<std::shared_ptr<UnaryOperation<TOutput, TInput, TDeviceType>>( std::shared_ptr<DeviceContext> )> creator ) {

            TypeID type_id{
                std::type_index( typeid(TOutput) ),
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For unary ops, input2_type is the same as input1_type
                TDeviceType,
                variant
            };

            // Convert UnaryOperation creator to GenericCreator
            auto genericCreator = [creator]( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<void> {
                return creator( context );
                };

            registry_[ type_id ][ operation_name ] = std::move( genericCreator );
        }

        /**
         * @brief Register a binary operation creator for a specific device type.
         *
         * This method registers a factory function for binary operations with specific
         * type parameters and for a specific device type.
         *
         * @tparam TOutput The precision used for computation (typically float or half).
         * @tparam TInput The input tensor element type (defaults to TOutput).
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @param operation_name The name of the operation.
         * @param variant The implementation variant name.
         * @param creator The function that creates the binary operation.
         */
        template<typename TOutput, typename TInput1 = TOutput, typename TInput2 = TInput1, DeviceType TDeviceType = DeviceType::Cuda>
            requires ValidFloatTensorType<TOutput>&& ValidTensorTypes<TInput1, TInput2>
        void registerBinaryOperation(
            const std::string& operation_name,
            const std::string& variant,
            std::function<std::shared_ptr<BinaryOperation<TOutput, TInput1, TInput2, TDeviceType>>( std::shared_ptr<DeviceContext> )> creator ) {

            TypeID type_id{
                std::type_index( typeid(TOutput) ),
                std::type_index( typeid(TInput1) ),
                std::type_index( typeid(TInput2) ),
                TDeviceType,
                variant
            };

            // Convert BinaryOperation creator to GenericCreator
            auto genericCreator = [creator]( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<void> {
                return creator( context );
                };

            registry_[ type_id ][ operation_name ] = std::move( genericCreator );
        }

        /**
         * @brief Register a fused operation.
         *
         * This method registers metadata for a fused operation, which combines multiple
         * standard operations into a single optimized implementation. Fused operations
         * can provide better performance by reducing memory traffic and enabling more
         * aggressive compiler optimizations.
         *
         * @tparam TDataType The precision type of the operation.
         * @param operation_types The sequence of operation types to fuse.
         * @param fused_op_name The name of the fused operation.
         * @param variant The variant of the operation (defaults to "Default").
         *
         * @note Fused operations must be registered with their metadata before they can be
         *       discovered during module execution. The actual implementation should be
         *       registered separately using registerUnaryOperation or registerBinaryOperation.
         */
        template<typename TOutput>
        void registerFusedOperation(
            const std::vector<OperationType>& operation_types,
            const std::string& fused_op_name,
            const std::string& variant = "Default" ) {

            // Convert OperationType values to strings
            std::vector<std::string> module_types;
            for ( const auto& op_type : operation_types ) {
                // Map operation types to module names
                std::string module_name = operationTypeToString( op_type );

                // Convert "Op" suffix to "Module" suffix if needed
                if ( module_name.length() > 2 && module_name.substr( module_name.length() - 2 ) == "Op" ) {
                    module_name = module_name.substr( 0, module_name.length() - 2 ) + "Module";
                }

                module_types.push_back( module_name );
            }

            FusedOpMeta meta{
                module_types,
                fused_op_name,
                std::type_index( typeid(TOutput) ),
                variant
            };

            fused_ops_.push_back( meta );
        }

        /**
         * @brief Create a unary operation based on the type information, device type, and operation name.
         *
         * This method creates a unary operation instance with the specified parameters.
         * If a variant-specific implementation is not found, it falls back to the default variant.
         *
         * @tparam TOutput The precision used for computation (typically float or half).
         * @tparam TInput The input tensor element type (defaults to TOutput).
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param variant The implementation variant name (defaults to "Default").
         * @return std::shared_ptr<UnaryOperation<TOutput, TInput, TDeviceType>> The created unary operation.
         * @throws std::runtime_error If the type combination, device type, or operation name is invalid.
         * @throws std::invalid_argument If the context is null.
         */
        template<typename TOutput, typename TInput = TOutput, DeviceType TDeviceType = DeviceType::Cuda>
            requires ValidFloatTensorType<TOutput> && ValidTensorType<TInput>
        std::shared_ptr<UnaryOperation<TOutput, TInput, TDeviceType>> createUnaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const std::string& variant = "Default" ) const {

            TypeID type_id{
                std::type_index( typeid(TOutput) ),
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For unary ops, input2_type is the same as input1_type
                TDeviceType,
                variant
            };

            // Find the operation in the registry
            auto type_it = registry_.find( type_id );
            if ( type_it == registry_.end() ) {
                // If variant-specific operation not found, try with default variant as fallback
                if ( variant != "Default" ) {
                    try {
                        return createUnaryOperation<TOutput, TInput, TDeviceType>( operation_name, context, "Default" );
                    }
                    catch ( const std::runtime_error& ) {
                        // Fall through to the original error if the fallback also fails
                    }
                }

                throw std::runtime_error( std::format(
                    "createUnaryOperation: No operations registered for types, Input:{}, Output:{}, Device:{}, Variant:{}",
                    typeid(TInput).name(), typeid(TOutput).name(),
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA", variant
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if ( op_it == type_it->second.end() ) {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: Operation not found: {}",
                    operation_name
                ) );
            }

            if ( !context ) {
                throw std::invalid_argument( "DeviceContext cannot be null when creating an operation" );
            }

            return std::static_pointer_cast<UnaryOperation<TOutput, TInput, TDeviceType>>(op_it->second( context ));
        }

        /**
         * @brief Create a binary operation based on the type information, device type, and operation name.
         *
         * This method creates a binary operation instance with the specified parameters.
         * If a variant-specific implementation is not found, it falls back to the default variant.
         *
         * @tparam TOutput The precision used for computation (typically float or half).
         * @tparam TInput The input tensor element type (defaults to TOutput).
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param variant The implementation variant name (defaults to "Default").
         * @return std::shared_ptr<BinaryOperation<TOutput, TInput, TDeviceType>> The created binary operation.
         * @throws std::runtime_error If the type combination, device type, or operation name is invalid.
         * @throws std::invalid_argument If the context is null.
         */
        template<typename TOutput, typename TInput1 = TOutput, typename TInput2 = TInput1, DeviceType TDeviceType = DeviceType::Cuda>
            requires ValidFloatTensorType<TOutput>&& ValidTensorTypes<TInput1, TInput2>
        std::shared_ptr<BinaryOperation<TOutput, TInput1, TInput2, TDeviceType>> createBinaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const std::string& variant = "Default" ) const {

            TypeID type_id{
                std::type_index( typeid(TOutput) ),
                std::type_index( typeid(TInput1) ),
                std::type_index( typeid(TInput2) ),
                TDeviceType,
                variant
            };

            auto type_it = registry_.find( type_id );
            if ( type_it == registry_.end() ) {
                // If variant-specific operation not found, try with default variant as fallback
                if ( variant != "Default" ) {
                    try {
                        return createBinaryOperation<TOutput, TInput1, TInput2, TDeviceType>( operation_name, context, "Default" );
                    }
                    catch ( const std::runtime_error& ) {
                        // Fall through to the original error if the fallback also fails
                    }
                }

                throw std::runtime_error( std::format(
                    "createBinaryOperation: No operations registered for types, Precision: {}, Input1:{}, Input2:{}, Device:{}, Variant:{}",
                    typeid(TOutput).name(), typeid(TInput1).name(), typeid(TInput2).name(),
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA", variant
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if ( op_it == type_it->second.end() ) {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: Operation not found: {}",
                    operation_name
                ) );
            }

            if ( !context ) {
                throw std::invalid_argument( "DeviceContext cannot be null when creating an operation" );
            }

            return std::static_pointer_cast<BinaryOperation<TOutput, TInput1, TInput2, TDeviceType>>(op_it->second( context ));
        }

        /**
         * @brief Find a fused operation match for a sequence of module types.
         *
         * This method searches for a registered fused operation that matches a given sequence
         * of module types. It can be used to identify opportunities for using optimized fused
         * implementations during module execution.
         *
         * @param child_types The sequence of module types to match.
         * @param device_type The device type.
         * @param precision_type The precision type.
         * @param variant The variant of the operation (defaults to "Default").
         * @return std::optional<FusedOpMeta> The matched fused operation metadata if found, or nullopt if no match.
         *
         * @note The method performs a sliding window search to find matches anywhere within
         *       the provided sequence of module types.
         */
        std::optional<FusedOpMeta> findFusedMatch(
            const std::vector<std::string>& child_types,
            DeviceType device_type,
            std::type_index precision_type,
            const std::string& variant = "Default" ) {

            for ( const auto& entry : fused_ops_ ) {
                if ( entry.precision_type != precision_type ) continue;
                if ( entry.variant != variant && entry.variant != "Default" ) continue;
                if ( entry.module_types.size() > child_types.size() ) continue;

                for ( size_t i = 0; i <= child_types.size() - entry.module_types.size(); ++i ) {
                    if ( std::equal(
                        entry.module_types.begin(), entry.module_types.end(),
                        child_types.begin() + i ) ) {
                        return entry;  // match found
                    }
                }
            }
            return std::nullopt;
        }

    private:
        /**
         * @brief Type alias for a generic operation creator function.
         *
         * This function type can create any operation type given a device context.
         * It returns a void shared_ptr that must be cast to the appropriate type.
         */
        using GenericCreator = std::function<std::shared_ptr<void>( std::shared_ptr<DeviceContext> )>;

        /**
         * @brief List of registered fused operations.
         *
         * This vector stores metadata for all registered fused operations.
         */
        std::vector<FusedOpMeta> fused_ops_;

        /**
         * @brief Registry mapping type info to operations.
         *
         * This nested map structure first indexes by TypeID (which includes input type,
         * output type, device type, and variant), then by operation name. The values
         * are factory functions that can create the corresponding operations.
         */
        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        /**
         * @brief Default constructor.
         *
         * Private constructor for the singleton pattern.
         */
        OperationRegistry() = default;
    };
}
