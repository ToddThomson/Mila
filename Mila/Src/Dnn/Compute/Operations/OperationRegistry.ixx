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
         * @brief Type ID structure to uniquely identify operations based on input type, output type, and device type.
         *
         * This structure serves as a composite key for the operation registry, enabling
         * precise lookup of operations based on their type parameters and target device.
         */
        struct TypeID {
            std::type_index input_type;    ///< Type index of the input tensor element type
            std::type_index output_type;   ///< Type index of the output tensor element type
            DeviceType device_type;        ///< Target device type for the operation
            std::string variant;           ///< Variant name for specialized implementations

            /**
             * @brief Equality comparison operator for TypeID.
             *
             * @param other The TypeID to compare with
             * @return bool True if all fields match, false otherwise
             */
            bool operator==( const TypeID& other ) const {
                return input_type == other.input_type &&
                    output_type == other.output_type &&
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
                std::size_t h1 = std::hash<std::type_index>{}(id.input_type);
                std::size_t h2 = std::hash<std::type_index>{}(id.output_type);
                std::size_t h3 = std::hash<DeviceType>{}(id.device_type);
                std::size_t h4 = std::hash<std::string>{}(id.variant);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
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
         * @brief Register an operation creator for a specific device type.
         *
         * This method registers a factory function for creating operations with specific
         * type parameters and for a specific device type. Operations are identified by name
         * and can have multiple variants for specialized implementations.
         *
         * @tparam TInput The input tensor element type.
         * @tparam TPrecision The output tensor element type.
         * @tparam TDeviceType The device type for the operation.
         * @param operation_name The name of the operation.
         * @param variant The variant of the operation.
         * @param creator The function that creates the operation with a device context.
         *
         * @note The creator function should return a valid operation instance when called with
         *       a device context. The operation will be owned by a shared_ptr.
         */
        template<typename TInput, typename TPrecision, DeviceType TDeviceType>
            requires ValidTensorTypes<TInput, TPrecision>
        void registerOperation(
            const std::string& operation_name,
            const std::string& variant,
            std::function<std::shared_ptr<OperationBase<TInput, TPrecision, TDeviceType>>( std::shared_ptr<DeviceContext> )> creator ) {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TPrecision) ),
                TDeviceType,
                variant
            };

            registry_[ type_id ][ operation_name ] = std::move( creator );
        }

        /**
         * @brief Register a fused operation.
         *
         * This method registers metadata for a fused operation, which combines multiple
         * standard operations into a single optimized implementation. Fused operations
         * can provide better performance by reducing memory traffic and enabling more
         * aggressive compiler optimizations.
         *
         * @tparam TPrecision The precision type of the operation.
         * @param operation_types The sequence of operation types to fuse.
         * @param fused_op_name The name of the fused operation.
         * @param variant The variant of the operation (defaults to "Default").
         *
         * @note Fused operations must be registered with their metadata before they can be
         *       discovered during module execution. The actual implementation should be
         *       registered separately using registerOperation().
         */
        template<typename TPrecision>
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
                std::type_index( typeid(TPrecision) ),
                variant
            };

            fused_ops_.push_back( meta );
        }

        /**
         * @brief Create an operation based on the type information, device type, and operation name.
         *
         * This method creates an operation instance with the specified parameters. It looks up
         * the appropriate factory function in the registry and calls it with the provided device context.
         * If a variant-specific implementation is not found, it falls back to the default variant.
         *
         * @tparam TInput The input tensor element type.
         * @tparam TPrecision The output tensor element type.
         * @tparam TDeviceType The device type for the operation.
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param variant The variant of the operation (defaults to "Default").
         * @return std::shared_ptr<OperationBase<TInput, TPrecision, TDeviceType>> The created operation.
         * @throws std::runtime_error If the type combination, device type, or operation name is invalid.
         * @throws std::invalid_argument If the context is null.
         */
        template<typename TInput, typename TPrecision, DeviceType TDeviceType>
            requires ValidTensorTypes<TInput, TPrecision>
        std::shared_ptr<OperationBase<TInput, TPrecision, TDeviceType>> createOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const std::string& variant = "Default" ) const {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TPrecision) ),
                TDeviceType,
                variant
            };

            auto type_it = registry_.find( type_id );
            if ( type_it == registry_.end() ) {
                // If variant-specific operation not found, try with default variant as fallback
                if ( variant != "Default" ) {
                    try {
                        return createOperation<TInput, TPrecision, TDeviceType>( operation_name, context, "Default" );
                    }
                    catch ( const std::runtime_error& ) {
                        // Fall through to the original error if the fallback also fails
                    }
                }

                throw std::runtime_error( std::format(
                    "createOperation: No operations registered for types, Input:{}, Output:{}, Device:{}, Variant:{}",
                    typeid(TInput).name(), typeid(TPrecision).name(),
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA", variant
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if ( op_it == type_it->second.end() ) {
                throw std::runtime_error( std::format(
                    "createOperation: Operation not found: {}",
                    operation_name
                ) );
            }

            if ( !context ) {
                throw std::invalid_argument( "DeviceContext cannot be null when creating an operation" );
            }

            return std::static_pointer_cast<OperationBase<TInput, TPrecision, TDeviceType>>(op_it->second( context ));
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