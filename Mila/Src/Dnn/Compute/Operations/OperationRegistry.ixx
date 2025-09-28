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
        std::string variant;                     ///< Variant of the operation implementation
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
        * @brief Type ID structure to uniquely identify operations based on input types and device type.
        *
        * This structure serves as a composite key for the operation registry, enabling
        * precise lookup of operations based on their type parameters and target device.
        */
        struct TypeID {
            std::type_index input1_type;        ///< Type index of the first input tensor element type
            std::type_index input2_type;        ///< Type index of the second input tensor element type (for binary operations)
            std::type_index output_type;        ///< Type index of the output tensor element type
            DeviceType device_type;             ///< Target device type for the operation

            /**
             * @brief Equality comparison operator for TypeID.
             *
             * @param other The TypeID to compare with
             * @return bool True if all fields match, false otherwise
             */
            bool operator==( const TypeID& other ) const {
                return input1_type == other.input1_type &&
                    input2_type == other.input2_type &&
                    output_type == other.output_type &&
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
                std::size_t h1 = std::hash<std::type_index>{}(id.input1_type);
                std::size_t h2 = std::hash<std::type_index>{}(id.input2_type);
                std::size_t h3 = std::hash<std::type_index>{}(id.output_type);
                std::size_t h4 = std::hash<DeviceType>{}(id.device_type);
                return (h1 << 1) ^ (h2 << 2) ^ (h3 << 3) ^ (h4 << 4);
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
         * @brief Register a unary operation creator for a specific device type.
         *
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @tparam TInput The input tensor element type.
         * @tparam TOutput The output tensor element type (defaults to TInput).
         * @param operation_name The name of the operation.
         * @param creator The function that creates the unary operation.
         * @param precision_policy The compute precision policy to use (defaults to Auto).
         */
        template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
            requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
        void registerUnaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>>( std::shared_ptr<DeviceContext>, const ConfigurationBase& )> creator ) {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For unary ops, input2_type is the same as input1_type
                std::type_index( typeid(TOutput) ),
                TDeviceType
            };

            auto genericCreator = [creator]( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<void> {
                return creator( context, config );
                };

            registry_[ type_id ][ operation_name ] = std::move( genericCreator );
        }

        /**
         * @brief Register a binary operation creator for a specific device type.
         *
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @tparam TInput The input tensor element type.
         * @tparam TOutput The output tensor element type (defaults to TInput).
         * @param operation_name The name of the operation.
         * @param creator The function that creates the binary operation.
         * @param precision_policy The compute precision policy to use (defaults to Auto).
         */
        template<DeviceType TDeviceType, typename TInput, typename TOutput>
            requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
        void registerBinaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput>>( std::shared_ptr<DeviceContext>, const ConfigurationBase& )> creator ) {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For binary ops, now both input types are the same
                std::type_index( typeid(TOutput) ),
                TDeviceType
            };

            // Convert BinaryOperation creator to GenericCreator
            auto genericCreator = [creator]( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<void> {
                return creator( context, config );
                };

            registry_[ type_id ][ operation_name ] = std::move( genericCreator );
        }

        /**
         * @brief Register a fused operation.
         *
         * @tparam TOutput The precision type of the operation.
         * @param operation_types The sequence of operation types to fuse.
         * @param fused_op_name The name of the fused operation.
         * @param variant The variant of the operation (defaults to "Default").
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
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @tparam TInput The input tensor element type.
         * @tparam TOutput The output tensor element type (defaults to TInput).
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param precision_policy The compute precision policy to use.
         * @return std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> The created unary operation.
         * @throws std::runtime_error If the type combination, device type, or operation name is invalid.
         * @throws std::invalid_argument If the context is null.
         */
        template<DeviceType TDeviceType, typename TInput, typename TOutput>
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> createUnaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const ConfigurationBase& config ) const {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For unary ops, input2_type is the same as input1_type
                std::type_index( typeid(TOutput) ),
                TDeviceType
            };

            // Find the operation in the registry
            auto type_it = registry_.find( type_id );
            if ( type_it == registry_.end() ) {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: No operations registered for types, Input:{}, Output:{}, Device:{}",
                    typeid(TInput).name(), typeid(TOutput).name(),
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
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

            auto op = std::static_pointer_cast<UnaryOperation<TDeviceType, TInput, TOutput>>(op_it->second( context, config ));

            return op;
        }

        /**
         * @brief Create a binary operation based on the type information, device type, and operation name.
         *
         * @tparam TDeviceType The device type for the operation (defaults to CUDA).
         * @tparam TInput The input tensor element type.
         * @tparam TOutput The output tensor element type (defaults to TInput).
         * @param operation_name The name of the operation.
         * @param context The device context to use for the operation.
         * @param precision_policy The compute precision policy to use.
         * @return std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput>> The created binary operation.
         * @throws std::runtime_error If the type combination, device type, or operation name is invalid.
         * @throws std::invalid_argument If the context is null.
         */
        template<DeviceType TDeviceType, typename TInput, typename TOutput>
            requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput>> createBinaryOperation(
            const std::string& operation_name,
            std::shared_ptr<DeviceContext> context,
            const ConfigurationBase& config ) const {

            TypeID type_id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TInput) ),  // For binary ops, now both input types are the same
                std::type_index( typeid(TOutput) ),
                TDeviceType
            };

            auto type_it = registry_.find( type_id );
            if ( type_it == registry_.end() ) {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: No operations registered for types, Input:{}, Output:{}, Device:{}",
                    typeid(TInput).name(), typeid(TOutput).name(),
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA"
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

            auto op = std::static_pointer_cast<BinaryOperation<TDeviceType, TInput, TOutput>>(op_it->second( context, config ));

            return op;
        }

        /**
         * @brief Find a fused operation match for a sequence of module types.
         *
         * @param child_types The sequence of module types to match.
         * @param device_type The device type.
         * @param precision_type The precision type.
         * @param variant The variant of the operation (defaults to "Default").
         * @return std::optional<FusedOpMeta> The matched fused operation metadata if found, or nullopt if no match.
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
         */
        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        /**
         * @brief Default constructor.
         */
        OperationRegistry() = default;
    };
}