/**
 * @file OperationRegistry.ixx
 * @brief Central registry for creating and discovering compute operations.
 *
 * This module implements a singleton registry that maps (DeviceType, TensorDataType)
 * pairs and operation names to factory functions that construct concrete
 * UnaryOperation and BinaryOperation implementations.
 *
 * Responsibilities:
 * - Allow registration of typed operation creators: creators receive a
 *   std::shared_ptr<ExecutionContext<TDeviceType>> and ConfigurationBase.
 * - Provide creation helpers that return strongly-typed operation instances
 *   (e.g., std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>).
 * - Maintain metadata for fused operations and provide discovery utilities.
 *
 * Notes:
 * - Factories are stored type-erased as GenericCreator accepting
 *   std::shared_ptr<IExecutionContext> so registry can store creators for
 *   varied ExecutionContext<TDeviceType> types in a single container.
 * - The registry remains keyed by TensorDataType to preserve compile-time
 *   type-safety for operation implementations.
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
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.BinaryOperation;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuDevice;
import Compute.CudaDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Metadata for fused operations in the neural network.
     */
    struct FusedOpMeta {
        std::vector<std::string> module_types;   ///< List of module types that make up this fused operation
        std::string fused_op_name;               ///< Name of the fused operation
        TensorDataType data_type;                ///< Abstract data type used by the fused operation
        std::string variant;                     ///< Variant of the operation implementation
        DeviceType device_type;                  ///< Target device type for the fused operation
    };

    /**
     * @brief A registry for operations using abstract tensor data types.
     *
     * Stores factory functions for concrete, typed operations. Factories are
     * registered per (DeviceType, TensorDataType) and operation name.
     */
    export class OperationRegistry {
    public:
        
        /*using UnaryOpFactory = std::function<std::shared_ptr<UnaryOperation<TDeviceType, TDataType>>(
            std::shared_ptr<ExecutionContext<TDeviceType>>,
            const ConfigurationBase& )>;*/
        

        /**
         * @brief Composite key for registry lookup.
         */
        struct TypeID {
            TensorDataType data_type;           ///< Abstract tensor data type
            DeviceType device_type;             ///< Target device type for the operation

            bool operator==( const TypeID& other ) const {
                return data_type == other.data_type &&
                    device_type == other.device_type;
            }
        };

        struct TypeIDHash {
            std::size_t operator()( const TypeID& id ) const {
                std::size_t h1 = std::hash<TensorDataType>{}(id.data_type);
                std::size_t h2 = std::hash<DeviceType>{}(id.device_type);
                return h1 ^ (h2 << 1);
            }
        };

        /**
         * @brief Get the singleton instance of the OperationRegistry.
         */
        static OperationRegistry& instance() {
            static OperationRegistry registry;
            return registry;
        }

        /**
         * @brief Register a unary operation creator for a specific device type and data type.
         *
         * Creator must accept a std::shared_ptr<ExecutionContext<TDeviceType>> and ConfigurationBase
         * and return a std::shared_ptr<UnaryOperation<TDeviceType, TDataType>>.
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        void registerUnaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<UnaryOperation<TDeviceType, TDataType>>(
                const ConfigurationBase&,
                std::shared_ptr<ExecutionContext<TDeviceType>> )> creator ) {

            TypeID type_id{ TDataType, TDeviceType };

            // Wrap typed creator into a generic creator that accepts IExecutionContext
            auto genericCreator = [creator](
                const ConfigurationBase& config,
                std::shared_ptr<IExecutionContext> ictx) -> std::shared_ptr<void> {

                // Safe cast: ExecutionContext<TDeviceType> inherits from IExecutionContext
                auto ctx = std::static_pointer_cast<ExecutionContext<TDeviceType>>(ictx);
                if (!ctx) {
                    throw std::runtime_error( "ExecutionContext cast failed in registry creator." );
                }

                return creator( config, ctx );
            };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Register a binary operation creator for a specific device type and data type.
         *
         * Creator must accept a std::shared_ptr<ExecutionContext<TDeviceType>> and ConfigurationBase
         * and return a std::shared_ptr<BinaryOperation<TDeviceType, TDataType>>.
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        void registerBinaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<BinaryOperation<TDeviceType, TDataType>>(
                std::shared_ptr<ExecutionContext<TDeviceType>>,
                const ConfigurationBase& )> creator ) {

            TypeID type_id{ TDataType, TDeviceType };

            auto genericCreator = [creator](
                std::shared_ptr<IExecutionContext> ictx,
                const ConfigurationBase& config ) -> std::shared_ptr<void> {

                if (!ictx) {
                    throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
                }

                auto ctx = std::static_pointer_cast<ExecutionContext<TDeviceType>>(ictx);
                if (!ctx) {
                    throw std::runtime_error( "ExecutionContext cast failed in registry creator." );
                }

                return creator( ctx, config );
            };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Register a fused operation (metadata only).
         */
        void registerFusedOperation(
            const std::vector<OperationType>& operation_types,
            const std::string& fused_op_name,
            TensorDataType data_type,
            DeviceType device_type,
            const std::string& variant = "Default" ) {

            std::vector<std::string> module_types;
            for (const auto& op_type : operation_types) {
                std::string module_name = operationTypeToString( op_type );
                if (module_name.size() > 2 && module_name.substr(module_name.size() - 2) == "Op") {
                    module_name = module_name.substr(0, module_name.size() - 2) + "Module";
                }
                module_types.push_back( module_name );
            }

            FusedOpMeta meta{ module_types, fused_op_name, data_type, variant, device_type };
            fused_ops_.push_back( meta );
        }

        /**
         * @brief Create a unary operation instance (typed).
         *
         * The caller provides a std::shared_ptr<ExecutionContext<TDeviceType>>; the registry
         * will invoke the stored generic creator with an IExecutionContext wrapper and
         * return a strongly-typed operation pointer.
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::shared_ptr<UnaryOperation<TDeviceType, TDataType>> createUnaryOperation(
            const std::string& operation_name,
            const ConfigurationBase& config,
            std::shared_ptr<ExecutionContext<TDeviceType>> context ) const {

            TypeID type_id{ TDataType, TDeviceType };

            auto type_it = registry_.find( type_id );
            
            if (type_it == registry_.end())
            {
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
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
            }

            // Invoke generic creator with type-erased IExecutionContext and cast result
            auto op = std::static_pointer_cast<UnaryOperation<TDeviceType, TDataType>>(
                op_it->second(  config, std::static_pointer_cast<IExecutionContext>(context) ) );

            return op;
        }

        /**
         * @brief Create a binary operation instance (typed).
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::shared_ptr<BinaryOperation<TDeviceType, TDataType>> createBinaryOperation(
            const std::string& operation_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> context,
            const ConfigurationBase& config ) const {

            TypeID type_id{ TDataType, TDeviceType };

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
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
            }

            auto op = std::static_pointer_cast<BinaryOperation<TDeviceType, TDataType>>(
                op_it->second( std::static_pointer_cast<IExecutionContext>(context), config ) );

            return op;
        }

        /**
         * @brief Find a fused operation match for a sequence of module types.
         */
        std::optional<FusedOpMeta> findFusedMatch(
            const std::vector<std::string>& child_types,
            DeviceType device_type,
            TensorDataType data_type,
            const std::string& variant = "Default" ) const {

            for (const auto& entry : fused_ops_) {
                if (entry.data_type != data_type) continue;
                if (entry.device_type != device_type) continue;
                if (entry.variant != variant && entry.variant != "Default") continue;
                if (entry.module_types.size() > child_types.size()) continue;

                for (size_t i = 0; i <= child_types.size() - entry.module_types.size(); ++i) {
                    if (std::equal(
                        entry.module_types.begin(), entry.module_types.end(),
                        child_types.begin() + i )) {
                        return entry;
                    }
                }
            }
            return std::nullopt;
        }

        /**
         * @brief Get list of registered operation names for a given type configuration.
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        std::vector<std::string> getRegisteredOperations() const {
            TypeID type_id{ TDataType, TDeviceType };
            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) return {};
            std::vector<std::string> operations;
            operations.reserve( type_it->second.size() );
            for (const auto& [name, _] : type_it->second) operations.push_back( name );
            return operations;
        }

        /**
         * @brief Check if an operation is registered for given type configuration.
         */
        template<DeviceType TDeviceType, TensorDataType TDataType>
        bool isOperationRegistered( const std::string& operation_name ) const {
            TypeID type_id{ TDataType, TDeviceType };
            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) return false;
            return type_it->second.find( operation_name ) != type_it->second.end();
        }

    private:
        using GenericCreator = std::function<std::shared_ptr<void>(
            const ConfigurationBase&,
            std::shared_ptr<IExecutionContext> )>;

        std::vector<FusedOpMeta> fused_ops_;
        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        OperationRegistry() = default;
        OperationRegistry( const OperationRegistry& ) = delete;
        OperationRegistry& operator=( const OperationRegistry& ) = delete;
    };
}