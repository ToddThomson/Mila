/**
 * @file OperationRegistry.ixx
 * @brief Central registry for creating and discovering compute operations.
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
import Dnn.ComponentConfig;
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
     * @brief A registry for operations using abstract tensor data types.
     *
     * Stores factory functions for concrete, typed operations. Factories are
     * registered per (DeviceType, input types, compute precision) and operation name.
     */
    export class OperationRegistry
    {
    public:

        /**
         * @brief Composite key for registry lookup.
         *
         * For unary operations the second data_type equals the input type.
         */
        struct TypeID
        {
            DeviceType device_type;                 ///< Target device type for the operation
            TensorDataType data_type_a;             ///< Abstract tensor data type for input A
            TensorDataType data_type_b;             ///< Abstract tensor data type for input B (or same as A for unary)
            TensorDataType compute_precision;       ///< Internal computation/accumulation precision

            bool operator==( const TypeID& other ) const
            {
                return device_type == other.device_type &&
                    data_type_a == other.data_type_a &&
                    data_type_b == other.data_type_b &&
                    compute_precision == other.compute_precision;
            }
        };

        struct TypeIDHash
        {
            std::size_t operator()( const TypeID& id ) const
            {
                std::size_t h1 = std::hash<DeviceType>{}(id.device_type);
                std::size_t h2 = std::hash<TensorDataType>{}(id.data_type_a);
                std::size_t h3 = std::hash<TensorDataType>{}(id.data_type_b);
                std::size_t h4 = std::hash<TensorDataType>{}(id.compute_precision);

                // Combine hashes (simple, deterministic mix)
                std::size_t seed = h1;
                seed ^= (h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                seed ^= (h3 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                seed ^= (h4 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                return seed;
            }
        };

        /**
         * @brief Get the singleton instance of the OperationRegistry.
         */
        static OperationRegistry& instance()
        {
            static OperationRegistry registry;
            return registry;
        }

        /**
         * @brief Register a unary operation creator for a specific device type, input type, and compute precision.
         *
         * Creator must accept a std::shared_ptr<ExecutionContext<TDeviceType>> and ModuleConfig
         * and return a std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>.
         */
        template<DeviceType TDeviceType, TensorDataType TInputType, TensorDataType TComputePrecision = TInputType>
        void registerUnaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>(
                std::shared_ptr<ExecutionContext<TDeviceType>>,
                const ComponentConfig& )> creator )
        {

            TypeID type_id{ TDeviceType, TInputType, TInputType, TComputePrecision };

            auto genericCreator = [creator](
                std::shared_ptr<IExecutionContext> ictx,
                const ComponentConfig& config ) -> std::shared_ptr<void> {

                    if (!ictx)
                    {
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
                    }

                    auto ctx = std::static_pointer_cast<ExecutionContext<TDeviceType>>(ictx);
                    if (!ctx)
                    {
                        throw std::runtime_error( "ExecutionContext cast failed in registry creator." );
                    }

                    return creator( ctx, config );
                };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Register a binary operation creator for a specific device type, input types, and compute precision.
         *
         * Creator must accept a std::shared_ptr<ExecutionContext<TDeviceType>> and ModuleConfig
         * and return a std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        void registerBinaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>(
                std::shared_ptr<ExecutionContext<TDeviceType>>,
                const ComponentConfig& )> creator )
        {

            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };

            auto genericCreator = [creator](
                std::shared_ptr<IExecutionContext> ictx,
                const ComponentConfig& config ) -> std::shared_ptr<void> {

                    if (!ictx)
                    {
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
                    }

                    auto ctx = std::static_pointer_cast<ExecutionContext<TDeviceType>>(ictx);
                    if (!ctx)
                    {
                        throw std::runtime_error( "ExecutionContext cast failed in registry creator." );
                    }

                    return creator( ctx, config );
                };

            registry_[type_id][operation_name] = std::move( genericCreator );
        }

        /**
         * @brief Create a unary operation instance (typed).
         */
        template<DeviceType TDeviceType, TensorDataType TInputType, TensorDataType TComputePrecision = TInputType>
        std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>> createUnaryOperation(
            const std::string& operation_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> context,
            const ComponentConfig& config ) const
        {
            TypeID type_id{ TDeviceType, TInputType, TInputType, TComputePrecision };

            auto type_it = registry_.find( type_id );

            if (type_it == registry_.end())
            {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: No operations registered for Device: {}, InputType: {}, ComputePrecision: {}",
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputType>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name
                ) );
            }

            auto op_it = type_it->second.find( operation_name );

            if (op_it == type_it->second.end())
            {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: Operation '{}' not found for Device: {}, InputType: {}, ComputePrecision: {}",
                    operation_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputType>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name
                ) );
            }

            if (!context)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
            }

            auto op = std::static_pointer_cast<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>(
                op_it->second( std::static_pointer_cast<IExecutionContext>(context), config ));

            return op;
        }

        /**
         * @brief Create a binary operation instance (typed).
         *
         * Template parameters allow specifying distinct data types for the two inputs and compute precision.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>> createBinaryOperation(
            const std::string& operation_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> context,
            const ComponentConfig& config ) const
        {

            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };

            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end())
            {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: No operations registered for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name
                ) );
            }

            auto op_it = type_it->second.find( operation_name );
            if (op_it == type_it->second.end())
            {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: Operation '{}' not found for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    operation_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name
                ) );
            }

            if (!context)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );
            }

            auto op = std::static_pointer_cast<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>(
                op_it->second( std::static_pointer_cast<IExecutionContext>(context), config ));

            return op;
        }

        /**
         * @brief Get list of registered operation names for a given type configuration.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        std::vector<std::string> getRegisteredOperations() const
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };
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
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        bool isOperationRegistered( const std::string& operation_name ) const
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };
            auto type_it = registry_.find( type_id );
            if (type_it == registry_.end()) return false;
            return type_it->second.find( operation_name ) != type_it->second.end();
        }

    private:
        using GenericCreator = std::function<std::shared_ptr<void>(
            std::shared_ptr<IExecutionContext>,
            const ComponentConfig& )>;

        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        OperationRegistry() = default;
        OperationRegistry( const OperationRegistry& ) = delete;
        OperationRegistry& operator=( const OperationRegistry& ) = delete;
    };
}