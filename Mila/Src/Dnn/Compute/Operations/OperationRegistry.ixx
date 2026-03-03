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
import Compute.PairedOperation;
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
     * @brief Central registry for typed, device-aware compute operations.
     *
     * Maintains three independent stores keyed by (DeviceType, input types, compute precision)
     * and operation name — one per operation arity. Separation prevents cross-arity
     * static_pointer_cast UB when an op is retrieved under the wrong base type.
     */
    export class OperationRegistry
    {
    public:

        /**
         * @brief Composite key for registry lookup.
         */
        struct TypeID
        {
            DeviceType     device_type;       ///< Target device type for the operation
            TensorDataType data_type_a;       ///< Element type for input A
            TensorDataType data_type_b;       ///< Element type for input B (equals data_type_a for unary)
            TensorDataType compute_precision; ///< Internal computation/accumulation precision

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
                std::size_t h1 = std::hash<DeviceType>{}( id.device_type );
                std::size_t h2 = std::hash<TensorDataType>{}( id.data_type_a );
                std::size_t h3 = std::hash<TensorDataType>{}( id.data_type_b );
                std::size_t h4 = std::hash<TensorDataType>{}( id.compute_precision );

                std::size_t seed = h1;
                seed ^= (h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                seed ^= (h3 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                seed ^= (h4 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));

                return seed;
            }
        };

        static OperationRegistry& instance()
        {
            static OperationRegistry registry;
            return registry;
        }

        // ====================================================================
        // Registration
        // ====================================================================

        /**
         * @brief Register a unary operation factory.
         *
         * Creator must return a std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>.
         */
        template<DeviceType TDeviceType, TensorDataType TInputType, TensorDataType TComputePrecision = TInputType>
        void registerUnaryOperation(
            std::string_view operation_name,
            std::function<std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>(
                IExecutionContext*, const ComponentConfig& )> creator )
        {
            TypeID type_id{ TDeviceType, TInputType, TInputType, TComputePrecision };

            unary_registry_[type_id][std::string( operation_name )] =
                [creator]( IExecutionContext* ctx, const ComponentConfig& cfg ) -> std::shared_ptr<void>
                {
                    if ( !ctx )
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

                    return creator( ctx, cfg );
                };
        }

        /**
         * @brief Register a binary operation factory.
         *
         * Creator must return a std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        void registerBinaryOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>(
                IExecutionContext*, const ComponentConfig& )> creator )
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };

            binary_registry_[type_id][operation_name] =
                [creator]( IExecutionContext* ctx, const ComponentConfig& cfg ) -> std::shared_ptr<void>
                {
                    if ( !ctx )
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

                    return creator( ctx, cfg );
                };
        }

        /**
         * @brief Register a paired operation factory.
         *
         * Creator must return a std::shared_ptr<PairedOperation<TDeviceType, TPrecision, TInputA, TInputB>>.
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision,
            TensorDataType TInputA = TPrecision, TensorDataType TInputB = TInputA>
        void registerPairedOperation(
            const std::string& operation_name,
            std::function<std::shared_ptr<PairedOperation<TDeviceType, TPrecision, TInputA, TInputB>>(
                IExecutionContext*, const ComponentConfig& )> creator )
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TPrecision };

            paired_registry_[type_id][operation_name] =
                [creator]( IExecutionContext* ctx, const ComponentConfig& cfg ) -> std::shared_ptr<void>
                {
                    if ( !ctx )
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

                    return creator( ctx, cfg );
                };
        }

        // ====================================================================
        // Creation
        // ====================================================================

        /**
         * @brief Create a unary operation instance.
         *
         * @throws std::runtime_error if no matching registration exists.
         * @throws std::invalid_argument if context is null.
         */
        template<DeviceType TDeviceType, TensorDataType TInputType, TensorDataType TComputePrecision = TInputType>
        std::shared_ptr<UnaryOperation<TDeviceType, TInputType, TComputePrecision>> createUnaryOperation(
            const std::string& operation_name,
            IExecutionContext* context,
            const ComponentConfig& config ) const
        {
            if ( !context )
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

            TypeID type_id{ TDeviceType, TInputType, TInputType, TComputePrecision };

            auto type_it = unary_registry_.find( type_id );

            if ( type_it == unary_registry_.end() )
            {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: No operations registered for Device: {}, InputType: {}, ComputePrecision: {}",
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputType>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name ) );
            }

            auto op_it = type_it->second.find( operation_name );

            if ( op_it == type_it->second.end() )
            {
                throw std::runtime_error( std::format(
                    "createUnaryOperation: Operation '{}' not found for Device: {}, InputType: {}, ComputePrecision: {}",
                    operation_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputType>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name ) );
            }

            return std::static_pointer_cast<UnaryOperation<TDeviceType, TInputType, TComputePrecision>>(
                op_it->second( context, config ) );
        }

        /**
         * @brief Create a binary operation instance.
         *
         * @throws std::runtime_error if no matching registration exists.
         * @throws std::invalid_argument if context is null.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        std::shared_ptr<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>> createBinaryOperation(
            const std::string& operation_name,
            IExecutionContext* context,
            const ComponentConfig& config ) const
        {
            if ( !context )
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };

            auto type_it = binary_registry_.find( type_id );

            if ( type_it == binary_registry_.end() )
            {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: No operations registered for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name ) );
            }

            auto op_it = type_it->second.find( operation_name );

            if ( op_it == type_it->second.end() )
            {
                throw std::runtime_error( std::format(
                    "createBinaryOperation: Operation '{}' not found for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    operation_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TComputePrecision>::type_name ) );
            }

            return std::static_pointer_cast<BinaryOperation<TDeviceType, TInputA, TInputB, TComputePrecision>>(
                op_it->second( context, config ) );
        }

        /**
         * @brief Create a paired operation instance.
         *
         * @throws std::runtime_error if no matching registration exists.
         * @throws std::invalid_argument if context is null.
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision,
            TensorDataType TInputA = TPrecision, TensorDataType TInputB = TInputA>
        std::shared_ptr<PairedOperation<TDeviceType, TPrecision, TInputA, TInputB>> createPairedOperation(
            const std::string& operation_name,
            IExecutionContext* context,
            const ComponentConfig& config ) const
        {
            if ( !context )
                throw std::invalid_argument( "ExecutionContext cannot be null when creating an operation" );

            TypeID type_id{ TDeviceType, TInputA, TInputB, TPrecision };

            auto type_it = paired_registry_.find( type_id );

            if ( type_it == paired_registry_.end() )
            {
                throw std::runtime_error( std::format(
                    "createPairedOperation: No operations registered for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TPrecision>::type_name ) );
            }

            auto op_it = type_it->second.find( operation_name );

            if ( op_it == type_it->second.end() )
            {
                throw std::runtime_error( std::format(
                    "createPairedOperation: Operation '{}' not found for Device: {}, InputTypes: ({}, {}), ComputePrecision: {}",
                    operation_name,
                    TDeviceType == DeviceType::Cpu ? "CPU" : "CUDA",
                    TensorDataTypeTraits<TInputA>::type_name,
                    TensorDataTypeTraits<TInputB>::type_name,
                    TensorDataTypeTraits<TPrecision>::type_name ) );
            }

            return std::static_pointer_cast<PairedOperation<TDeviceType, TPrecision, TInputA, TInputB>>(
                op_it->second( context, config ) );
        }

        // ====================================================================
        // Discovery
        // ====================================================================

        /**
         * @brief Return all registered operation names across all arities for a given type configuration.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        std::vector<std::string> getRegisteredOperations() const
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };
            std::vector<std::string> operations;

            for ( const auto* store : { &unary_registry_, &binary_registry_, &paired_registry_ } )
            {
                auto type_it = store->find( type_id );

                if ( type_it != store->end() )
                {
                    for ( const auto& [name, _] : type_it->second )
                        operations.push_back( name );
                }
            }

            return operations;
        }

        /**
         * @brief Return true if an operation name is registered under any arity for a given type configuration.
         */
        template<DeviceType TDeviceType, TensorDataType TInputA, TensorDataType TInputB = TInputA,
            TensorDataType TComputePrecision = TInputA>
        bool isOperationRegistered( const std::string& operation_name ) const
        {
            TypeID type_id{ TDeviceType, TInputA, TInputB, TComputePrecision };

            for ( const auto* store : { &unary_registry_, &binary_registry_, &paired_registry_ } )
            {
                auto type_it = store->find( type_id );

                if ( type_it != store->end() &&
                    type_it->second.find( operation_name ) != type_it->second.end() )
                {
                    return true;
                }
            }

            return false;
        }

    private:

        using GenericCreator = std::function<std::shared_ptr<void>(
            IExecutionContext*, const ComponentConfig& )>;

        using RegistryMap = std::unordered_map<TypeID,
            std::unordered_map<std::string, GenericCreator>,
            TypeIDHash>;

        RegistryMap unary_registry_;
        RegistryMap binary_registry_;
        RegistryMap paired_registry_;

        OperationRegistry() = default;
        OperationRegistry( const OperationRegistry& ) = delete;
        OperationRegistry& operator=( const OperationRegistry& ) = delete;
    };
}