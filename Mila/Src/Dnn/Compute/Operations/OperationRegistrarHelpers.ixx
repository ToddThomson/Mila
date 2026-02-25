/**
 * @file OperationRegistrarHelpers.ixx
 * @brief Helpers to standardize registration of unary/binary ops.
 *
 * Small helpers that produce consistent factory lambdas for OperationRegistry.
 */

module;
#include <memory>
#include <string>
#include <type_traits>
#include <stdexcept>

export module Compute.OperationRegistrarHelpers;

import Dnn.ComponentConfig;
import Dnn.TensorDataType;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.BinaryOperation;

namespace Mila::Dnn::Compute
{
    // Helper used only for static_assert diagnostics.
    template<typename T>
    struct always_false : std::false_type {};

    /**
     * @brief Attempt to construct an OpType instance from a raw IExecutionContext*.
     *
     * Supports three constructor shapes (in priority order):
     *  - OpType(IExecutionContext*, const ConfigT&)
     *  - OpType(ExecutionContext<DT>*, const ConfigT&)
     *  - OpType(std::shared_ptr<ExecutionContext<DT>>, const ConfigT&)
     *
     * Throws std::invalid_argument when a required dynamic cast fails.
     */
    template<typename OpType, DeviceType DT, typename ConfigT>
    std::shared_ptr<OpType> makeOpInstance( IExecutionContext* ctx, const ConfigT& cfg )
    {
        if constexpr ( std::is_constructible_v<OpType, IExecutionContext*, const ConfigT&> )
        {
            return std::make_shared<OpType>( ctx, cfg );
        }
        else if constexpr ( std::is_constructible_v<OpType, ExecutionContext<DT>*, const ConfigT&> )
        {
            auto real = dynamic_cast<ExecutionContext<DT>*>( ctx );
            if ( !real )
            {
                throw std::invalid_argument( "makeOpInstance: failed to cast IExecutionContext* to ExecutionContext<DT>*" );
            }

            return std::make_shared<OpType>( real, cfg );
        }
        else if constexpr ( std::is_constructible_v<OpType, std::shared_ptr<ExecutionContext<DT>>, const ConfigT&> )
        {
            auto real = dynamic_cast<ExecutionContext<DT>*>( ctx );
            if ( !real )
            {
                throw std::invalid_argument( "makeOpInstance: failed to cast IExecutionContext* to ExecutionContext<DT>* for shared_ptr ctor" );
            }

            // Create a non-owning shared_ptr wrapper (deleter is no-op) so we don't change ownership semantics.
            std::shared_ptr<ExecutionContext<DT>> sp( real, [](ExecutionContext<DT>*){} );
            return std::make_shared<OpType>( sp, cfg );
        }
        else
        {
            static_assert( always_false<OpType>::value,
                "OpType must be constructible with one of: (IExecutionContext*, Config), (ExecutionContext<DT>*, Config) or (shared_ptr<ExecutionContext<DT>>, Config)" );
        }
    }

    /**
     * @brief Register a unary operation type with OperationRegistry using a common factory pattern.
     *
     * Template parameter ordering and names:
     *  - TDataType : DeviceType
     *  - OpType    : Concrete operation class (must define `using ConfigType = ...`)
     *  - TInput    : Abstract input tensor precision
     *  - TPrecision: Compute/output precision (defaults to TInput)
     *
     * The factory lambda casts ComponentConfig -> OpType::ConfigType and forwards the
     * IExecutionContext* + concrete config to the operation constructor via makeOpInstance.
     */
    export template<DeviceType TDataType, typename OpType, Dnn::TensorDataType TInput, Dnn::TensorDataType TPrecision = TInput>
    void registerUnaryOpType( std::string_view op_name )
    {
        using ConfigType = typename OpType::ConfigType;

        static_assert(
            std::is_class_v<ConfigType>,
            "OpType must expose a ConfigType alias"
        );

        constexpr bool constructible =
            std::is_constructible_v<OpType, IExecutionContext*, const ConfigType&> ||
            std::is_constructible_v<OpType, ExecutionContext<TDataType>*, const ConfigType&> ||
            std::is_constructible_v<OpType, std::shared_ptr<ExecutionContext<TDataType>>, const ConfigType&>;

        static_assert( constructible,
            "OpType must be constructible with (IExecutionContext*, ConfigType) or (ExecutionContext<DT>*, ConfigType) or (shared_ptr<ExecutionContext<DT>>, ConfigType)" );

        OperationRegistry::instance().registerUnaryOperation<TDataType, TInput, TPrecision>(
            op_name,
            [/*capture nothing*/]( IExecutionContext* ctx, const ComponentConfig& cfg )
                -> std::shared_ptr<UnaryOperation<TDataType, TInput, TPrecision>>
            {
                const auto& concreteCfg = static_cast<const ConfigType&>( cfg );
                return makeOpInstance<OpType, TDataType>( ctx, concreteCfg );
            }
        );
    }

    /**
     * @brief Register a binary operation type with OperationRegistry using a common factory pattern.
     *
     * Template parameter ordering:
     *  - TDataType : DeviceType
     *  - OpType    : Concrete operation class (must define `using ConfigType = ...`)
     *  - TA        : Input A precision
     *  - TB        : Input B precision
     *  - TP        : Compute/output precision (defaults to TA)
     */
    export template<DeviceType TDataType, typename OpType, Dnn::TensorDataType TA, Dnn::TensorDataType TB, Dnn::TensorDataType TP = TA>
    void registerBinaryOpType( const std::string& opName )
    {
        using ConfigType = typename OpType::ConfigType;

        static_assert(
            std::is_class_v<ConfigType>,
            "OpType must expose a ConfigType alias"
        );

        constexpr bool constructible =
            std::is_constructible_v<OpType, IExecutionContext*, const ConfigType&> ||
            std::is_constructible_v<OpType, ExecutionContext<TDataType>*, const ConfigType&> ||
            std::is_constructible_v<OpType, std::shared_ptr<ExecutionContext<TDataType>>, const ConfigType&>;

        static_assert( constructible,
            "OpType must be constructible with (IExecutionContext*, ConfigType) or (ExecutionContext<DT>*, ConfigType) or (shared_ptr<ExecutionContext<DT>>, ConfigType)" );

        OperationRegistry::instance().registerBinaryOperation<TDataType, TA, TB, TP>(
            opName,
            [/*capture nothing*/]( IExecutionContext* ctx, const ComponentConfig& cfg )
                -> std::shared_ptr<BinaryOperation<TDataType, TA, TB, TP>>
            {
                const auto& concreteCfg = static_cast<const ConfigType&>( cfg );
                return makeOpInstance<OpType, TDataType>( ctx, concreteCfg );
            }
        );
    }
}