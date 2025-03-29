module;
#include <string>
#include <format>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <typeindex>

export module Compute.OperationRegistry;

import Dnn.TensorTraits;

import Compute.OperationBase;
import Compute.DeviceType;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

export namespace Mila::Dnn::Compute
{
    /**
     * @brief A registry for operations that can be created based on device, operation names, and type information.
     */
    export class OperationRegistry {
    public:
        /**
         * @brief Type ID structure to uniquely identify operations based on input type, output type, and device type.
         */
        struct TypeID {
            std::type_index input_type;
            std::type_index output_type;
            DeviceType device_type;

            bool operator==( const TypeID& other ) const {
                return input_type == other.input_type &&
                    output_type == other.output_type &&
                    device_type == other.device_type;
            }
        };

        /**
         * @brief Hash function for TypeID to use in unordered_map.
         */
        struct TypeIDHash {
            std::size_t operator()( const TypeID& id ) const {
                std::size_t h1 = std::hash<std::type_index>{}(id.input_type);
                std::size_t h2 = std::hash<std::type_index>{}(id.output_type);
                std::size_t h3 = std::hash<DeviceType>{}(id.device_type);
                return h1 ^ (h2 << 1) ^ (h3 << 2);
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
         * @brief Register an operation creator for a specific type combination and operation name.
         *
         * @tparam TInput The input tensor element type.
         * @tparam TPrecision The output tensor element type.
         * @tparam TDeviceType The device type.
         * @param operation_name The name of the operation.
         * @param creator The function that creates the operation.
         */
        template<typename TInput, typename TPrecision, DeviceType TDeviceType>
            requires ValidTensorTypes<TInput, TPrecision>
        void registerOperation( const std::string& operation_name,
            std::function<std::shared_ptr<OperationBase<TInput, TPrecision, TDeviceType>>()> creator ) {

            TypeID id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TPrecision) ),
                TDeviceType
            };

            registry_[ id ][ operation_name ] =
                [creator = std::move( creator )]() -> std::shared_ptr<void> {
                   return creator();
                };
        }

        /**
         * @brief Create an operation based on the type information and operation name.
         *
         * @tparam TInput The input tensor element type.
         * @tparam TPrecision The output tensor element type.
         * @tparam TDeviceType The device type.
         * @param operation_name The name of the operation.
         * @return std::unique_ptr<OperationBase<TInput, TPrecision, TDeviceType>> The created operation.
         * @throws std::runtime_error If the type combination or operation name is invalid.
         */
        template<typename TInput, typename TPrecision, DeviceType TDeviceType>
            requires ValidTensorTypes<TInput, TPrecision>
        std::shared_ptr<OperationBase<TInput, TPrecision, TDeviceType>> createOperation( const std::string& operation_name ) const {
            TypeID id{
                std::type_index( typeid(TInput) ),
                std::type_index( typeid(TPrecision) ),
                TDeviceType
            };

            auto typeIt = registry_.find( id );
            if ( typeIt == registry_.end() ) {
                throw std::runtime_error( std::format(
                    "createOperation: No operations registered for types Input:{}, Output:{}, Device:{}",
                    typeid(TInput).name(), typeid(TPrecision).name(), deviceToString( TDeviceType )
                ) );
            }

            auto opIt = typeIt->second.find( operation_name );
            if ( opIt == typeIt->second.end() ) {
                throw std::runtime_error( std::format( "createOperation: Operation not found: {}", operation_name ) );
            }

            return std::static_pointer_cast<OperationBase<TInput, TPrecision, TDeviceType>>(opIt->second());
        }

    private:
        using GenericCreator = std::function<std::shared_ptr<void>()>;
        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;

        OperationRegistry() = default; ///< Default constructor.
    };
}