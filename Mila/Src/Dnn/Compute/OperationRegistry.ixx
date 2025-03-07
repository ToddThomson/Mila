module;
#include <string>
#include <memory>
#include <unordered_map>
#include <format>
#include <functional>
#include <stdexcept>
#include <type_traits>

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
     * @brief A registry for operations that can be created based on device and operation names.
     * 
     * @tparam T The type of the operation.
     */
    export 
    template<typename TInput, typename TOutput, typename TDevice>
        requires ValidTensorTypes<TInput, TOutput> && ( std::is_same_v<TDevice, CpuDevice> || std::is_same_v<TDevice, CudaDevice> )
    class OperationRegistry {
    public:

        using OperationCreator = std::function<std::unique_ptr<OperationBase<TInput, TOutput, TDevice>>()>; ///< Type alias for the operation creator function.

        /**
         * @brief Get the singleton instance of the OperationRegistry.
         * 
         * @return OperationRegistry& The singleton instance.
         */
        static OperationRegistry& instance() {
            static OperationRegistry registry;
            if ( !is_initialized_ ) {
                is_initialized_ = true;
            }

            return registry;
        }

        /**
         * @brief Register an operation creator for a specific device and operation name.
         * 
         * @param device_name The name of the device.
         * @param operation_name The name of the operation.
         * @param creator The function that creates the operation.
         */
        void registerOperation( const DeviceType& device_type, const std::string& operation_name, OperationCreator creator ) {
            registry_[ device_type ][ operation_name ] = std::move( creator );
        }
        
        /**
         * @brief Create an operation based on the device and operation names.
         * 
         * @param device_name The name of the device.
         * @param operation_name The name of the operation.
         * @return std::shared_ptr<OperationBase<T>> The created operation.
         * @throws std::runtime_error If the device or operation name is invalid.
         */
        std::shared_ptr<OperationBase<TInput, TOutput, TDevice>> createOperation( const DeviceType& device_type, const std::string& operation_name ) const {
            auto deviceIt = registry_.find( device_type );
            if ( deviceIt == registry_.end() ) {
                throw std::runtime_error( std::format( "createOperation: No operations registered for device type: {}", deviceToString( device_type ) ));
            }
            auto opIt = deviceIt->second.find( operation_name );
            if ( opIt == deviceIt->second.end() ) {
                throw std::runtime_error( std::format( "createOperation: Operation not found: {}", operation_name ) );
            }
            return opIt->second();
        }

    private:
        std::unordered_map<DeviceType, std::unordered_map<std::string, OperationCreator>> registry_; ///< The registry of operation creators.
        static inline bool is_initialized_ = false; ///< Flag to check if the registry is initialized.

        OperationRegistry() = default; ///< Default constructor.
    };
}