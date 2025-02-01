module;
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>

export module Compute.OperationRegistry;

import Compute.OperationBase;

export namespace Mila::Dnn::Compute
{
    /**
     * @brief A registry for operations that can be created based on device and operation names.
     * 
     * @tparam T The type of the operation.
     */
    export 
    template<typename T> 
    class OperationRegistry {
    public:

        using OperationCreator = std::function<std::shared_ptr<OperationBase<T>>()>; ///< Type alias for the operation creator function.

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
        void registerOperation( const std::string& device_name, const std::string& operation_name, OperationCreator creator ) {
            registry_[ device_name ][ operation_name ] = std::move( creator );
        }
        
        /**
         * @brief Create an operation based on the device and operation names.
         * 
         * @param device_name The name of the device.
         * @param operation_name The name of the operation.
         * @return std::shared_ptr<OperationBase<T>> The created operation.
         * @throws std::runtime_error If the device or operation name is invalid.
         */
        std::shared_ptr<OperationBase<T>> createOperation( const std::string& device_name, const std::string& operation_name ) const {
            auto deviceIt = registry_.find( device_name );
            if ( deviceIt == registry_.end() ) {
                throw std::runtime_error( "createOperation: Invalid device name." );
            }
            auto opIt = deviceIt->second.find( operation_name );
            if ( opIt == deviceIt->second.end() ) {
                throw std::runtime_error( "createOperation: Invalid operation name." );
            }
            return opIt->second();
        }

    private:
        std::unordered_map<std::string, std::unordered_map<std::string, OperationCreator>> registry_; ///< The registry of operation creators.
        static inline bool is_initialized_ = false; ///< Flag to check if the registry is initialized.

        OperationRegistry() = default; ///< Default constructor.
    };
}