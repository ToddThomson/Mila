// File: Mila/Src/Dnn/Compute/OperationBase.ixx
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
    export 
    template<typename T> 
    class OperationRegistry {
    public:

        using OperationCreator = std::function<std::shared_ptr<OperationBase<T>>()>;

        static OperationRegistry& instance() {
            static OperationRegistry registry;
			if ( !is_initialized_ ) {
				is_initialized_ = true;

			}

            return registry;
        }

        void registerOperation( const std::string& device_name, const std::string& operation_name, OperationCreator creator ) {
            registry_[ device_name ][ operation_name ] = std::move( creator );
        }
        
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
        std::unordered_map<std::string, std::unordered_map<std::string, OperationCreator>> registry_;
        static inline bool is_initialized_ = false;

        OperationRegistry() = default;
    };
}