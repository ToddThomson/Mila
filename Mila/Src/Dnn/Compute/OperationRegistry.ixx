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
            return registry;
        }

        void registerOperation( const std::string& deviceType, const std::string& operationName, OperationCreator creator ) {
            registry_[ deviceType ][ operationName ] = std::move( creator );
        }
        
        std::shared_ptr<OperationBase<T>> createOperation( const std::string& deviceType, const std::string& operationName ) const {
			auto deviceIt = registry_.find( deviceType );
			if ( deviceIt == registry_.end() ) {
				throw std::runtime_error( "createOperation: Invalid device type." );
			}
			auto opIt = deviceIt->second.find( operationName );
			if ( opIt == deviceIt->second.end() ) {
				throw std::runtime_error( "Invalid operation name." );
			}
			return opIt->second();
        }

    private:
        std::unordered_map<std::string, std::unordered_map<std::string, OperationCreator>> registry_;

        OperationRegistry() = default;
    };
}