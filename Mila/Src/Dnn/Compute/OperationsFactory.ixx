module;
#include <memory>
#include <string>

export module Compute.OperationsFactory;

import Compute.OperationBase;
import Compute.OperationRegistry;

export namespace Mila::Dnn::Compute
{
    export template<typename T> 
    class OperationsFactory {
    public:
        static std::shared_ptr<OperationBase<T>> createOperation( const std::string& deviceType, const std::string& operationName ) {
            return OperationRegistry<T>::instance().createOperation( deviceType, operationName );
        }
    };
}