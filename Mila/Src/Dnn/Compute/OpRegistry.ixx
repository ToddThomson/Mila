module;
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

export module Compute.OpRegistry;

export class OpRegistry {
public:
    using FactoryFunction = void* (*)();

    static OpRegistry& instance() {
        static OpRegistry inst;
        return inst;
    }

    void registerClass( const std::string& name, FactoryFunction factory ) {
        registry_[ name ] = factory;
    }

    template<typename T>
    std::unique_ptr<T> create( const std::string& name ) {
        auto it = registry_.find( name );
        if ( it != registry_.end() ) {
            void* rawPtr = it->second();
            return std::unique_ptr<T>( static_cast<T*>(rawPtr) );
        }
        return nullptr;
    }

private:
    std::unordered_map<std::string, FactoryFunction> registry_;
    OpRegistry() = default;
};

