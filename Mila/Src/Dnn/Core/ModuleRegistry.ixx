/**
 * @file ModuleRegistry.ixx
 * @brief Registry for module creators used by ModuleFactory.
 *
 * Centralized, type-erased registry that maps (DeviceType, Precision, module-type)
 * -> creator. Creators are registered per concrete template instantiation and
 * invoked via typed helpers.
 */

module;
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <format>

export module Dnn.ModuleRegistry;

import Dnn.Module;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.ExecutionContext;
//import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export class ModuleRegistry
    {
    public:
        static ModuleRegistry& instance()
        {
            static ModuleRegistry registry;
            return registry;
        }

        ModuleRegistry( const ModuleRegistry& ) = delete;
        ModuleRegistry& operator=( const ModuleRegistry& ) = delete;

        /**
         * @brief Register a module creator for a specific DeviceType + Precision.
         *
         * Creator must construct and return a std::shared_ptr<Module<TDeviceType>>
         * (typically a concrete Module<TDeviceType, TPrecision> instance upcasted to the base).
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        void registerModuleCreator(
            const std::string& module_type,
            std::function<std::shared_ptr<Module<TDeviceType>>(
                ModelArchive&,
                const std::string&,
                std::shared_ptr<ExecutionContext<TDeviceType>> )> creator )
        {
            TypeID key{ TDeviceType, TPrecision };

            GenericCreator generic = [creator]( std::shared_ptr<void> ctx_void,
                ModelArchive& archive,
                const std::string& module_name ) -> std::shared_ptr<void>
                {
                    if (!ctx_void)
                    {
                        throw std::invalid_argument( "ExecutionContext cannot be null when creating module" );
                    }

                    auto ctx = std::static_pointer_cast<ExecutionContext<TDeviceType>>(ctx_void);
                    if (!ctx)
                    {
                        throw std::runtime_error( "ExecutionContext cast failed in module creator." );
                    }

                    auto typed = creator( archive, module_name, ctx );
                    return std::static_pointer_cast<void>(typed);
                };

            std::scoped_lock lock( mutex_ );
            registry_[key][module_type] = std::move( generic );
        }

        /**
         * @brief Unregister a module creator for the given device/precision/type.
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        bool unregisterModuleCreator( const std::string& module_type )
        {
            TypeID key{ TDeviceType, TPrecision };

            std::scoped_lock lock( mutex_ );
            auto it = registry_.find( key );
            if (it == registry_.end()) return false;

            auto& inner = it->second;
            auto jt = inner.find( module_type );
            if (jt == inner.end()) return false;

            inner.erase( jt );

            if (inner.empty())
            {
                registry_.erase( it );
            }

            return true;
        }

        /**
         * @brief Create a typed module instance using a registered creator.
         *
         * Caller provides an ExecutionContext<TDeviceType>; the registry invokes
         * the stored generic creator with a type-erased shared_ptr (std::shared_ptr<void>)
         * which is cast back to ExecutionContext<TDeviceType> inside the creator wrapper.
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        std::shared_ptr<Module<TDeviceType>> createModule(
            const std::string& module_type,
            ModelArchive& archive,
            const std::string& module_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            TypeID key{ TDeviceType, TPrecision };

            GenericCreator creator;

            {
                std::scoped_lock lock( mutex_ );
                auto it = registry_.find( key );
                if (it == registry_.end())
                {
                    throw std::runtime_error(
                        std::format( "ModuleRegistry: no creators registered for Device={}, Precision={} (module '{}')",
                            deviceTypeToString( TDeviceType ), precisionToString( TPrecision ), module_name )
                    );
                }

                auto jt = it->second.find( module_type );
                if (jt == it->second.end())
                {
                    throw std::runtime_error(
                        std::format( "ModuleRegistry: no creator for type '{}' registered for Device={}, Precision={} (module '{}')",
                            module_type, deviceTypeToString( TDeviceType ), precisionToString( TPrecision ), module_name )
                    );
                }

                creator = jt->second;
            }

            try
            {
                // Pass execution context as a type-erased shared_ptr<void>.
                auto result = creator( std::static_pointer_cast<void>(exec_context), archive, module_name );

                if (!result)
                {
                    throw std::runtime_error(
                        std::format( "ModuleRegistry: creator returned null for '{}' (module '{}')", module_type, module_name )
                    );
                }

                auto typed = std::static_pointer_cast<Module<TDeviceType>>(result);

                return typed;
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(
                    std::format( "ModuleRegistry: creator for '{}' failed for module '{}': {}", module_type, module_name, e.what() )
                );
            }
        }

    private:
        struct TypeID
        {
            DeviceType device_type;
            TensorDataType precision;

            bool operator==( const TypeID& o ) const
            {
                return device_type == o.device_type && precision == o.precision;
            }
        };

        struct TypeIDHash
        {
            std::size_t operator()( const TypeID& id ) const noexcept
            {
                std::size_t h1 = std::hash<DeviceType>{}(id.device_type);
                std::size_t h2 = std::hash<TensorDataType>{}(id.precision);

                std::size_t seed = h1;
                seed ^= (h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                return seed;
            }
        };

        using GenericCreator = std::function<std::shared_ptr<void>( std::shared_ptr<void>, ModelArchive&, const std::string& )>;

        std::unordered_map<TypeID, std::unordered_map<std::string, GenericCreator>, TypeIDHash> registry_;
        std::mutex mutex_;

        ModuleRegistry() = default;
    };

    inline std::string precisionToString( TensorDataType p )
    {
        return std::to_string( static_cast<int>(p) );
    }
}