module;
#include <string>
#include <memory>
#include <stdexcept>
#include <format>
#include <unordered_map>
#include <functional>

export module Dnn.ModuleFactory;

import Dnn.Module;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;
import nlohmann.json;

import Dnn.Core.ModuleRegistry;
import Dnn.ModuleRegistrar;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    export class ModuleFactory
    {
    private:
        // Type-erased factory function signature
        template<DeviceType TDeviceType>
        using FactoryFunc = std::function<std::unique_ptr<Module<TDeviceType>>(
            ModelArchive&,
            const std::string&,
            std::shared_ptr<ExecutionContext<TDeviceType>>
        )>;

        // Registry: maps (module_type, device, precision, ...) -> factory function
        template<DeviceType TDeviceType>
        struct Registry
        {
            std::unordered_map<std::string, FactoryFunc<TDeviceType>> factories;
        };

        template<DeviceType TDeviceType>
        static Registry<TDeviceType>& getRegistry()
        {
            static Registry<TDeviceType> registry;
            return registry;
        }

    public:
        /**
         * @brief Register a module type for deserialization
         */
        template<DeviceType TDeviceType, typename ModuleType>
        static void registerModule( const std::string& key )
        {
            auto& registry = getRegistry<TDeviceType>();

            registry.factories[key] = [](
                ModelArchive& archive,
                const std::string& module_name,
                std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
                {
                    return ModuleType::fromArchive_( archive, module_name, exec_context );
                };
        }

        /**
         * @brief Create module from archive
         */
        template<DeviceType TDeviceType>
        static std::unique_ptr<Module<TDeviceType>>
            create( ModelArchive& archive,
                const std::string& module_name,
                std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            const std::string prefix = "modules/" + module_name;
            json meta = archive.readJson( prefix + "/meta.json" );

            // Build lookup key from metadata
            std::string key = buildFactoryKey( meta );

            auto& registry = getRegistry<TDeviceType>();
            auto it = registry.factories.find( key );

            if (it == registry.factories.end())
            {
                throw std::runtime_error(
                    std::format( "No factory registered for key '{}' (module: '{}')",
                        key, module_name )
                );
            }

            return it->second( archive, module_name, exec_context );
        }

    private:
        static std::string buildFactoryKey( const json& meta )
        {
            // Build key like "Gelu:float32" or "Encoder:int32:float32"
            std::string type = meta.at( "type" ).get<std::string>();
            std::string precision = meta.at( "template_precision" ).get<std::string>();

            std::string key = type + ":" + precision;

            // Handle additional template parameters if present
            if (meta.contains( "template_index" ))
            {
                std::string index = meta.at( "template_index" ).get<std::string>();
                key = type + ":" + index + ":" + precision;
            }

            return key;
        }
    };
}