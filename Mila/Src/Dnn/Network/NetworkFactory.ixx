module;
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <format>


export module Dnn.NetworkFactory;

import Dnn.Network;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;
import nlohmann.json;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;
    using nlohmann::json;

    export class NetworkFactory
    {
    public:

        template<DeviceType TDeviceType>
        using NetworkFactoryFunc = std::function<std::unique_ptr<Network<TDeviceType>>(
            ModelArchive&,
            std::shared_ptr<ExecutionContext<TDeviceType>>
        )>;

        template<DeviceType TDeviceType>
        static void registerNetwork(
            const std::string& network_type,
            NetworkFactoryFunc<TDeviceType> factory )
        {
            getRegistry<TDeviceType>()[network_type] = std::move( factory );
        }

        template<Compute::DeviceType TDeviceType>
        static std::unique_ptr<Network<TDeviceType>> create(
            ModelArchive& archive,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            json net_meta = archive.readJson( "network/meta.json" );

            std::string network_type = net_meta.at( "network_type" ).get<std::string>();
            std::string precision = net_meta.at( "template_precision" ).get<std::string>();

            std::string key = network_type + ":" + precision;

            auto& registry = getRegistry<TDeviceType>();
            auto it = registry.find( key );

            if (it == registry.end())
            {
                throw std::runtime_error(
                    std::format( "No network factory registered for '{}'", key ) );
            }

            return it->second( archive, exec_context );
        }

    private:
        template<DeviceType TDeviceType>
        static std::unordered_map<std::string, NetworkFactoryFunc<TDeviceType>>&
            getRegistry()
        {
            static std::unordered_map<std::string, NetworkFactoryFunc<TDeviceType>> registry;
            return registry;
        }
    };
}
