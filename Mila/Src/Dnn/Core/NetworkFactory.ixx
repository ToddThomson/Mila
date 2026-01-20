module;
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <format>

export module Dnn.NetworkFactory;

import Dnn.Network;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Factory registry for Network deserialization.
     *
     * Provides type-safe network reconstruction from archives using registered
     * factory functions. Each concrete network type registers its own Load()
     * method to enable polymorphic deserialization.
     *
     * Design Pattern:
     * - Registration: Concrete networks register factory functions at startup
     * - Dispatch: Factory reads metadata to determine network type and precision
     * - Construction: Invokes appropriate registered factory function
     *
     * Usage:
     * @code
     * // Registration (typically in network implementation file)
     * NetworkFactory::registerNetwork<DeviceType::Cpu, TensorDataType::FP32>(
     *     "MnistClassifier",
     *     [](ModelArchive& archive, auto exec_ctx) {
     *         return MnistClassifier::Load(archive, exec_ctx->getDeviceId());
     *     });
     *
     * // Deserialization
     * auto network = NetworkFactory::create<DeviceType::Cpu, TensorDataType::FP32>(
     *     archive, exec_context);
     * @endcode
     */
    export class NetworkFactory
    {
    public:

        template<DeviceType TDeviceType, TensorDataType TPrecision>
        using NetworkFactoryFunc = std::function<std::unique_ptr<Network<TDeviceType, TPrecision>>(
            ModelArchive&,
            std::shared_ptr<ExecutionContext<TDeviceType>>
        )>;

        /**
         * @brief Register a network factory function for a given type.
         *
         * Associates a network type string with a factory function that can
         * reconstruct instances of that network from an archive.
         *
         * @tparam TDeviceType Device type for the network
         * @tparam TPrecision Precision type for the network
         * @param network_type String identifier for the network type (e.g., "MnistClassifier")
         * @param factory Factory function that reconstructs the network
         *
         * @note Registration is typically performed at static initialization time
         * @note Multiple registrations for the same type will overwrite previous entries
         *
         * @example
         * NetworkFactory::registerNetwork<DeviceType::Cpu, TensorDataType::FP32>(
         *     "MnistClassifier",
         *     [](ModelArchive& archive, auto ctx) {
         *         return MnistClassifier::Load(archive, ctx->getDeviceId());
         *     });
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        static void registerNetwork(
            const std::string& network_type,
            NetworkFactoryFunc<TDeviceType, TPrecision> factory )
        {
            getRegistry<TDeviceType, TPrecision>()[ network_type ] = std::move( factory );
        }

        /**
         * @brief Create a network instance from archive using registered factories.
         *
         * Reads network metadata from the archive to determine the network type
         * and dispatches to the appropriate registered factory function for
         * reconstruction.
         *
         * Metadata Requirements:
         * - network/meta.json must contain:
         *   - "network_type": String identifying the concrete network class
         *   - "template_precision": String identifying the precision type
         *
         * @tparam TDeviceType Device type for the network
         * @tparam TPrecision Precision type for the network
         * @param archive Archive containing serialized network
         * @param exec_context Execution context for the reconstructed network
         * @return Unique pointer to reconstructed network
         *
         * @throws std::runtime_error if network metadata is missing or malformed
         * @throws std::runtime_error if no factory is registered for the network type
         * @throws std::runtime_error if network reconstruction fails
         *
         * @example
         * auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>(...);
         * auto network = NetworkFactory::create<DeviceType::Cpu, TensorDataType::FP32>(
         *     archive, exec_ctx);
         */
        template<Compute::DeviceType TDeviceType, TensorDataType TPrecision>
        static std::unique_ptr<Network<TDeviceType, TPrecision>> create(
            ModelArchive& archive,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            SerializationMetadata net_meta = archive.readMetadata( "network/meta.json" );

            if ( !net_meta.has( "network_type" ) )
            {
                throw std::runtime_error(
                    "NetworkFactory::create: network metadata missing 'network_type' field" );
            }

            if ( !net_meta.has( "template_precision" ) )
            {
                throw std::runtime_error(
                    "NetworkFactory::create: network metadata missing 'template_precision' field" );
            }

            std::string network_type = net_meta.getString( "network_type" );
            std::string precision = net_meta.getString( "template_precision" );

            std::string key = network_type + ":" + precision;

            auto& registry = getRegistry<TDeviceType, TPrecision>();
            auto it = registry.find( key );

            if ( it == registry.end() )
            {
                throw std::runtime_error(
                    std::format( "NetworkFactory::create: No factory registered for '{}'", key ) );
            }

            try
            {
                return it->second( archive, exec_context );
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error(
                    std::format( "NetworkFactory::create: Failed to create network '{}': {}",
                        key, e.what() ) );
            }
        }

    private:
        /**
         * @brief Get the registry for a specific device type and precision.
         *
         * Returns a reference to the static registry map for the given
         * device and precision combination. Each instantiation has its
         * own independent registry.
         *
         * @tparam TDeviceType Device type
         * @tparam TPrecision Precision type
         * @return Reference to the static registry map
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        static std::unordered_map<std::string, NetworkFactoryFunc<TDeviceType, TPrecision>>&
            getRegistry()
        {
            static std::unordered_map<std::string, NetworkFactoryFunc<TDeviceType, TPrecision>> registry;
            return registry;
        }
    };
}
