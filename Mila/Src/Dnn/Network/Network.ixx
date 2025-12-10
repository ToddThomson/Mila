/**
 * @file Network.ixx
 * @brief Lightweight composite network container.
 *
 * Provides a simple CompositeComponent-derived container that owns child components
 * and exposes a minimal introspection and device access API.
 */

module;
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <algorithm>
#include <vector>
#include <exception>
#include <format>

export module Dnn.Network;

import Dnn.CompositeComponent;
import Dnn.ComponentFactory;
import Dnn.TensorDataType;
import Compute.ExecutionContext;
import Compute.DeviceType;
import Compute.Device;
import Compute.DeviceId;
import Serialization.ModelArchive;
import Serialization.Mode;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Lightweight composite network container.
     *
     * Network is a specialized CompositeComponent that represents a complete neural
     * network model. It manages child components and provides high-level serialization
     * and model management operations.
     *
     * Construction patterns:
     * - **Standalone**: Public constructor accepting DeviceId and name creates owned ExecutionContext
     * - **Deserialization**: Protected constructor accepting IExecutionContext* for factory reconstruction
     *
     * The Network owns and manages the shared ExecutionContext that is propagated to all
     * child components for efficient resource sharing.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class Network : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using CompositeBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeBase::ComponentPtr;
        using ExecutionContextType = ExecutionContext<TDeviceType>;

        /**
         * @brief Construct network with owned ExecutionContext.
         *
         * Creates a new network with its own ExecutionContext that will be shared
         * with all child components.
         *
         * @param device_id DeviceId identifying the device for this network and its children.
         * @param name Network name for identification and serialization.
         *
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails.
         */
        explicit Network( DeviceId device_id, const std::string& name )
            : CompositeBase( device_id )
            , name_( name )
        {
            if (name_.empty())
            {
                throw std::invalid_argument( "Network: name cannot be empty" );
            }
        }

        ~Network() override = default;

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save network to archive
         *
         * Produces:
         *  - network/meta.json         : small metadata (name, format_version, export_time, mode, num_components)
         *  - network/architecture.json : manifest (array of component descriptors: name, path, index)
         *  - components/<name>/...     : each component's own save_() output (scoped)
         *
         * Behavior:
         *  - Writes components in a deterministic (sorted by name) order.
         *  - Calls each child's `save_()` with a scoped archive prefix so leaf
         *    implementations write relative paths only.
         */
        void save( ModelArchive& archive, SerializationMode mode ) const
        {
            const auto& named_map = this->getNamedComponents();
            std::vector<std::string> names;
            names.reserve( named_map.size() );

            for (const auto& p : named_map)
            {
                names.push_back( p.first );
            }

            std::sort( names.begin(), names.end() );

            json net_meta;
            net_meta["format_version"] = 1;
            net_meta["name"] = name_;
            net_meta["num_components"] = names.size();
            net_meta["mode"] = serializationModeToString( mode );

            auto now = std::chrono::system_clock::now();
            net_meta["export_time"] = static_cast<int64_t>(
                std::chrono::system_clock::to_time_t( now )
            );

            archive.writeJson( "network/meta.json", net_meta );

            json arch = json::array();

            for (size_t i = 0; i < names.size(); ++i)
            {
                const auto& nm = names[i];
                json entry = json::object();
                entry["name"] = nm;
                entry["path"] = "components/" + nm;
                entry["index"] = static_cast<int>( i );
                arch.push_back( entry );
            }

            archive.writeJson( "network/architecture.json", arch );

            for (const auto& nm : names)
            {
                auto it = named_map.find( nm );

                if (it == named_map.end())
                {
                    throw std::runtime_error(
                        "Network::save: inconsistent named components map for '" + nm + "'"
                    );
                }

                const auto& component = it->second;

                try
                {
                    ModelArchive::ScopedScope scope( archive, std::string( "components/" ) + nm );
                    component->save_( archive, mode );
                }
                catch (const std::exception& e)
                {
                    throw std::runtime_error(
                        std::format(
                            "Network::save: failed saving component '{}' into archive '{}': {}",
                            nm,
                            archive.getFilepath(),
                            e.what()
                        )
                    );
                }
            }
        }

        /**
         * @brief Load network from archive
         *
         * Reconstructs a Network using the provided execution context. The network
         * will share the provided context (typically from a parent composite or factory).
         *
         * @param archive Archive containing the serialized network
         * @param exec_context Execution context to share with the reconstructed network
         * @return Unique pointer to reconstructed Network
         *
         * @throws std::runtime_error if archive is malformed or component reconstruction fails
         *
         * @note ComponentFactory integration is TODO; this method validates meta/architecture schema.
         */
        static std::unique_ptr<Network> load(
            ModelArchive& archive,
            IExecutionContext* exec_context )
        {
            json net_meta = archive.readJson( "network/meta.json" );
            std::string name = net_meta.at( "name" );

            json arch = archive.readJson( "network/architecture.json" );

            if (net_meta.contains( "num_components" ))
            {
                size_t hint = net_meta.at( "num_components" ).get<size_t>();
                size_t actual = 0;

                if (arch.is_array())
                {
                    actual = arch.size();
                }

                if (hint != actual)
                {
                    throw std::runtime_error(
                        std::format(
                            "Network::load: metadata num_components ({}) does not match architecture.json size ({})",
                            hint,
                            actual
                        )
                    );
                }
            }

            auto network = std::unique_ptr<Network>(
                new Network( exec_context, name )
            );

            for (const auto& item : arch)
            {
                std::string component_name;

                if (item.is_object() && item.contains( "name" ))
                {
                    component_name = item.at( "name" ).get<std::string>();
                }
                else if (item.is_string())
                {
                    component_name = item.get<std::string>();
                }
                else
                {
                    throw std::runtime_error(
                        "Network::load: unexpected architecture.json entry format"
                    );
                }

                // FIXME: reconstruct child components using ComponentFactory when available.
                (void)component_name;
                //auto component = ComponentFactory::create( archive, component_name, exec_context );
                //network->addComponent( std::move( component ) );
            }

            return network;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        /**
         * @brief Get the network name.
         *
         * @return Network name used for identification and serialization
         */
        std::string getName() const override
        {
            return name_;
        }

        /**
         * @brief Generate a human-readable description.
         *
         * @return String representation showing network name and children
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Network: " << name_ << " " << CompositeBase::toString();

            return oss.str();
        }

    protected:

        /**
         * @brief Construct network as child component sharing parent's ExecutionContext.
         *
         * Used by factory or deserialization to create a network that shares
         * an existing execution context. Typically called via the static load() method.
         *
         * @param exec_context Non-owning pointer to shared execution context (must be non-null).
         * @param name Network name for identification.
         *
         * @throws std::invalid_argument if exec_context is null or name is empty.
         */
        explicit Network( IExecutionContext* exec_context, const std::string& name )
            : CompositeBase( exec_context )
            , name_( name )
        {
            if (name_.empty())
            {
                throw std::invalid_argument( "Network: name cannot be empty" );
            }
        }

    private:

        std::string name_;

        friend class ComponentFactory;
    };
}