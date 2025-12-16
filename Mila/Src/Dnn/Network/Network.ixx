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
import Compute.ExecutionContextFactory;
import Compute.IExecutionContext;
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
     * network model and serves as the top-level entry point for users. It manages
     * child components and provides high-level serialization and model management
     * operations.
     *
     * Ownership Model:
     * - **Network owns ExecutionContext** when constructed via DeviceId (primary use case)
     * - All child components share the Network's ExecutionContext (efficient resource pooling)
     * - Network can also share a parent's context when used as a nested component (rare)
     *
     * Construction Patterns:
     * - **Primary**: Public constructor accepting DeviceId creates and owns ExecutionContext
     *   - User-facing API: `Network(Device::Cpu(), "my_model")`
     *   - Network manages context lifetime, children borrow it
     * - **Secondary**: Public constructor accepting IExecutionContext* for deserialization/nesting
     *   - Used by factory reconstruction and nested network scenarios
     *   - Network shares parent's context, does not own it
     *
     * Child Component Construction:
     * - Use fluent `addComponent<ComponentType>(name, config_args...)` to build network hierarchy
     * - Factory method ensures all children share the Network's ExecutionContext
     * - Children are constructed with IExecutionContext* (single constructor pattern)
     *
     * Design Rationale:
     * - Network is special because it's the ownership boundary for ExecutionContext
     * - Users shouldn't need to manage ExecutionContext manually (ergonomics)
     * - All other components use single IExecutionContext* constructor (simplicity)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        class Network : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using CompositeBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeBase::ComponentPtr;

        /**
         * @brief Construct network with owned ExecutionContext (primary constructor).
         *
         * This is the main user-facing constructor. Creates and owns an ExecutionContext
         * for the specified device, which will be shared with all child components added
         * via the factory method pattern.
         *
         * @param device_id DeviceId identifying the device for this network and its children.
         * @param name Network name for identification and serialization.
         *
         * @throws std::invalid_argument if name is empty.
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails.
         */
        explicit Network( const std::string& name, DeviceId device_id )
            : CompositeBase( name )
        {
            // Create and own an execution context for the requested device
            createOwnedContext( device_id );

            // Propagate owned context to composite base and therefore to any children added
            CompositeBase::setExecutionContext( owned_context_.get() );
        }

        /**
         * @brief Construct network that shares an existing execution context.
         *
         * Secondary constructor used for deserialization and nested networks.
         * The provided context is non-owning; the network will share it with children.
         *
         * @param exec_context Non-owning execution context to share with the network.
         * @param name Network name for identification and serialization.
         *
         * @throws std::invalid_argument if name is empty or exec_context is null
         * @throws std::invalid_argument if exec_context device type does not match TDeviceType
         */
        explicit Network( const std::string& name, IExecutionContext* exec_context )
            : CompositeBase( name )
        {
            if ( exec_context == nullptr )
            {
                throw std::invalid_argument( "Network: exec_context cannot be null" );
            }

            if ( exec_context->getDeviceId().type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format(
                        "Network: constructor device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( exec_context->getDeviceId().type )
                    )
                );
            }

            // share the provided execution context with composite base and children
            CompositeBase::setExecutionContext( exec_context );
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

            for ( const auto& p : named_map )
            {
                names.push_back( p.first );
            }

            std::sort( names.begin(), names.end() );

            json net_meta;
            net_meta[ "format_version" ] = 1;
            net_meta[ "name" ] = this->getName();
            net_meta[ "num_components" ] = names.size();
            net_meta[ "mode" ] = serializationModeToString( mode );

            auto now = std::chrono::system_clock::now();
            net_meta[ "export_time" ] = static_cast<int64_t>(
                std::chrono::system_clock::to_time_t( now )
            );

            archive.writeJson( "network/meta.json", net_meta );

            json arch = json::array();

            for ( size_t i = 0; i < names.size(); ++i )
            {
                const auto& nm = names[ i ];
                json entry = json::object();
                entry[ "name" ] = nm;
                entry[ "path" ] = "components/" + nm;
                entry[ "index" ] = static_cast<int>( i );
                arch.push_back( entry );
            }

            archive.writeJson( "network/architecture.json", arch );

            for ( const auto& nm : names )
            {
                auto it = named_map.find( nm );

                if ( it == named_map.end() )
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
                catch ( const std::exception& e )
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
         * Child components are reconstructed via ComponentFactory and added using the
         * factory method to ensure ExecutionContext sharing.
         *
         * @param archive Archive containing the serialized network
         * @param exec_context Execution context to share with the reconstructed network
         * @return Unique pointer to reconstructed Network
         *
         * @throws std::runtime_error if archive is malformed or component reconstruction fails
         *
         * @note ComponentFactory dispatches to concrete component fromArchive_() methods.
         *       Each component is added via the factory method pattern to enforce context sharing.
         */
        static std::unique_ptr<Network> load(
            ModelArchive& archive,
            IExecutionContext* exec_context )
        {
            json net_meta = archive.readJson( "network/meta.json" );
            std::string name = net_meta.at( "name" );

            json arch = archive.readJson( "network/architecture.json" );

            if ( net_meta.contains( "num_components" ) )
            {
                size_t hint = net_meta.at( "num_components" ).get<size_t>();
                size_t actual = 0;

                if ( arch.is_array() )
                {
                    actual = arch.size();
                }

                if ( hint != actual )
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

            auto network = std::unique_ptr<Network>( new Network( name, exec_context ) );

            for ( const auto& item : arch )
            {
                std::string component_name;

                if ( item.is_object() && item.contains( "name" ) )
                {
                    component_name = item.at( "name" ).get<std::string>();
                }
                else if ( item.is_string() )
                {
                    component_name = item.get<std::string>();
                }
                else
                {
                    throw std::runtime_error(
                        "Network::load: unexpected architecture.json entry format"
                    );
                }

                try
                {
                    // ComponentFactory is expected to reconstruct a component instance
                    // that already shares the provided exec_context. Once available,
                    // it should be registered with the network so it participates in
                    // build/serialization.
                    //
                    // The actual dispatch to ComponentFactory::createFromArchive is
                    // TODO: implement in ComponentFactory. This is the documented hook.
                    throw std::runtime_error(
                        std::format(
                            "Network::load: ComponentFactory dispatch not yet implemented for component '{}'",
                            component_name
                        )
                    );
                }
                catch ( const std::exception& e )
                {
                    throw std::runtime_error(
                        std::format(
                            "Network::load: failed to reconstruct component '{}' from archive '{}': {}",
                            component_name,
                            archive.getFilepath(),
                            e.what()
                        )
                    );
                }
            }

            return network;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        // TJT: Temporary API

        IExecutionContext* getExecutionContext() const
        {
            return this->getExecutionContext();
        }

        /**
         * @brief Get the network name.
         *
         * @return Network name used for identification and serialization
         */
        /*std::string getName() const override
        {
            return name_;
        }*/

        /**
         * @brief Generate a human-readable description.
         *
         * @return String representation showing network name and children
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Network: " << this->getName() << " " << CompositeBase::toString();

            return oss.str();
        }

    protected:

        /**
         * @brief Register a pre-constructed component during deserialization.
         *
         * Internal helper used by load() to register components reconstructed by
         * ComponentFactory. This bypasses the factory method since the component
         * was already constructed with the correct ExecutionContext by the factory.
         *
         * @param name Component name for registration
         * @param component Component to register (must share network's ExecutionContext)
         *
         * @throws std::invalid_argument if name already exists
         * @throws std::runtime_error if called after build()
         */
        void registerReconstructedComponent( const std::string& name, ComponentPtr component )
        {
            if ( this->isBuilt() )
            {
                throw std::runtime_error(
                    "Network::registerReconstructedComponent: cannot add components after build()"
                );
            }

            if ( !component )
            {
                throw std::invalid_argument(
                    "Network::registerReconstructedComponent: component cannot be null"
                );
            }

            if ( this->hasComponent( name ) )
            {
                throw std::invalid_argument(
                    "Network::registerReconstructedComponent: component name '" + name + "' already exists"
                );
            }

            // Direct registration without using factory method since component was
            // already constructed by ComponentFactory with the correct ExecutionContext
            this->child_component_map_[ name ] = component;
            this->child_components_.push_back( component );
        }

    private:

        /**
         * @brief Create and validate owned execution context.
         *
         * Helper for the primary (DeviceId) constructor. Creates an ExecutionContext
         * for the specified device and validates that the device type matches the
         * Network's template parameter.
         *
         * @param device_id Device identifier for context creation
         * @return Unique pointer to created execution context
         *
         * @throws std::invalid_argument if device type mismatches template parameter
         * @throws std::runtime_error if context creation fails
         */
        IExecutionContext* createOwnedContext( DeviceId device_id )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format(
                        "Network: constructor device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type )
                    )
                );
            }

            owned_context_ = createExecutionContext( device_id );

            if ( !owned_context_ )
            {
                throw std::runtime_error(
                    "Network: failed to create execution context for device"
                );
            }

            return owned_context_.get();
        }

        /**
         * @brief Owned ExecutionContext (only populated for primary constructor).
         *
         * Ownership model:
         * - Primary constructor (DeviceId): owned_context_ is populated, passed to CompositeBase
         * - Secondary constructor (IExecutionContext*): owned_context_ is null, shares parent's context
         */
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        /**
         * @brief Network name for identification and serialization.
         */
        //std::string name_;

        friend class ComponentFactory;
    };
}