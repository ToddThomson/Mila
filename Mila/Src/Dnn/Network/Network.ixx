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

export module Dnn.Network;

import Dnn.CompositeComponent;
import Dnn.ComponentFactory;
import Dnn.TensorDataType;
import Compute.ExecutionContext;
import Compute.DeviceType;
import Compute.ComputeDevice;
import Serialization.ModelArchive;
import Serialization.Mode;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        class Network : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using CompositeBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeBase::ComponentPtr;
        using ExecutionContextType = ExecutionContext<TDeviceType>;

    protected:

        explicit Network(
            std::shared_ptr<ExecutionContext<TDeviceType>> context, const std::string& name )
            : context_( std::move( context ) ), name_( name )
        {
            //validateInputs();
        }

    public:

        ~Network() override = default;

        // ====================================================================
        // Lifecycle - 
        // ====================================================================

        /**
         * @brief Build the network.
         *
         * Default behavior delegates to `CompositeComponent::build` which builds all
         * child components with the provided `input_shape`, validates children were
         * built and marks the composite built.
         *
         * Derived `Network` implementations may override this method to provide
         * custom shape propagation, build ordering, or to allocate network-level
         * buffers. When overriding, call `validateChildrenBuilt()` and set the
         * built flag (inherited `is_built_`) once children are successfully built.
         */
        /*virtual void build( const shape_t& input_shape ) override
        {
            // REVIEW: Not required. No Op
            CompositeBase::build( input_shape );
        }*/

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
            // Gather named components deterministically (sort names)
            const auto& named_map = this->getNamedComponents();
            std::vector<std::string> names;
            names.reserve( named_map.size() );
            for (const auto& p : named_map)
            {
                names.push_back( p.first );
            }
            std::sort( names.begin(), names.end() );

            // Network metadata
            json net_meta;
            net_meta["format_version"] = 1;
            net_meta["name"] = name_;
            net_meta["num_components"] = names.size();
            net_meta["mode"] = serializationModeToString( mode );

            // export_time as epoch seconds
            auto now = std::chrono::system_clock::now();
            net_meta["export_time"] = static_cast<int64_t>(std::chrono::system_clock::to_time_t( now )) ;

            archive.writeJson( "network/meta.json", net_meta );

            // Architecture manifest: array of objects with metadata per component
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

            // Save each component using deterministic order; call child's save_() under a scoped prefix.
            for (const auto& nm : names)
            {
                auto it = named_map.find( nm );
                if (it == named_map.end())
                {
                    // This should not happen since we built names from named_map, but guard defensively.
                    throw std::runtime_error( "Network::save: inconsistent named components map for '" + nm + "'" );
                }

                const auto& component = it->second;
                try
                {
                    // Scope the archive to components/<name>/ so leaf save_() implementations
                    // write relative files like "meta.json", "config.json", "tensors/weight", etc.
                    ModelArchive::ScopedScope scope( archive, std::string( "components/" ) + nm );
                    component->save_( archive, mode );
                }
                catch (const std::exception& e)
                {
                    std::ostringstream oss;
                    oss << "Network::save: failed saving component '" << nm << "' into archive '"
                        << archive.getFilepath() << "': " << e.what();

                    throw std::runtime_error( oss.str() );
                }
            }
        }

        /**
         * @brief Load network from archive
         *
         * Reconstructs a Network using the provided execution context. The
         * context is stored as non-owning (weak_ptr) in the returned Network.
         *
         * Note: ComponentFactory integration is TODO; this method validates meta/architecture schema.
         */
        static std::unique_ptr<Network> load(
            ModelArchive& archive,
            std::shared_ptr<ExecutionContextType> exec_context )
        {
            json net_meta = archive.readJson( "network/meta.json" );
            std::string name = net_meta.at( "name" );

            json arch = archive.readJson( "network/architecture.json" );

            // Validate optional num_components hint vs manifest size
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
                    std::ostringstream oss;
                    oss << "Network::load: metadata num_components (" << hint
                        << ") does not match architecture.json size (" << actual << ")";
                    throw std::runtime_error( oss.str() );
                }
            }

            auto network = std::make_unique<Network>( exec_context, name );

            // Support two manifest shapes:
            //  - legacy: ["mod1","mod2",...]
            //  - new: [{ "name": "...", "path": "...", "index": ... }, ...]
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
                    throw std::runtime_error( "Network::load: unexpected architecture.json entry format" );
                }

                // FIXME: reconstruct child components using ComponentFactory when available.
                (void)component_name;
                //auto component = ComponentFactory::create( archive, component_name, exec_context );
                //network->addComponent( std::move( component ) );
            }

            return network;
        }

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return name_;
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return context_->getDevice();
        }

        void synchronize() override
        {
            context_->synchronize();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Network: " << name_ << CompositeBase::toString();

            return oss.str();
        }

    protected:

        std::shared_ptr<ExecutionContextType> getExecutionContext() const
        {
            return context_;
        }

    private:
        std::shared_ptr<ExecutionContext<TDeviceType>> context_;
        std::string name_;

        static std::shared_ptr<ExecutionContext<TDeviceType>> makeExecutionContext( int device_id )
        {
            if constexpr (TDeviceType == DeviceType::Cuda)
            {
                return std::make_shared<ExecutionContext<TDeviceType>>( device_id );
            }
            else
            {
                return std::make_shared<ExecutionContext<TDeviceType>>();
            }
        }
    };
}