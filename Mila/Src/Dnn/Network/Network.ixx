/**
 * @file Network.ixx
 * @brief Lightweight composite network container.
 *
 * Provides a simple CompositeModule-derived container that owns child modules
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

import Dnn.CompositeModule;
import Dnn.ModuleFactory;
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

    export template<DeviceType TDeviceType>
        class Network : public CompositeModule<TDeviceType>
    {
    public:
        using CompositeBase = CompositeModule<TDeviceType>;
        using ModulePtr = typename CompositeBase::ModulePtr;
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
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the network.
         *
         * Default behavior delegates to `CompositeModule::build` which builds all
         * child modules with the provided `input_shape`, validates children were
         * built and marks the composite built.
         *
         * Derived `Network` implementations may override this method to provide
         * custom shape propagation, build ordering, or to allocate network-level
         * buffers. When overriding, call `validateChildrenBuilt()` and set the
         * built flag (inherited `is_built_`) once children are successfully built.
         */
        virtual void build( const shape_t& input_shape ) override
        {
            CompositeBase::build( input_shape );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save network to archive
         *
         * Produces:
         *  - network/meta.json         : small metadata (name, format_version, export_time, mode, num_modules)
         *  - network/architecture.json : manifest (array of module descriptors: name, path, index)
         *  - modules/<name>/...        : each module's own save_() output
         *
         * Behavior:
         *  - Writes modules in a deterministic (sorted by name) order.
         *  - Wraps child saves with contextual diagnostics on error.
         */
        void save( ModelArchive& archive, SerializationMode mode ) const
        {
            // Gather named modules deterministically (sort names)
            const auto& named_map = this->getNamedModules();
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
            net_meta["num_modules"] = names.size();
            net_meta["mode"] = serializationModeToString( mode );

            // export_time as epoch seconds
            auto now = std::chrono::system_clock::now();
            net_meta["export_time"] = static_cast<int64_t>(std::chrono::system_clock::to_time_t( now ));

            archive.writeJson( "network/meta.json", net_meta );

            // Architecture manifest: array of objects with metadata per module
            json arch = json::array();
            for (size_t i = 0; i < names.size(); ++i)
            {
                const auto& nm = names[i];
                json entry = json::object();
                entry["name"] = nm;
                entry["path"] = "modules/" + nm;
                entry["index"] = static_cast<int>( i );
                arch.push_back( entry );
            }

            archive.writeJson( "network/architecture.json", arch );

            // Save each module using deterministic order; wrap exceptions with context
            for (const auto& nm : names)
            {
                auto it = named_map.find( nm );
                if (it == named_map.end())
                {
                    // This should not happen since we built names from named_map, but guard defensively.
                    throw std::runtime_error( "Network::save: inconsistent named modules map for '" + nm + "'" );
                }

                const auto& module = it->second;
                try
                {
                    module->save_( archive, mode );
                }
                catch (const std::exception& e)
                {
                    std::ostringstream oss;
                    oss << "Network::save: failed saving module '" << nm << "' into archive '"
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
         * Note: ModuleFactory integration is TODO; this method validates meta/architecture schema.
         */
        static std::unique_ptr<Network> load(
            ModelArchive& archive,
            std::shared_ptr<ExecutionContextType> exec_context )
        {
            json net_meta = archive.readJson( "network/meta.json" );
            std::string name = net_meta.at( "name" );

            json arch = archive.readJson( "network/architecture.json" );

            // Validate optional num_modules hint vs manifest size
            if (net_meta.contains( "num_modules" ))
            {
                size_t hint = net_meta.at( "num_modules" ).get<size_t>();
                size_t actual = 0;
                if (arch.is_array())
                {
                    actual = arch.size();
                }

                if (hint != actual)
                {
                    std::ostringstream oss;
                    oss << "Network::load: metadata num_modules (" << hint
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
                std::string module_name;

                if (item.is_object() && item.contains( "name" ))
                {
                    module_name = item.at( "name" ).get<std::string>();
                }
                else if (item.is_string())
                {
                    module_name = item.get<std::string>();
                }
                else
                {
                    throw std::runtime_error( "Network::load: unexpected architecture.json entry format" );
                }

                // FIXME: reconstruct child modules using ModuleFactory when available.
                (void)module_name;
                //auto module = ModuleFactory::create( archive, module_name, exec_context );
                //network->addModule( module_name, std::move( module ) );
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
            //auto ctx = context_.lock();
            //if (!ctx)
            //{
            //    throw std::runtime_error( "Network::getDevice: execution context expired" );
            //}

            return context_->getDevice();
        }

        void synchronize() override
        {
            //if (auto ctx = context_.lock())
            //{
            context_->synchronize();
            //}
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