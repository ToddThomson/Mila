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

export module Dnn.Network;

import Dnn.CompositeModule;
//import Dnn.ModuleFactory;
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

        /**
         * @brief Construct a Network with an execution context and name.
         *
         * Network does not own the execution context; it stores a weak reference.
         *
         * @param context Shared execution context (must not be null/expired).
         * @param name Human readable name for diagnostics (must not be empty).
         */
        explicit Network( std::shared_ptr<ExecutionContextType> context, const std::string& name )
            : context_( std::move( context ) ), name_( name )
        {
            if ( name_.empty() )
            {
                throw std::invalid_argument( "Network name cannot be empty." );
            }

            if ( context_.expired() )
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }
        }

        ~Network() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Allow derived/clients to customize child build ordering.
         *
         * Default behavior delegates to CompositeModule which builds all
         * children with the provided input shape.
         */
        void buildImpl( const shape_t& input_shape ) override
        {
            CompositeBase::buildImpl( input_shape );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save network to archive
         */
        void save( ModelArchive& archive, SerializationMode mode ) const
        {
            const auto& modules = this->getModules();

            json net_meta;
            net_meta["name"] = name_;
            net_meta["num_modules"] = modules.size();
            archive.writeJson( "network/meta.json", net_meta );

            // Save architecture (module names and types)
            json arch = json::array();
            for (const auto& module : modules)
            {
                arch.push_back( module->getName() );
            }

            archive.writeJson( "network/architecture.json", arch );

            for (const auto& module : modules)
            {
                module->save_( archive, mode );
            }
        }

        /**
         * @brief Load network from archive
         *
         * Reconstructs a Network using the provided execution context. The
         * context is stored as non-owning (weak_ptr) in the returned Network.
         */
        static std::unique_ptr<Network> load(
            ModelArchive& archive,
            std::shared_ptr<ExecutionContextType> exec_context )
        {
            // Load network metadata
            json net_meta = archive.readJson( "network/meta.json" );

            // Read name
            std::string name;
            net_meta.at( "name" ).get_to( name );

            // Load architecture
            json arch = archive.readJson( "network/architecture.json" );

            auto network = std::make_unique<Network>( std::move( exec_context ), name );

            // Reconstruct each module via factory and add to network
            for (const auto& module_name_json : arch)
            {
                std::string module_name = module_name_json.get<std::string>();

                // FIXME:
                //auto module = ModuleFactory::create( archive, module_name, network->context_.lock() );
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
            auto ctx = context_.lock();
            if (!ctx)
            {
                throw std::runtime_error( "Network::getDevice: execution context expired" );
            }

            return ctx->getDevice();
        }

        void synchronize() override
        {
            if (auto ctx = context_.lock())
            {
                ctx->synchronize();
            }
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
            auto ctx = context_.lock();
            if (!ctx)
            {
                throw std::runtime_error( "Network::getExecutionContext: execution context expired" );
            }
            
            return ctx;
		}

    private:
        // Non-owning reference to execution context to avoid taking shared ownership.
        std::weak_ptr<ExecutionContextType> context_;
        std::string name_;
    };
}