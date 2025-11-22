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

        ///**
        // * @brief Allow derived/clients to customize child build ordering.
        // *
        // * Default behavior delegates to CompositeModule which builds all
        // * children with the provided input shape.
        // */
        //void buildImpl( const shape_t& input_shape ) override
        //{
        //    CompositeBase::buildImpl( input_shape );
        //}

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
			net_meta["mode"]  = static_cast<int>(mode);
            
            archive.writeJson( "network/meta.json", net_meta );

            // Save architecture (module names and types)
            json arch = json::array();
            for (const auto& [name, module] : modules)
            {
                arch.push_back( name );
            }
            archive.writeJson( "network/architecture.json", arch );

            // Save each module
            for (const auto& [name, module] : modules)
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
            json net_meta = archive.readJson( "network/meta.json" );
            std::string name = net_meta.at( "name" );

            json arch = archive.readJson( "network/architecture.json" );
            auto network = std::make_unique<Network>( exec_context, name );

            for (const auto& module_name_json : arch)
            {
                std::string module_name = module_name_json.get<std::string>();
                
                // FIXME:
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