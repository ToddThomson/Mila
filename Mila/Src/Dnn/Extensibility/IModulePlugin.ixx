/**
 * @brief Interface that all Mila module plugins must implement
 *
 * A plugin is a self-contained unit that provides:
 * - Module construction and serialization
 * - Backend compute operations
 * - Metadata about supported configurations
 */
module;
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

export module Extensibility.PluginInterface;

import Dnn.Component;
import Compute.OperationRegistry;
import Serialization.ModelArchive;
import Extensibility.PluginInfo;

namespace Mila::Dnn::Extensibility
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export class IModulePlugin
    {
    public:
        virtual ~IModulePlugin() = default;

        virtual PluginInfo getInfo() const = 0;

        /**
         * @brief Check if this plugin handles the given module type
         */
        virtual bool canHandle( const std::string& module_type ) const = 0;

        /**
         * @brief Create module instance from archive
         *
         * @param device_type Device as string ("Cpu", "Cuda")
         * @param precision Precision as string ("float32", etc.)
         * @param archive Archive to read from
         * @param module_name Module name in archive
         * @param exec_context Execution context (type-erased)
         * @return Type-erased module pointer
         */
		 
         // TJT: TODO, This needs a type-erased execution context and module pointer

        /*virtual std::unique_ptr<IModule> createFromArchive(
            const std::string& device_type,
            const std::string& precision,
            ModelArchive& archive,
            const std::string& module_name,
            void* exec_context ) const = 0;*/

        /**
         * @brief Register backend compute operations
         *
         * Called when plugin is loaded to register operations with
         * the compute backend (similar to how you register operations now).
         */
        virtual void registerOperations( Compute::OperationRegistry& registry ) = 0;
    };
}