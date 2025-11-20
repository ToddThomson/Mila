module;
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <format>
#include <utility>

export module Dnn.ModuleRegistrar;

import Dnn.Module;
import Dnn.ModuleRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Helper that registers a creator at static initialization time.
     *
     * Typical use from a module:
     *   static ModuleRegistrar<DeviceType::Cpu> regCpu( "Gelu", geluCpuCreator );
     *
     * Registrar registers on construction and optionally unregisters on destruction.
     */
    export template<DeviceType TDeviceType>
        class ModuleRegistrar
    {
    public:
        using CreatorT = typename ModuleRegistry::template CreatorFn<TDeviceType>;

        ModuleRegistrar( const std::string& type, CreatorT creator )
            : type_( type )
        {
            ModuleRegistry::registerCreator<TDeviceType>( type_, std::move( creator ) );
        }

        ~ModuleRegistrar()
        {
            // Best-effort unregister at teardown
            ModuleRegistry::unregisterCreator<TDeviceType>( type_ );
        }

    private:
        std::string type_;
    };

    /**
     * @brief Manager that performs a one-time global registration pass for modules.
     *
     * Use this to explicitly call per-module registration functions in a single
     * deterministic place. Call `ModuleRegistrarManager::instance()` early (for
     * example from ModuleFactory::create) to ensure creators are registered.
     */
    export class ModuleRegistrarManager
    {
    public:
        static ModuleRegistrarManager& instance()
        {
            static ModuleRegistrarManager inst;
            if (!inst.initialized_)
            {
                inst.registerAll();
                inst.initialized_ = true;
            }

            return inst;
        }

        // No copy
        ModuleRegistrarManager( const ModuleRegistrarManager& ) = delete;
        ModuleRegistrarManager& operator=( const ModuleRegistrarManager& ) = delete;

    private:
        ModuleRegistrarManager() = default;

        // Forward declarations for per-module registration functions.
        // Implement these in the individual module translation units (e.g., Gelu.ixx).
        // Example:
        //   void registerGeluCreators();
        //   void registerLinearCreators();
        //
        // If a module does not provide a registration function, omit its call below.
        extern void registerGeluCreators();
        extern void registerLinearCreators();
        // Add other module registration function declarations here...

        void registerAll()
        {
            // Deterministic registration order; add module registration calls here.
            //
            // Example:
            //   registerGeluCreators();
            //   registerLinearCreators();
            //
            // Keep calls explicit so static init ordering is no longer an issue.

            // NOTE: Uncomment and add calls as modules implement the registration functions.
            // registerGeluCreators();
            // registerLinearCreators();

            // Blank line before function end per style
        }

        bool initialized_{ false };
    };
}