/**
 * @brief Plugin manager (placeholder)
 *
 * Future: Will handle dynamic loading of plugin shared libraries.
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <exception>

export module Extensibility.PluginManager;

import Extensibility.PluginInterface;
import Extensibility.PluginInfo;
import Utils.Logger;

namespace Mila::Dnn::Extensibility
{
    //using namespace Mila::Dnn::Compute;
    //using namespace Mila::Dnn::Serialization;

    /**
     * @brief Manages loading and querying of module plugins
     */
    export class PluginManager
    {
    public:
        static PluginManager& instance()
        {
            static PluginManager mgr;
            return mgr;
        }

        /**
         * @brief Load a plugin from shared library
         *
         * @param plugin_path Path to .so/.dll file
         * @throws std::runtime_error if load fails or version incompatible
         */
        void loadPlugin( const std::string& plugin_path )
        {
            // Platform-specific: dlopen/LoadLibrary
            void* handle = loadLibrary( plugin_path );
            if (!handle)
            {
                throw std::runtime_error(
                    std::format( "Failed to load plugin: {}", plugin_path ) );
            }

            // Get plugin entry point
            auto create_fn = reinterpret_cast<IModulePlugin * (*)()>(
                getSymbol( handle, "mila_create_plugin" ));

            if (!create_fn)
            {
                unloadLibrary( handle );
                throw std::runtime_error(
                    std::format( "Plugin missing entry point: {}", plugin_path ) );
            }

            // Create plugin instance
            std::unique_ptr<IModulePlugin> plugin( create_fn() );

            // Check version compatibility
            auto info = plugin->getInfo();
            if (!isCompatible( info.mila_api_version ))
            {
				// FIXME: getAPIVersion() not accessible??
                //throw std::runtime_error(
                //    std::format( "Plugin API version {} incompatible with Mila {}",
                //        info.mila_api_version, Mila::getAPIVersion() ) );
            }

            // Register operations
            plugin->registerOperations( Compute::OperationRegistry::instance() );

            // Store plugin
            plugins_.push_back( {
                .handle = handle,
                .plugin = std::move( plugin ),
                .path = plugin_path
                } );
        }

        /**
         * @brief Load all plugins from a directory
         */
        void loadPluginsFromDirectory( const std::string& dir_path )
        {
            for (const auto& entry : std::filesystem::directory_iterator( dir_path ))
            {
                if (entry.path().extension() == PLUGIN_EXTENSION)
                {
                    try
                    {
                        loadPlugin( entry.path().string() );
                    }
                    catch (const std::exception& e)
                    {
                        // Log warning but continue loading other plugins
                        // FIXME: Utils::Logger:: warning_fmt( "Warning: Failed to load plugin {}: {}", entry.path(), e.what() );
                    }
                }
            }
        }

        /**
         * @brief Find plugin that can handle given module type
         */
        IModulePlugin* findPlugin( const std::string& module_type ) const
        {
            for (const auto& entry : plugins_)
            {
                if (entry.plugin->canHandle( module_type ))
                {
                    return entry.plugin.get();
                }
            }
            return nullptr;
        }

        /**
         * @brief List all loaded plugins
         */
        std::vector<PluginInfo> listPlugins() const
        {
            std::vector<PluginInfo> infos;
            for (const auto& entry : plugins_)
            {
                infos.push_back( entry.plugin->getInfo() );
            }
            return infos;
        }

        ~PluginManager()
        {
            // Unload all plugins
            for (auto& entry : plugins_)
            {
                entry.plugin.reset();  // Destroy plugin first
                unloadLibrary( entry.handle );
            }
        }

    private:
        struct PluginEntry
        {
            void* handle;  // dlopen/LoadLibrary handle
            std::unique_ptr<IModulePlugin> plugin;
            std::string path;
        };

        std::vector<PluginEntry> plugins_;

#ifdef _WIN32
        static constexpr const char* PLUGIN_EXTENSION = ".dll";
#else
        static constexpr const char* PLUGIN_EXTENSION = ".so";
#endif

        bool isCompatible( const std::string& plugin_api_version ) const
        {
            // Simple semantic versioning check
            // Could be more sophisticated (major.minor matching, etc.)
            return (true); //FIXME: plugin_api_version == MILA_API_VERSION;
        }

        // Platform-specific helpers
        void* loadLibrary( const std::string& path );
        void* getSymbol( void* handle, const char* name );
        void unloadLibrary( void* handle );
    };
}

// Plugin entry point signature (for future reference)
// extern "C" {
//     MILA_PLUGIN_EXPORT IModulePlugin* mila_create_plugin();
// }