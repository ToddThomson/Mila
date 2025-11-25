module;
#include <string>
#include <vector>

export module Extensibility.PluginInfo;

namespace Mila::Dnn::Extensibility
{
    /**
    * @brief Get plugin metadata
    */
    export struct PluginInfo
    {
        std::string name;              // e.g., "MyCustomLayer"
        std::string version;           // e.g., "1.0.0"
        std::string mila_api_version;  // e.g., "0.1.0" - compatibility check
        std::vector<std::string> supported_devices;    // {"Cpu", "Cuda"}
        std::vector<std::string> supported_precisions; // {"float32", "float64"}
    };
}