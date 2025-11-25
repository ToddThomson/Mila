module;
#include <string>

export module Serialization.Mode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Serialization mode - what to save/load
     */
    export enum class SerializationMode
    {
        Checkpoint,
        WeightsOnly,
        Architecture
    };

    export std::string serializationModeToString( SerializationMode mode )
    {
        switch (mode)
        {
            case SerializationMode::Checkpoint:
                return "Checkpoint";
            case SerializationMode::WeightsOnly:
                return "WeightsOnly";
            case SerializationMode::Architecture:
                return "Architecture";
            default:
                return "Unknown";
        }
	}
}