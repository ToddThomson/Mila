module;
#include <string>

export module Serialization.Mode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Modes for serialization and deserialization.
     *
     * Selects which parts of a network are saved or restored:
     * - Checkpoint: full state (architecture + weights + optimizer state)
     * - WeightsOnly: only parameter tensors
     * - Architecture: only network topology and metadata
     */
    export enum class SerializationMode
    {
        Checkpoint,
        WeightsOnly,
        Architecture
    };

    /**
     * @brief Convert a SerializationMode value to a human-readable string.
     *
     * @param mode The serialization mode to convert.
     * @return A short string identifying the mode ("Checkpoint", "WeightsOnly", or "Architecture").
     */
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