export module Serialization.Mode;

namespace Mila::Dnn::Serialization
{
	// TJT: Maybe bool flag for "include optimizer state" would be simpler?

    /**
     * @brief Serialization mode - what to save/load
     */
    export enum class SerializationMode
    {
        Checkpoint,     // Full state: config + weights + optimizer state
        Inference,      // Minimal: config + weights only (no optimizer state)
        WeightsOnly     // Just weights (assume architecture known)
    };
}