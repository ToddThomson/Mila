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
}