/**
 * @file Chat.ModelSize.ixx
 * @brief Parameter-count classes of the models Chat runs.
 */

export module Chat.ModelSize;

namespace Mila::ChatApp
{
    export enum class ModelSize
    {
        B1,  // 1B parameters (Llama 3.2 1B)
        B3,  // 3B parameters (Llama 3.2 3B)
        B8,  // 8B parameters (Llama 3.1 8B)
        B12  // 12B parameters (Gemma 4 12B)
    };
}
