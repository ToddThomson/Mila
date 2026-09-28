/**
 * @file Chat.ModelType.ixx
 * @brief The model families Chat runs.
 */

export module Chat.ModelType;

namespace Mila::ChatApp
{
    // Instruct families only. Chat refuses a base model at the catalogue (Chat.ModelCatalog.ixx),
    // so GPT-2 has no entry here: the harness renders turns, applies a chat template and keeps
    // history, none of which a base model reads. GPT-2 remains a first-class Mila model; it is
    // Chat that does not run it.
    export enum class ModelType
    {
        Llama,
        Gemma,
        Qwen
    };
}
