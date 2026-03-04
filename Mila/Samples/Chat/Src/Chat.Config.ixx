/**
 * @file Chat.Config.ixx
 * @brief Configuration for the Mila chat application.
 *
 * Provides ChatConfig and ModelType used to select and parameterize the inference backend.
 */

module;
#include <filesystem>
#include <cstddef>

export module Chat.Config;

namespace Mila::ChatApp
{
    export enum class ModelType
    {
        Gpt,
        Llama
    };

    /**
     * @brief Runtime configuration for a Chat session.
     *
     * Holds the model backend selection, file paths, and generation hyper-parameters.
     * All fields have defaults that target GPT-2 under MODELS_DIR.
     */
    export struct ChatConfig
    {
        ModelType             model_type{ ModelType::Gpt };
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 512 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };
    };
}