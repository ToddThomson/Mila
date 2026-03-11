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
     * Holds the model backend selection, file paths, and generation
     * hyper-parameters. All fields have defaults suitable for GPT-2.
     *
     * ## context_length
     *
     * Controls the maximum sequence length allocated at build time.
     * This is a deployment constraint — it must not exceed the model's
     * architectural maximum but may be set lower to reduce GPU memory usage.
     *
     * Default is 0 (unset sentinel). parseArgs() resolves this to a
     * model-type-aware default:
     *   Gpt   — 1024  (GPT-2 architectural maximum)
     *   Llama — 4096  (consumer GPU safe default for Llama 3.x)
     *
     * Users on high-VRAM devices may increase this via --context-length.
     * fromPretrained() validates the final value against the architectural
     * maximum and throws if exceeded.
     */
    export struct ChatConfig
    {
        ModelType             model_type{ ModelType::Gpt };
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 512 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };
        size_t                context_length{ 0 };  // 0 = unset, resolved by parseArgs()
    };
}
