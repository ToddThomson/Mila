/**
 * @file Chat.Config.ixx
 * @brief Configuration for the Mila chat application.
 *
 * Provides ChatConfig, ModelType, ModelSize, and ModelPrecision used to
 * select and parameterize the inference backend.
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

    export enum class ModelSize
    {
        B1,  // 1B parameters (Llama 3.2 1B)
        B3   // 3B parameters (Llama 3.2 3B)
    };

    export enum class ModelPrecision
    {
        FP32,
        BF16
    };

    /**
     * @brief Runtime configuration for a Chat session.
     *
     * Holds the model backend selection, size, precision, file paths, and
     * generation hyper-parameters.
     *
     * ## Defaults
     *
     * The defaults are tuned for Llama 3.2 3B BF16, which is the recommended
     * configuration for consumer GPU inference.
     *
     * ## context_length
     *
     * Controls the maximum sequence length allocated at build time.
     * Must not exceed the model's architectural maximum; may be set lower
     * to reduce GPU memory usage. parseArgs() resolves 0 (unset) to a
     * model-type-aware default:
     *   Gpt   — 1024  (GPT-2 architectural maximum)
     *   Llama — 4096  (consumer GPU safe default for Llama 3.x)
     *
     * ## precision
     *
     * Selects the weight dtype used by the model at load time. Inferred
     * from the model filename (bf16/fp32 substring) when not explicitly set.
     * Must match the dtype stored in the weights file.
     *
     * ## model_size
     *
     * Selects the Llama parameter count variant. Ignored for GPT models.
     * Inferred from the model filename (_1b_/_3b_ substring) when not
     * explicitly set via --model-size.
     */
    export struct ChatConfig
    {
        ModelType             model_type{ ModelType::Llama };
        ModelSize             model_size{ ModelSize::B3 };
        ModelPrecision        precision{ ModelPrecision::BF16 };
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 512 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };
        size_t                context_length{ 0 };  // 0 = unset, resolved by parseArgs()
    };
}
