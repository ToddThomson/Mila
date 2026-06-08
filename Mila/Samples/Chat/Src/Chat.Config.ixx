/**
 * @file Chat.Config.ixx
 * @brief Configuration for the Mila chat application.
 *
 * Provides ChatConfig, ModelType, ModelSize, and ModelPrecision used to
 * select and parameterize the inference backend.
 */

module;
#include <filesystem>
#include <optional>
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
        B3,  // 3B parameters (Llama 3.2 3B)
        B8   // 8B parameters (Llama 3.1 8B)
    };

    export enum class ModelPrecision
    {
        FP32,
        BF16
    };

    export enum class QuantizationMode
    {
        None,  ///< BF16 weights, no KV cache compression — default.
        FP8,   ///< FP8 weights + FP8 KV cache (PerChannelFp8 + PerChannelKvFp8).
        FP4,   ///< INT4 weights + FP8 KV cache (PerGroupInt4 + PerChannelKvFp8). W4A16 kernel path.
    };

    /**
     * @brief Runtime configuration for a Chat session.
     *
     * Holds the model backend selection, size, precision, file paths, and
     * generation hyper-parameters. All fields are plain value types so the
     * struct is cheap to copy and requires no JSON dependency.
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
     *
     * ## config_path
     *
     * Optional path to a JSON session config file. When set, parseArgs()
     * loads it first and applies it as a baseline; explicit CLI arguments
     * override any values it provides.
     *
     * ## system_prompt_path
     *
     * Optional path to a JSON file containing a system_prompt string and
     * an optional tools array. Loaded by Chat on construction. When absent
     * no system message is prepended and tool calling is disabled.
     */
    export struct ChatConfig
    {
        ModelType             model_type{ ModelType::Llama };
        ModelSize             model_size{ ModelSize::B3 };
        ModelPrecision        precision{ ModelPrecision::BF16 };
        QuantizationMode      quantization_mode{ QuantizationMode::None };
        bool                  is_instruct{ false };
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 2048 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };
        size_t                context_length{ 0 };

        std::filesystem::path                 models_dir;
        std::optional<std::filesystem::path> config_path;
        std::optional<std::filesystem::path> system_prompt_path;
    };
}
