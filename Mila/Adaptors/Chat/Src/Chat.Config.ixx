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
#include <string_view>
#include <cstddef>

export module Chat.Config;

namespace Mila::ChatApp
{
    export enum class ModelType
    {
        Gpt,
        Llama,
        Gemma
    };

    export enum class ModelSize
    {
        B1,  // 1B parameters (Llama 3.2 1B)
        B3,  // 3B parameters (Llama 3.2 3B)
        B8,  // 8B parameters (Llama 3.1 8B)
        B12  // 12B parameters (Gemma 4 12B)
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
     * @brief Display-verbosity ladder for a chat turn. Each level includes the lower ones.
     *
     * Controls how much of the model's internal activity is shown — independent of
     * whether thinking mode is active (that is the model-side toggle). Off shows the
     * answer plus the always-visible agentic trace (tool calls); Thoughts adds the
     * reasoning channel; All adds raw model output plus INFO logging and load dumps.
     * (Tool calls are conversational content, always shown, so they are no longer a
     * verbosity level of their own.)
     */
    export enum class DetailLevel
    {
        Off,       ///< Answer + agentic trace (tool calls).
        Thoughts,  ///< + the reasoning channel.
        All,       ///< + raw model output, INFO logging, and model/memory load dumps.
    };

    /**
     * @brief Parse a detail keyword ("off"/"thoughts"/"all") to a DetailLevel.
     */
    export constexpr std::optional<DetailLevel> parseDetailLevel( std::string_view s )
    {
        if ( s.empty() || s == "off" || s == "none" ) return DetailLevel::Off;
        if ( s == "thoughts" )                        return DetailLevel::Thoughts;
        if ( s == "all" )                             return DetailLevel::All;
        return std::nullopt;
    }

    /**
     * @brief Display name for a DetailLevel.
     */
    export constexpr std::string_view detailLevelName( DetailLevel level )
    {
        switch ( level )
        {
            case DetailLevel::Off:      return "off";
            case DetailLevel::Thoughts: return "thoughts";
            case DetailLevel::All:      return "all";
        }

        return "off";
    }

    /**
     * @brief Parse a quantization keyword ("none"/"fp8"/"fp4") to a QuantizationMode.
     *
     * Returns std::nullopt for an unrecognized value. Shared by the session-config
     * loader and the in-session /model command so both accept the same vocabulary.
     */
    export constexpr std::optional<QuantizationMode> parseQuantization( std::string_view s )
    {
        if ( s.empty() || s == "none" ) return QuantizationMode::None;
        if ( s == "fp8" )               return QuantizationMode::FP8;
        if ( s == "fp4" )               return QuantizationMode::FP4;
        return std::nullopt;
    }

    /**
     * @brief Display name for a QuantizationMode ("none"/"fp8"/"fp4").
     */
    export constexpr std::string_view quantizationName( QuantizationMode mode )
    {
        switch ( mode )
        {
            case QuantizationMode::FP8: return "fp8";
            case QuantizationMode::FP4: return "fp4";
            case QuantizationMode::None: return "none";
        }

        return "none";
    }

    /**
     * @brief Runtime configuration for a Chat session.
     *
     * Holds the model backend selection, size, precision, file paths, and
     * generation hyper-parameters. All fields are plain value types so the
     * struct is cheap to copy and requires no JSON dependency.
     *
     * ## model selection
     *
     * model_type / precision / is_instruct / quantization_mode /
     * model_path / tokenizer_path are all resolved from a single model alias via
     * the ModelEntry catalog (see Chat.ModelCatalog), either at startup (the
     * session config "model" key) or by the /model command. They are not set
     * field-by-field and are never inferred from the weight filename.
     *
     * ## context_length
     *
     * Maximum sequence length allocated at model build time, and the primary VRAM
     * lever. Defaults to the selected model's per-model default (catalog
     * default_context); the session config "context_length" key overrides it.
     *
     * ## config_path
     *
     * Path to the JSON session config the run was built from (the single startup
     * source of truth). Selected with --config; otherwise the default
     * Data/session.json next to the executable.
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
        ModelPrecision        precision{ ModelPrecision::BF16 };
        QuantizationMode      quantization_mode{ QuantizationMode::None };

        /// True when quantization_mode is a load-time choice rather than what the artifact
        /// already is. A pre-quantized model carries it in its name; a dynamic one does not,
        /// and that is the only case where quoting it tells the reader something.
        bool                  quantization_applied_at_load{ false };
        bool                  is_instruct{ false };
        bool                  streaming_capable{ false };  ///< Live token-streaming display (from the model catalog).
        bool                  show_thinking{ false };  ///< Thinking mode: activate the model's reasoning (<|think|>).
        int                   thinking_effort{ 3 };    ///< 1..5 token-budget scale for the reasoning (when thinking on).
        DetailLevel           detail{ DetailLevel::Off };  ///< Display verbosity: thoughts / tool calls / all.
        /// Catalog alias the current model came from. This, not the family/size/precision
        /// triple, is what identifies a model: two entries can share an architecture and
        /// quantization while pointing at different weights -- a coordinate resolved from the
        /// store, and a converted .bin still awaiting migration.
        std::string           model_name;

        /// Why nothing is selected, when model_name is empty. A store with no usable model is a
        /// working session rather than a fatal condition -- /install and /models live inside the
        /// session, so exiting here is what left a clean machine unable to get its first model.
        std::string           no_model_reason;

        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 2048 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };
        size_t                context_length{ 0 };

        std::optional<std::filesystem::path> config_path;
        std::optional<std::filesystem::path> system_prompt_path;
    };
}
