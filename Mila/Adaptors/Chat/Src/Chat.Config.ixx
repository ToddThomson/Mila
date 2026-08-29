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
#include <string>
#include <string_view>
#include <cstddef>

export module Chat.Config;

namespace Mila::ChatApp
{
    export enum class ModelType
    {
        Gpt,
        Llama,
        Gemma,
        Qwen
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

        /**
         * The family's own per-role bit allocation, from pre-quantized weights
         * (WeightQuantization::Plan). Qwen 3.8 spends 2 and 3 bits per weight across the body
         * against 4.125 on full attention, which averages 2.82.
         *
         * Unlike the three above this can never be a load-time choice: a codebook is fitted
         * offline against calibration data, so the weights either carry the codes or cannot
         * be built this way at all.
         */
        Codebook,
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
     * @brief Parse a quantization keyword ("none"/"fp8"/"fp4"/"cb2-3").
     *
     * Returns std::nullopt for an unrecognized value. Shared by the session-config
     * loader and the in-session /model command so both accept the same vocabulary.
     */
    export constexpr std::optional<QuantizationMode> parseQuantization( std::string_view s )
    {
        if ( s.empty() || s == "none" )  return QuantizationMode::None;
        if ( s == "fp8" )                return QuantizationMode::FP8;
        if ( s == "fp4" )                return QuantizationMode::FP4;

        // Accepted although it can never be applied at load, so that a user who names the
        // variant a model already IS gets the load they asked for rather than a vocabulary
        // error. Naming it for a model that is not one is refused where the weights are
        // known, which is the only place the difference can be seen.
        if ( s == "cb2-3" )              return QuantizationMode::Codebook;

        return std::nullopt;
    }

    /**
     * @brief Display name for a QuantizationMode. The spelling parseQuantization accepts.
     */
    export constexpr std::string_view quantizationName( QuantizationMode mode )
    {
        switch ( mode )
        {
            case QuantizationMode::FP8: return "fp8";
            case QuantizationMode::FP4: return "fp4";
            case QuantizationMode::Codebook: return "cb2-3";
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
     * lever. Resolved by merging the layers of ChatConfiguration.md section 3: the
     * family default, then any file or flag that names the key, clamped to what the
     * architecture can address.
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

        /// True when quantization_mode is a load-time choice rather than what the weights
        /// already are. A pre-quantized model carries it in its name; a dynamic one does not,
        /// and that is the only case where quoting it tells the reader something.
        bool                  quantization_applied_at_load{ false };
        bool                  is_instruct{ false };
        bool                  streaming_capable{ false };  ///< Live token-streaming display (from the model catalog).

        /// Whether this model HAS a reasoning channel, from the model catalog. A capability, not a
        /// preference: it is why show_thinking is no longer a session setting -- a config key
        /// cannot give Llama a channel it does not have, and offering one only ever misreported.
        bool                  thinking_capable{ false };
        bool                  show_thinking{ false };  ///< Reasoning surfaced. Follows thinking_capable.
        int                   thinking_effort{ 3 };    ///< 1..5 token-budget scale for the reasoning (when thinking on).
        DetailLevel           detail{ DetailLevel::Off };  ///< Display verbosity: thoughts / tool calls / all.
        /// Catalog alias the current model came from. This, not the family/size/precision
        /// triple, is what identifies a model: two entries can share an architecture and
        /// quantization while pointing at different weights -- a coordinate resolved from the
        /// store, and a converted .bin still awaiting migration.
        std::string           model_name;

        /// Lineage of the loaded model, from its store record. `license` is the identifier the
        /// manifest declares (llama3.1, apache-2.0), not the text; the text ships with the
        /// model on the hub. Both are empty for a model whose record declares neither.
        std::string           base_model;
        std::string           license;

        /// Why nothing is selected, when model_name is empty. A store with no usable model is a
        /// working session rather than a fatal condition -- /model install and /model list live inside
        /// session, so exiting here is what left a clean machine unable to get its first model.
        std::string           no_model_reason;

        /**
         * @brief Which CUDA device the session runs on, as the N in the runtime's `CUDA:N`.
         *
         * A deployment decision like context length, not an identity: the same model on the
         * other card is the same model. It reaches every place that asks the hardware a
         * question -- the load, the footprint pre-flight, and the fit column -- because a
         * verdict measured against one card and a load onto another would silently disagree.
         *
         * Note this is the CUDA ordinal, which is NOT nvidia-smi's index: the two orders differ
         * on a mixed-generation rig, and a mismatched pick fails as an out-of-memory abort
         * rather than as a wrong-card message.
         */
        int                   device_index{ 0 };

        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        size_t                max_new_tokens{ 2048 };
        float                 temperature{ 0.8f };
        int                   top_k{ 40 };

        /// Nucleus truncation. 1.0 disables it, matching the runtime's own default -- Chat was the
        /// only consumer of the sampler not passing this knob through.
        float                 top_p{ 1.0f };
        size_t                context_length{ 0 };

        /// What was asked for, kept apart from the live value because a model switch rewrites the
        /// latter. 0 means no layer above the compiled defaults named a context, so the new
        /// family's own default stands. Without this a switch dropped to a family default -- 512
        /// for Gemma -- silently discarding the number the user actually configured.
        size_t                configured_context_length{ 0 };

        /// Which layer named the context, for /context to report. Carried as the rendered string
        /// rather than the layer, because the merged settings know the FILE a layer wrote from and
        /// the session does not hold them -- threading MergedSettings in to recover one string
        /// would put the whole resolution order behind a display decision.
        std::string           context_origin;

        /// True when context_length was measured from the device rather than named by a layer.
        /// A switch re-measures it: auto means "whatever fits the card", and what fits depends on
        /// the model, so carrying a number derived for the previous one would be the same defect
        /// configured_context_length exists to prevent.
        bool                  context_is_automatic{ false };

        std::optional<std::filesystem::path> system_prompt_path;
    };
}
