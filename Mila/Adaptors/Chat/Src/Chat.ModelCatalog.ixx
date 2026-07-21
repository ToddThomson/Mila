/**
 * @file Chat.ModelCatalog.ixx
 * @brief Single source of truth mapping a model alias to its full descriptor.
 *
 * Both startup (the session config "model" key) and the in-session /model command
 * resolve aliases through this catalog, so model selection, weight/tokenizer paths,
 * and per-model defaults are defined in exactly one place. Adding a model is a
 * single table row.
 */

module;
#include <array>
#include <string_view>
#include <cstddef>

export module Chat.ModelCatalog;

import Chat.Config;

namespace Mila::ChatApp
{
    /**
     * @brief Everything needed to select and load a model from a short alias.
     *
     * weights_file and tokenizer_file are relative to the configured models
     * directory. default_quantization is used when the session config / command
     * omits an explicit quantization. default_context is the per-model maximum
     * sequence length used when the config does not override it (the primary VRAM
     * lever; Gemma 4 12B is deliberately conservative for a 12 GB card).
     *
     * streaming_capable marks models whose responses the chat harness streams to
     * the console live. True only for Gemma, whose tool calls are protocol tokens
     * detectable per-token; Llama's text-convention tool calls need the full
     * buffered response to parse, so those models stay buffered until their
     * deferred tool/sampler migration.
     */
    export struct ModelEntry
    {
        std::string_view alias;
        ModelType        family;
        ModelSize        size;
        ModelPrecision   precision;
        bool             is_instruct;
        bool             streaming_capable;
        QuantizationMode default_quantization;
        std::string_view weights_file;
        std::string_view tokenizer_file;
        std::size_t      default_context;
    };

    export inline constexpr std::array<ModelEntry, 8> kModelCatalog = { {
        { "gemma-12b",     ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true,  true,  QuantizationMode::FP4,  "gemma/gemma4_12b_it_bf16.bin",       "gemma/gemma_tokenizer.bin",  512 },
        { "llama-1b",      ModelType::Llama, ModelSize::B1,  ModelPrecision::BF16, true,  false, QuantizationMode::None, "llama/llama32_1b_instruct_bf16.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "llama-3b",      ModelType::Llama, ModelSize::B3,  ModelPrecision::BF16, true,  false, QuantizationMode::None, "llama/llama32_3b_instruct_bf16.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "llama-8b",      ModelType::Llama, ModelSize::B8,  ModelPrecision::BF16, true,  false, QuantizationMode::None, "llama/llama31_8b_instruct_bf16.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "llama-1b-fp32", ModelType::Llama, ModelSize::B1,  ModelPrecision::FP32, true,  false, QuantizationMode::None, "llama/llama32_1b_instruct_fp32.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "llama-3b-fp32", ModelType::Llama, ModelSize::B3,  ModelPrecision::FP32, true,  false, QuantizationMode::None, "llama/llama32_3b_instruct_fp32.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "llama-8b-fp32", ModelType::Llama, ModelSize::B8,  ModelPrecision::FP32, true,  false, QuantizationMode::None, "llama/llama31_8b_instruct_fp32.bin", "llama/llama32_tokenizer.bin", 4096 },
        { "gpt2",          ModelType::Gpt,   ModelSize::B3,  ModelPrecision::FP32, false, false, QuantizationMode::None, "gpt2/gpt2_small_fp32.bin",           "gpt2/gpt2_tokenizer.bin",    1024 },
    } };

    /// Alias used when the session config does not name a model.
    export inline constexpr std::string_view kDefaultModelAlias = "gemma-12b";

    /**
     * @brief Look up a model entry by alias, or nullptr when the alias is unknown.
     */
    export constexpr const ModelEntry* findModel( std::string_view alias )
    {
        for ( const auto& entry : kModelCatalog )
        {
            if ( entry.alias == alias )
                return &entry;
        }

        return nullptr;
    }
}
