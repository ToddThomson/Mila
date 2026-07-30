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
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <cstddef>

export module Chat.ModelCatalog;

import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    /**
     * @brief Everything needed to select and load a model from a short alias.
     *
     * weights_file is either a path relative to the configured models directory, or a
     * HuggingFace coordinate of the form `<organization>/<repository>[:<variant>]`. When it
     * is a coordinate the artifact is fetched and cached on first use and tokenizer_file is
     * ignored, because the repository manifest names the tokenizer too.
     *
     * default_quantization is used when the session config / command omits an explicit
     * quantization. default_context is the per-model maximum sequence length used when the
     * config does not override it (the primary VRAM lever; Gemma 4 12B is deliberately
     * conservative for a 12 GB card).
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

    export inline constexpr std::array<ModelEntry, 10> kModelCatalog = { {
        { "gemma-12b",     ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true,  true,  QuantizationMode::FP4,  "gemma/gemma4_12b_it_bf16.bin",       "gemma/gemma_tokenizer.bin",  512 },
        // The same model from a pre-quantized safetensors artifact: 6.33 GB rather than
        // 23.8 GB, with the FP4 packing done once by Tools/ExportArtifact instead of on
        // every load. Quantization must stay FP4 -- the artifact declares the policy its
        // bytes were packed with and a mismatched load is refused.
        //
        // A local file of this name still wins if one is present, so an exported artifact
        // in the models directory is used without touching the network. Otherwise this
        // resolves as a coordinate and is fetched once into the content-addressed cache.
        { "gemma-12b-packed", ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true, true, QuantizationMode::FP4, "gemma/gemma4_12b_it_fp4.safetensors", "gemma/gemma_tokenizer.bin", 512 },
        { "gemma-12b-hub",    ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true, true, QuantizationMode::FP4, "mila-llm/gemma-4-12b-it:fp4",         "gemma/gemma_tokenizer.bin", 512 },
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

    /**
     * @brief Where an entry's weights and tokenizer actually live.
     */
    export struct EntryPaths
    {
        std::filesystem::path weights;
        std::filesystem::path tokenizer;
    };

    /**
     * @brief Resolve an entry to concrete paths, fetching from HuggingFace if it names a repository.
     *
     * Startup and the in-session /model command both go through here, so a coordinate behaves
     * identically whichever way a model is selected.
     *
     * Progress is printed only while a transfer is actually running: a cached artifact resolves
     * with no output at all, which is what makes a second run feel instant rather than merely be
     * instant.
     *
     * @throws std::runtime_error if a coordinate cannot be resolved, or if the entry names one
     *         in a build compiled without model download.
     */
    export EntryPaths resolveEntryPaths(
        const ModelEntry& entry, const std::filesystem::path& models_dir )
    {
        const std::string weights_spec( entry.weights_file );

        // A relative file under the models directory is the historical form and stays the
        // fast path: no parsing, no cache, no network.
        const auto local_candidate = models_dir / weights_spec;

        if ( std::filesystem::exists( local_candidate ) )
        {
            return { local_candidate, models_dir / std::string( entry.tokenizer_file ) };
        }

#ifdef MILA_HAS_MODEL_DOWNLOAD
        const auto coordinate = Mila::Distribution::parseCoordinate( weights_spec );

        if ( !coordinate.has_value() )
        {
            throw std::runtime_error( std::format(
                "Model '{}': '{}' is not under {} and is not a HuggingFace coordinate",
                entry.alias, weights_spec, models_dir.string() ) );
        }

        std::cout << std::format( "Resolving {}\n", coordinate->toString() );

        bool reported = false;

        auto progress = [&reported]( uint64_t received, uint64_t total ) -> bool
            {
                // Coarse on purpose: this is a console line, not a UI, and a 6 GB transfer
                // does not want a redraw per chunk.
                if ( total == 0 )
                {
                    return true;
                }

                const int percent = static_cast<int>( ( received * 100 ) / total );

                if ( percent % 5 == 0 )
                {
                    std::cout << std::format( "\r  {:>3}%  {:.2f} / {:.2f} GB", percent,
                        static_cast<double>( received ) / ( 1024.0 * 1024.0 * 1024.0 ),
                        static_cast<double>( total ) / ( 1024.0 * 1024.0 * 1024.0 ) )
                        << std::flush;

                    reported = true;
                }

                return true;
            };

        Mila::Distribution::ModelCache cache;
        Mila::Distribution::ModelResolver resolver(
            cache,
            Mila::Distribution::makeHuggingFaceRemoteAccess(
                Mila::Distribution::discoverHuggingFaceToken(), progress ) );

        const auto resolved = resolver.resolveCoordinate( *coordinate );

        if ( reported )
        {
            std::cout << "\n";
        }

        return { resolved.weights_path, resolved.tokenizer_path };
#else
        throw std::runtime_error( std::format(
            "Model '{}': '{}' is not under {}, and this build was compiled without "
            "MILA_ENABLE_MODEL_DOWNLOAD so a HuggingFace coordinate cannot be fetched",
            entry.alias, weights_spec, models_dir.string() ) );
#endif
    }
}
