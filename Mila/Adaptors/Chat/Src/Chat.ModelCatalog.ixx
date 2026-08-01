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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

    /**
     * @brief The variant name a quantization selects.
     *
     * None means unquantized weights, which the manifest names by their dtype rather than by
     * the absence of a policy -- there is no variant called "none".
     */
    export constexpr std::string_view variantName( QuantizationMode mode )
    {
        switch ( mode )
        {
            case QuantizationMode::FP8: return "fp8";
            case QuantizationMode::FP4: return "fp4";
            case QuantizationMode::None: return "bf16";
        }

        return "bf16";
    }

    export inline constexpr std::array<ModelEntry, 8> kModelCatalog = { {
        // A coordinate with no variant: the requested quantization supplies it, so
        // `/model gemma-12b fp8` asks the hub for :fp8 and is told which variants exist
        // rather than loading FP4 bytes under an FP8 policy. Loaded from the store as a
        // pre-quantized 6.33 GB artifact rather than quantizing 23.8 GB of BF16 on the way in.
        { "gemma-12b",     ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true,  true,  QuantizationMode::FP4,  "mila-llm/gemma-4-12b-it",            "gemma/gemma_tokenizer.bin",  512 },
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
        const ModelEntry& entry,
        const std::filesystem::path& models_dir,
        QuantizationMode quantization )
    {
        const std::string weights_spec( entry.weights_file );

        auto coordinate = Mila::Distribution::parseCoordinate( weights_spec );

        if ( coordinate.has_value() )
        {
            // Quantization is the variant, not part of the name. An entry that pins one keeps
            // it; otherwise the request chooses, so asking for a quantization nobody published
            // is answered with the list of variants that exist rather than by loading the
            // wrong bytes under the right-sounding policy.
            if ( coordinate->variant.empty() )
            {
                coordinate->variant = std::string( variantName( quantization ) );
            }

            // The store, with no network call. A pull is a deliberate act; a load is not, so
            // an installed model resolves on nothing but a directory read.
            Mila::Distribution::ModelStore store;

            if ( auto installed = store.locate(
                coordinate->organization, coordinate->repository, coordinate->variant ) )
            {
                return { installed->weights_path, installed->tokenizer_path };
            }
        }

        // REVIEW: the models-directory branch is retired by the catalogue migration -- only a
        // model in the store is loadable. It stays until the catalogue's remaining .bin rows
        // have been exported, packaged and installed, because removing it first would leave
        // every entry but the Gemma one unloadable. It is now *after* the store lookup, so a
        // migrated model loads from the store even when a stale loose file is still on disk.
        const auto local_candidate = models_dir / weights_spec;

        if ( std::filesystem::exists( local_candidate ) )
        {
            return { local_candidate, models_dir / std::string( entry.tokenizer_file ) };
        }

#ifdef MILA_HAS_MODEL_DOWNLOAD
        if ( !coordinate.has_value() )
        {
            throw std::runtime_error( std::format(
                "Model '{}': '{}' is not under {} and is not a HuggingFace coordinate",
                entry.alias, weights_spec, models_dir.string() ) );
        }

        std::cout << std::format( "{} is not installed. Pulling {}\n",
            entry.alias, coordinate->toString() );

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

        const Mila::Distribution::HuggingFaceHub hub(
            Mila::Distribution::discoverHuggingFaceToken(), progress );

        Mila::Distribution::ModelStore pull_store;
        Mila::Distribution::ModelResolver resolver( pull_store, hub );

        const auto pulled = resolver.pull( *coordinate );

        if ( reported )
        {
            std::cout << "\n";
        }

        return { pulled.weights_path, pulled.tokenizer_path };
#else
        throw std::runtime_error( std::format(
            "Model '{}': '{}' is not under {}, and this build was compiled without "
            "MILA_ENABLE_MODEL_DOWNLOAD so a HuggingFace coordinate cannot be fetched",
            entry.alias, weights_spec, models_dir.string() ) );
#endif
    }

    /**
     * @brief Render a byte count at a scale a person reads.
     */
    inline std::string formatBytes( std::uint64_t bytes )
    {
        constexpr double kGigabyte = 1024.0 * 1024.0 * 1024.0;
        constexpr double kMegabyte = 1024.0 * 1024.0;

        if ( bytes >= static_cast<std::uint64_t>( kGigabyte ) )
        {
            return std::format( "{:.2f} GB", static_cast<double>( bytes ) / kGigabyte );
        }

        return std::format( "{:.1f} MB", static_cast<double>( bytes ) / kMegabyte );
    }

    /**
     * @brief What the store holds, one line per model plus a total.
     *
     * Reads only the record tree, so it is instant and works with no network and in a build
     * with no hub at all.
     */
    export std::vector<std::string> describeInstalledModels()
    {
        Mila::Distribution::ModelStore store;

        const auto models = store.list();

        std::vector<std::string> lines;

        if ( models.empty() )
        {
            lines.push_back( std::format(
                "No models installed. Store: {}", store.root().string() ) );

            return lines;
        }

        lines.push_back( std::format( "Installed ({}):", store.root().string() ) );

        for ( const auto& model : models )
        {
            // A record whose blobs went missing is shown rather than hidden: a store that
            // silently omits a broken entry cannot be repaired by the person who owns it.
            lines.push_back( std::format( "  {:<40} {:>10}  {}{}",
                model.record.coordinate(),
                formatBytes( model.bytes_on_disk ),
                model.record.architecture.empty() ? "-" : model.record.architecture,
                model.complete ? "" : "  [INCOMPLETE - blobs missing]" ) );
        }

        const auto usage = store.usage();

        lines.push_back( std::format( "  {} model(s), {} on disk{}",
            usage.model_count,
            formatBytes( usage.blob_bytes ),
            usage.reclaimable_bytes > 0
                ? std::format( ", {} reclaimable with /rm --prune",
                    formatBytes( usage.reclaimable_bytes ) )
                : std::string{} ) );

        return lines;
    }

    /**
     * @brief What an owner publishes, one line per repository.
     *
     * Repository names and tags are authored by whoever owns the repository. They are printed
     * as data and nothing here interprets them.
     */
    export std::vector<std::string> describeHubModels( const std::string& owner )
    {
        std::vector<std::string> lines;

#ifdef MILA_HAS_MODEL_DOWNLOAD
        const Mila::Distribution::HuggingFaceHub hub;

        const auto models = hub.listModels( owner );

        if ( models.empty() )
        {
            lines.push_back( std::format( "{} publishes no Mila models.", owner ) );

            return lines;
        }

        // Which repositories are already here. The listing knows no variants, so this matches
        // on the repository: "some variant of this is installed" is what a reader wants before
        // deciding whether to pull.
        const auto installed = Mila::Distribution::ModelStore{}.list();

        const auto isInstalled = [&installed]( const Mila::Distribution::HubModel& model )
            {
                for ( const auto& stored : installed )
                {
                    if ( stored.record.owner == model.owner
                        && stored.record.repository == model.repository )
                    {
                        return true;
                    }
                }

                return false;
            };

        lines.push_back( std::format( "Published by {}:", owner ) );

        for ( const auto& model : models )
        {
            // Gating is known from the listing, so a repository behind terms says so here
            // instead of surfacing as a 403 partway through a multi-gigabyte transfer.
            lines.push_back( std::format( "  {:<40} {}{}",
                model.coordinate(),
                isInstalled( model ) ? "[installed] " : "",
                model.gated ? "[gated - accept terms on huggingface.co]" : "" ) );
        }

        lines.push_back( "  Install one with /pull <owner>/<repository>:<variant>" );
#else
        lines.push_back(
            "This build was compiled without MILA_ENABLE_MODEL_DOWNLOAD, so no hub can be "
            "listed. Installed models are still available with /models." );
#endif

        return lines;
    }

    /**
     * @brief Pull a coordinate into the store.
     */
    export std::vector<std::string> pullModel( const std::string& spec )
    {
        std::vector<std::string> lines;

#ifdef MILA_HAS_MODEL_DOWNLOAD
        Mila::Distribution::ModelStore store;

        bool reported = false;

        auto progress = [&reported]( std::uint64_t received, std::uint64_t total ) -> bool
            {
                if ( total == 0 )
                {
                    return true;
                }

                const int percent = static_cast<int>( ( received * 100 ) / total );

                std::cout << std::format( "\r  {:>3}%  {} / {}", percent,
                    formatBytes( received ), formatBytes( total ) ) << std::flush;

                reported = true;

                return true;
            };

        const Mila::Distribution::HuggingFaceHub hub(
            Mila::Distribution::discoverHuggingFaceToken(), progress );

        Mila::Distribution::ModelResolver resolver( store, hub );

        const auto pulled = resolver.pull( spec );

        if ( reported )
        {
            std::cout << "\n";
        }

        lines.push_back( std::format( "Installed {} ({}, {}).",
            pulled.record.coordinate(),
            pulled.record.architecture.empty() ? "unknown architecture" : pulled.record.architecture,
            formatBytes( pulled.bytes_on_disk ) ) );
#else
        lines.push_back( std::format(
            "Cannot pull '{}': this build was compiled without MILA_ENABLE_MODEL_DOWNLOAD.",
            spec ) );
#endif

        return lines;
    }

    /**
     * @brief Remove an installed model, reclaiming only what nothing else references.
     *
     * A coordinate with no variant is resolved against what is installed: unambiguous when one
     * variant is present, and reported rather than guessed when several are.
     */
    export std::vector<std::string> removeModel( const std::string& spec )
    {
        std::vector<std::string> lines;

        const auto coordinate = Mila::Distribution::parseCoordinate( spec );

        if ( !coordinate.has_value() )
        {
            lines.push_back( std::format(
                "'{}' is not a coordinate of the form <owner>/<repository>[:<variant>]", spec ) );

            return lines;
        }

        Mila::Distribution::ModelStore store;

        std::string variant = coordinate->variant;

        if ( variant.empty() )
        {
            std::vector<std::string> installed;

            for ( const auto& model : store.list() )
            {
                if ( model.record.owner == coordinate->organization
                    && model.record.repository == coordinate->repository )
                {
                    installed.push_back( model.record.variant );
                }
            }

            if ( installed.empty() )
            {
                lines.push_back( std::format( "{}/{} is not installed.",
                    coordinate->organization, coordinate->repository ) );

                return lines;
            }

            if ( installed.size() > 1 )
            {
                std::string names;

                for ( const auto& name : installed )
                {
                    names += names.empty() ? name : ", " + name;
                }

                lines.push_back( std::format(
                    "{}/{} has several variants installed: {}. Name one.",
                    coordinate->organization, coordinate->repository, names ) );

                return lines;
            }

            variant = installed.front();
        }

        const auto report = store.remove(
            coordinate->organization, coordinate->repository, variant );

        if ( report.records_removed == 0 )
        {
            lines.push_back( std::format( "{}/{}:{} is not installed.",
                coordinate->organization, coordinate->repository, variant ) );

            return lines;
        }

        lines.push_back( std::format( "Removed {}/{}:{} -- {} blob(s), {} reclaimed.",
            coordinate->organization, coordinate->repository, variant,
            report.blobs_removed, formatBytes( report.bytes_reclaimed ) ) );

        // Shared blobs survive by design; saying so pre-empts "why did that free so little".
        if ( report.blobs_removed == 0 )
        {
            lines.push_back(
                "  Its files are shared with another installed variant, so none were deleted." );
        }

        for ( const auto& retained : report.retained )
        {
            lines.push_back( std::format( "  Could not delete (in use?): {}", retained ) );
        }

        return lines;
    }
}
