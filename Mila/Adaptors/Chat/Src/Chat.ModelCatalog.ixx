/**
 * @file Chat.ModelCatalog.ixx
 * @brief Resolving a model name against the store, and the store management commands.
 *
 * There is no catalogue. A model is whatever the store holds under its name, so the set Chat can
 * load is discovered at runtime rather than compiled in -- which is the point, since a compiled
 * table could not name a model the user pulled after the build.
 */

module;
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
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
     * @brief Everything Chat needs to load a model, resolved from its store record.
     *
     * Assembled rather than looked up. Architecture and variant come from the record; the rest
     * are deployment decisions Chat owns, keyed on architecture.
     */
    export struct ResolvedModel
    {
        std::string name;

        std::filesystem::path weights;
        std::filesystem::path tokenizer;

        ModelType family{ ModelType::Gemma };
        ModelPrecision precision{ ModelPrecision::BF16 };
        QuantizationMode quantization{ QuantizationMode::None };

        bool instruct{ false };
        bool streaming_capable{ false };

        std::size_t default_context{ 4096 };
    };

    /// The model a session loads when its config names none.
    export inline constexpr std::string_view kDefaultModelName = "gemma-4-12b-it-fp4";

    /**
     * @brief Architecture string from the record to the family Chat dispatches on.
     */
    export ModelType familyFromArchitecture( std::string_view architecture )
    {
        if ( architecture == "gemma" )
        {
            return ModelType::Gemma;
        }

        if ( architecture == "llama" )
        {
            return ModelType::Llama;
        }

        if ( architecture == "gpt2" )
        {
            return ModelType::Gpt;
        }

        throw std::runtime_error( std::format(
            "Architecture '{}' is not one this build can load.", architecture ) );
    }

    /**
     * @brief Compute precision and weight quantization, from the variant the record declares.
     *
     * The quantized paths run BF16 activations -- the model classes refuse anything else -- so
     * the variant settles both axes.
     */
    export void axesFromVariant(
        std::string_view variant, ModelPrecision& precision, QuantizationMode& quantization )
    {
        if ( variant == "fp4" )
        {
            precision = ModelPrecision::BF16;
            quantization = QuantizationMode::FP4;
        }
        else if ( variant == "fp8" )
        {
            precision = ModelPrecision::BF16;
            quantization = QuantizationMode::FP8;
        }
        else if ( variant == "fp32" )
        {
            precision = ModelPrecision::FP32;
            quantization = QuantizationMode::None;
        }
        else if ( variant == "bf16" )
        {
            precision = ModelPrecision::BF16;
            quantization = QuantizationMode::None;
        }
        else
        {
            throw std::runtime_error( std::format(
                "Variant '{}' is not one this build can load. Expected bf16, fp32, fp8 or fp4.",
                variant ) );
        }
    }

    /**
     * @brief Context length to build for, when the session config does not say.
     *
     * A deployment decision rather than a model property: Gemma 4 12B is conservative because
     * its KV cache is the primary VRAM lever on a 12 GB card, not because the architecture
     * cannot go further.
     */
    export std::size_t defaultContextFor( ModelType family )
    {
        switch ( family )
        {
            case ModelType::Gemma: return 512;
            case ModelType::Gpt:   return 1024;
            default:               return 4096;
        }
    }

    /**
     * @brief Apply a requested load-time quantization to an unquantized artifact.
     *
     * Quantizing on load is a deployment choice, like context length -- not an identity. A
     * pre-quantized artifact is a *different model*: different bytes, its own name. The same
     * BF16 artifact run at FP4 is one model deployed two ways, which is what lets an 8B whose
     * weights are 15 GB run on a card that cannot hold them.
     *
     * The cost is paid at load: the full BF16 file is still read, and quantization happens on
     * the way to the device. A pre-quantized artifact avoids that read entirely, which is why
     * it is worth producing for a model used often.
     *
     * @throws std::runtime_error when the artifact is already quantized, since its bytes cannot
     *         be turned back into something else.
     */
    void applyRequestedQuantization(
        const std::string& name,
        std::string_view variant,
        QuantizationMode requested,
        ResolvedModel& resolved )
    {
        const bool artifact_is_quantized =
            ( resolved.quantization != QuantizationMode::None );

        if ( artifact_is_quantized )
        {
            if ( requested != resolved.quantization )
            {
                throw std::runtime_error( std::format(
                    "'{}' is a pre-quantized {} artifact; its weights cannot be loaded as "
                    "something else. Install the variant you want as its own model.",
                    name, variant ) );
            }

            return;
        }

        if ( requested == QuantizationMode::None )
        {
            return;
        }

        if ( resolved.precision != ModelPrecision::BF16 )
        {
            throw std::runtime_error( std::format(
                "'{}' is an FP32 artifact, and quantized weights require BF16 compute.",
                name ) );
        }

        resolved.quantization = requested;
    }

    /**
     * @brief Resolve a model name against the store.
     *
     * The store is the only source. Nothing here consults a hub, reads a models directory or
     * accepts a path -- a load is not a deliberate act the way a pull is, so it must never
     * become a multi-gigabyte transfer.
     *
     * @param requested_quantization Quantize an unquantized artifact on the way in. Empty loads
     *        the artifact as it is.
     *
     * @throws std::runtime_error if no model of that name is installed, if this build cannot
     *         load what the record describes, or if the requested quantization contradicts it.
     */
    export ResolvedModel resolveModel(
        const std::string& name,
        std::optional<QuantizationMode> requested_quantization = std::nullopt )
    {
        Mila::Distribution::ModelStore store;

        const auto installed = store.locate( name );

        if ( !installed.has_value() )
        {
            // The alternatives are named here rather than pointing at /models, because this
            // also fires at startup, where the session never opens and no command can be run.
            std::string available;

            for ( const auto& model : store.list() )
            {
                available += available.empty() ? "" : ", ";
                available += model.record.name;
            }

            if ( available.empty() )
            {
                throw std::runtime_error( std::format(
                    "No model named '{}' is installed, and neither is anything else.\n"
                    "Store: {}\nPull one with /pull <name>, or install a package you built with "
                    "ExportArtifact --install.", name, store.root().string() ) );
            }

            throw std::runtime_error( std::format(
                "No model named '{}' is installed.\nInstalled: {}\nStore: {}",
                name, available, store.root().string() ) );
        }

        const auto& record = installed->record;

        ResolvedModel resolved;
        resolved.name = record.name;
        resolved.weights = installed->weights_path;
        resolved.tokenizer = installed->tokenizer_path;
        resolved.family = familyFromArchitecture( record.architecture );
        resolved.instruct = record.instruct;

        axesFromVariant( record.variant, resolved.precision, resolved.quantization );

        if ( requested_quantization.has_value() )
        {
            applyRequestedQuantization(
                record.name, record.variant, *requested_quantization, resolved );
        }

        // A harness capability, not a model one: only Gemma's tool calls are protocol tokens a
        // per-token router can see, so the others stay buffered.
        resolved.streaming_capable = ( resolved.family == ModelType::Gemma );

        resolved.default_context = defaultContextFor( resolved.family );

        return resolved;
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
            lines.push_back( std::format( "  {:<34} {:>10}  {:<8} {}{}",
                model.record.name,
                formatBytes( model.bytes_on_disk ),
                model.record.architecture.empty() ? "-" : model.record.architecture,
                model.record.origin(),
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
     * @brief What is available to pull, one line per model.
     *
     * The owner is supplied by the caller and never shown: one publisher makes it a constant
     * rather than a decision, and a name the user cannot act on is noise. Names and tags are
     * authored by whoever owns the repository -- printed as data, interpreted by nothing.
     */
    export std::vector<std::string> describeHubModels( const std::string& owner )
    {
        std::vector<std::string> lines;

#ifdef MILA_HAS_MODEL_DOWNLOAD
        const Mila::Distribution::HuggingFaceHub hub;

        const auto models = hub.listModels( owner );

        if ( models.empty() )
        {
            lines.push_back( "No models are available to pull." );

            return lines;
        }

        // One repository is one model at one precision, so the repository name is the store
        // name and matching on it is exact rather than approximate.
        const auto installed = Mila::Distribution::ModelStore{}.list();

        const auto isInstalled = [&installed]( const Mila::Distribution::HubModel& model )
            {
                for ( const auto& stored : installed )
                {
                    if ( stored.record.name == model.repository )
                    {
                        return true;
                    }
                }

                return false;
            };

        lines.push_back( "Available to pull:" );

        for ( const auto& model : models )
        {
            // Gating is known from the listing, so a repository behind terms says so here
            // instead of surfacing as a 403 partway through a multi-gigabyte transfer.
            lines.push_back( std::format( "  {:<40} {}{}",
                model.repository,
                isInstalled( model ) ? "[installed] " : "",
                model.gated ? "[gated - accept terms on huggingface.co]" : "" ) );
        }

        lines.push_back( "  Install one with /pull <name>" );
#else
        lines.push_back(
            "This build was compiled without MILA_ENABLE_MODEL_DOWNLOAD, so no hub can be "
            "listed. Installed models are still available with /models." );
#endif

        return lines;
    }

    /**
     * @brief Pull a model into the store by name.
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

        const auto pulled = resolver.pull(
            spec, std::string( Mila::Distribution::kDefaultHubOwner ) );

        if ( reported )
        {
            std::cout << "\n";
        }

        lines.push_back( std::format( "Installed {} ({}, {}).",
            pulled.record.name,
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
     */
    export std::vector<std::string> removeModel( const std::string& name )
    {
        std::vector<std::string> lines;

        Mila::Distribution::ModelStore store;

        const auto report = store.remove( name );

        if ( report.records_removed == 0 )
        {
            lines.push_back( std::format( "{} is not installed.", name ) );

            return lines;
        }

        lines.push_back( std::format( "Removed {} -- {} blob(s), {} reclaimed.",
            name, report.blobs_removed, formatBytes( report.bytes_reclaimed ) ) );

        // Shared blobs survive by design; saying so pre-empts "why did that free so little".
        if ( report.blobs_removed == 0 )
        {
            lines.push_back(
                "  Its files are shared with another installed model, so none were deleted." );
        }

        for ( const auto& retained : report.retained )
        {
            lines.push_back( std::format( "  Could not delete (in use?): {}", retained ) );
        }

        return lines;
    }
}
