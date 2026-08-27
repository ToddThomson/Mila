/**
 * @file Chat.ModelCatalog.ixx
 * @brief Resolving a model name against the store, and the store management commands.
 *
 * There is no catalogue. A model is whatever the store holds under its name, so the set Chat can
 * load is discovered at runtime rather than compiled in -- which is the point, since a compiled
 * table could not name a model the user installed after the build.
 */

module;
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

export module Chat.ModelCatalog;

import Chat.Ansi;
import Chat.Config;
import Chat.FamilyTraits;
import Chat.Footprint;
import Mila;
import nlohmann.json;

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

        /// Lineage from the record. Carried because a license that requires attribution requires
        /// it wherever the model is presented, and the session is one of those places.
        std::string base_model;
        std::string license;

        bool instruct{ false };
        bool streaming_capable{ false };

        /// Whether the model has a reasoning channel at all. Read as a capability so the session
        /// never offers a thinking mode the weights cannot produce.
        bool thinking_capable{ false };

        /// True when the quantization is a load-time choice rather than what the artifact already
        /// is. Both land in the same field, but only this one is a fact the model's name omits.
        bool quantization_applied_at_load{ false };
    };

    /**
     * @brief Where the session remembers the model it was last told to use.
     *
     * Beside the STORE, not beside the executable, for three reasons that agree: the store is what
     * actually holds models, it is the directory a container mounts as a volume so the choice
     * survives `--rm`, and writing here never mutates a tracked file in a developer's checkout.
     *
     * This replaces a compiled-in default model. A fresh install has no model, so naming one
     * reported a specific multi-gigabyte artifact as "missing" to a user who had never asked for
     * it -- an empty store is not a failed lookup.
     */
    export inline std::filesystem::path chatStatePath()
    {
        return Mila::Distribution::resolveStoreRoot() / "chat-state.json";
    }

    /// The last model chosen, or nullopt when nothing has been chosen yet. Never throws: an
    /// unreadable or malformed state file means "no choice recorded", which is the first-run state.
    export inline std::optional<std::string> readLastChosenModel()
    {
        try
        {
            const std::filesystem::path path = chatStatePath();

            if ( !std::filesystem::exists( path ) )
                return std::nullopt;

            std::ifstream file( path );
            nlohmann::json state;
            file >> state;

            if ( state.contains( "model" ) && state[ "model" ].is_string() )
            {
                std::string name = state[ "model" ].get<std::string>();

                if ( !name.empty() )
                    return name;
            }
        }
        catch ( const std::exception& )
        {
            // Deliberately swallowed -- see the contract above.
        }

        return std::nullopt;
    }

    /// Record the chosen model. Best-effort: a session that cannot write its state is still a
    /// working session, so a failure here is never allowed to end one.
    export inline void writeLastChosenModel( const std::string& name )
    {
        try
        {
            const std::filesystem::path path = chatStatePath();
            std::filesystem::create_directories( path.parent_path() );

            nlohmann::json state;
            state[ "model" ] = name;

            std::ofstream file( path );
            file << state.dump( 2 ) << "\n";
        }
        catch ( const std::exception& )
        {
        }
    }

    /**
     * @brief The attribution a license requires be displayed wherever the model is presented.
     *
     * Llama 3.1 and 3.2 sections 1.b.i require "Built with Llama" on a related website, user
     * interface, blogpost, about page or product documentation. A user interface is named there,
     * so a session that identifies the model owes the line -- the model card on the hub discharges
     * the same duty for the download, not for the running product.
     *
     * Empty for licenses that impose no display duty. Apache 2.0 requires the notice travel with
     * the artifact; it does not require a UI to render one.
     */
    export std::string_view requiredAttributionFor( std::string_view license )
    {
        return license.starts_with( "llama" )
            ? std::string_view{ "Built with Llama" }
            : std::string_view{};
    }

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

        if ( architecture == "qwen" )
        {
            return ModelType::Qwen;
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
        else if ( variant == "cb2-3" )
        {
            precision = ModelPrecision::BF16;
            quantization = QuantizationMode::Codebook;
        }
        else
        {
            throw std::runtime_error( std::format(
                "Variant '{}' is not one this build can load. Expected bf16, fp32, fp8, fp4 "
                "or cb2-3.", variant ) );
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

        // A codebook is fitted offline against calibration data, so there is no pass over BF16
        // weights that produces one. Every other refusal here is about what the artifact IS;
        // this one is about what no load path can do to it.
        if ( requested == QuantizationMode::Codebook )
        {
            throw std::runtime_error( std::format(
                "'{}' cannot be quantized to cb2-3 on the way in -- the codes are "
                "fitted offline, so an artifact either carries them or does not. Install the "
                "codebook build as its own model.", name ) );
        }

        if ( resolved.precision != ModelPrecision::BF16 )
        {
            throw std::runtime_error( std::format(
                "'{}' is an FP32 artifact, and quantized weights require BF16 compute.",
                name ) );
        }

        resolved.quantization = requested;
        resolved.quantization_applied_at_load = true;
    }

    /**
     * @brief Render a byte count at a scale a person reads.
     *
     * The one byte formatter the app has, so a size on disk and a size in VRAM are always
     * quoted the same way and can be compared without a conversion in the reader's head.
     */
    export inline std::string formatBytes( std::uint64_t bytes )
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
     * @brief Two byte counts against one unit, as "part / whole unit".
     *
     * Formatted as a pair rather than by two formatBytes calls, because the scale is chosen from
     * the whole: a part small enough to land in MB beside a whole in GB reads as a ratio between
     * numbers that are not comparable.
     */
    export inline std::string formatBytesOf( std::uint64_t part, std::uint64_t whole )
    {
        constexpr double kGigabyte = 1024.0 * 1024.0 * 1024.0;
        constexpr double kMegabyte = 1024.0 * 1024.0;

        const bool gigabytes = whole >= static_cast<std::uint64_t>( kGigabyte );
        const double scale = gigabytes ? kGigabyte : kMegabyte;

        return std::format( "{:.2f} / {:.2f} {}",
            static_cast<double>( part ) / scale,
            static_cast<double>( whole ) / scale,
            gigabytes ? "GB" : "MB" );
    }

    /**
     * @brief Render a remaining time the way someone waiting reads it.
     *
     * Coarse above a minute on purpose: a multi-gigabyte transfer over a domestic connection
     * does not hold a rate steady enough to justify quoting seconds, and a figure that jitters
     * every redraw reads as less trustworthy than one that does not.
     */
    export inline std::string formatDuration( double seconds )
    {
        const auto total = static_cast<std::uint64_t>(
            seconds > 0.0 ? seconds + 0.5 : 0.0 );

        const std::uint64_t hours = total / 3600;
        const std::uint64_t minutes = ( total % 3600 ) / 60;

        if ( hours > 0 )
        {
            return std::format( "{}h {:02}m", hours, minutes );
        }

        if ( minutes > 0 )
        {
            return std::format( "{}m", minutes );
        }

        return std::format( "{}s", total );
    }

    /// ASCII case folding, which is all a store name needs: names are repository names, and a
    /// repository name is ASCII by the hub's own rules.
    inline bool equalsIgnoringAsciiCase( std::string_view left, std::string_view right )
    {
        if ( left.size() != right.size() )
        {
            return false;
        }

        for ( std::size_t index = 0; index < left.size(); ++index )
        {
            const auto fold = []( char c )
                {
                    return ( c >= 'A' && c <= 'Z' ) ? static_cast<char>( c - 'A' + 'a' ) : c;
                };

            if ( fold( left[ index ] ) != fold( right[ index ] ) )
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief The store's own spelling of a name, or nullopt when nothing matches.
     *
     * Exact first, then a single case-insensitive match. The fallback is a correctness fix rather
     * than a convenience: `ModelStore::locate` resolves through `recordPath`, a filesystem path, so
     * matching inherits the filesystem's case rules -- insensitive on Windows, sensitive on Linux.
     * Without this the same command works on one platform and fails on the other, and the published
     * names mix conventions (`gemma-4-12b-it-fp4` beside `Llama-3.1-8B-Instruct-fp4`), so it is
     * reached often rather than rarely.
     *
     * Ambiguity resolves to nothing. Two records differing only by case is a store its owner needs
     * to look at, and guessing between them would pick differently on the two platforms -- which is
     * the defect this exists to remove.
     */
    export std::optional<std::string> resolveStoredName( const std::string& name )
    {
        Mila::Distribution::ModelStore store;

        std::optional<std::string> match;

        for ( const auto& model : store.list() )
        {
            if ( model.record.name == name )
            {
                return name;
            }

            if ( !equalsIgnoringAsciiCase( model.record.name, name ) )
            {
                continue;
            }

            if ( match.has_value() )
            {
                return std::nullopt;
            }

            match = model.record.name;
        }

        return match;
    }

    /**
     * @brief Resolve a model name against the store.
     *
     * The store is the only source. Nothing here consults a hub, reads a models directory or
     * accepts a path -- a load is not a deliberate act the way an install is, so it must never
     * become a multi-gigabyte transfer.
     *
     * @param requested_quantization Quantize an unquantized artifact on the way in. Empty loads
     *        the artifact as it is.
     *
     * @throws std::runtime_error if no model of that name is installed, if this build cannot
     *         load what the record describes, or if the requested quantization contradicts it.
     */
    export ResolvedModel resolveModel(
        const std::string& requested_name,
        std::optional<QuantizationMode> requested_quantization = std::nullopt )
    {
        Mila::Distribution::ModelStore store;

        // Folded to the store's own spelling before anything else, so every message below names
        // the model as the store holds it rather than as it was typed.
        const std::string name = resolveStoredName( requested_name ).value_or( requested_name );

        const auto installed = store.locate( name );

        if ( !installed.has_value() )
        {
            // The alternatives are named here rather than pointing at the listing, because this
            // also fires at startup, where the session never opens and no command can be run.
            std::string available;

            for ( const auto& model : store.list() )
            {
                // locate() refuses a record whose blobs are gone, and the listing shows that same
                // record -- so without this the message would name the model in its own list
                // of what is installed while claiming it is not.
                if ( model.record.name == name && !model.complete )
                {
                    throw std::runtime_error( std::format(
                        "'{}' is installed but its files are missing, so it cannot be loaded.\n"
                        "Reinstall it with /model install {}, or drop the record with "
                        "/model remove {}.", name, name, name ) );
                }

                available += available.empty() ? "" : ", ";
                available += model.record.name;
            }

            if ( available.empty() )
            {
                throw std::runtime_error( std::format(
                    "No model named '{}' is installed, and neither is anything else.\n"
                    "Install one with /model install <name>, or from a package you built with "
                    "ExportArtifact --install.", name ) );
            }

            // Deliberately does not ask the publisher whether the name exists there. A load is the
            // offline command, and a typo must not become a network wait.
            throw std::runtime_error( std::format(
                "No model named '{}' is installed.\nInstalled: {}\n"
                "/model list --online shows what can be installed.", name, available ) );
        }

        const auto& record = installed->record;

        // Chat is an INSTRUCT harness -- turns, templates, a system prompt, history, tool calls.
        // A base model uses none of them, and accommodating one meant switching all four off and
        // explaining why the harness was not behaving like itself. That is a second convention
        // inside one surface, so the model is refused here rather than special-cased throughout.
        //
        // Keyed on `instruct`, not on the architecture: a base Llama is as wrong here as GPT-2,
        // and the store already records the answer.
        if ( !record.instruct )
        {
            throw std::runtime_error( std::format(
                "'{}' is a base model, and Chat is an instruct harness -- it renders turns, "
                "applies a chat template and keeps history, none of which a base model reads.\n"
                "Base models generate by continuing a prompt; that is what the completion sample "
                "is for.", record.name ) );
        }

        ResolvedModel resolved;
        resolved.name = record.name;
        resolved.weights = installed->weights_path;
        resolved.tokenizer = installed->tokenizer_path;
        resolved.family = familyFromArchitecture( record.architecture );
        resolved.instruct = record.instruct;
        resolved.base_model = record.base_model;
        resolved.license = record.license;

        axesFromVariant( record.variant, resolved.precision, resolved.quantization );

        if ( requested_quantization.has_value() )
        {
            applyRequestedQuantization(
                record.name, record.variant, *requested_quantization, resolved );
        }

        // Both from the family table, where they sit in one row: streaming is a harness
        // capability and thinking is a model one, they agree today, and reading them from
        // adjacent fields is what makes a future disagreement visible.
        const FamilyTraits traits = familyTraits( resolved.family );

        resolved.streaming_capable = traits.streaming_capable;
        resolved.thinking_capable = traits.thinking_capable;

        return resolved;
    }

    /**
     * @brief The contexts a row is tested at, largest first.
     *
     * The question a row answers is whether the model runs here AT ALL, so what the ladder
     * establishes is that some context fits -- `/context auto` is what answers "how long", against
     * the full 1024-step grid rather than these rungs.
     *
     * Every rung is tried rather than stopping at the first miss, because the footprint curve is
     * NOT monotonic in context: it drops where prefill chunking caps the activation buffers (see
     * Chat.Footprint.ixx), so a rung that does not fit does not prove the ones below it will not.
     * Measured 2026-08-17 a probe is 1-2 ms, which is what makes trying all of them affordable.
     *
     * The bottom rungs exist for a family whose ceiling is low rather than for a card that is:
     * GPT-2 addresses 1024 positions and nothing larger would be tried for it at all.
     */
    inline constexpr Mila::Dnn::dim_t kContextLadder[] = {
        131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024 };


    /**
     * @brief Cell tints for a row that runs and one that does not.
     *
     * The red matches ConsoleRenderer::printError, so a row that will not run and a message saying
     * so are the same colour. Neither tint carries meaning of its own: the verdict is a phrase, so
     * it survives a redirected stream and a reader who cannot separate the two hues -- which is
     * why this replaced a "!" that had to be explained in a footnote.
     */
    // Functions rather than inline string constants: a namespace-scope std::string would need
    // dynamic initialization ordered across a module boundary, which buys nothing here.
    inline std::string overBudgetColour() { return fg( 195, 65, 65 ); }
    inline std::string fitsColour() { return fg( 110, 175, 120 ); }

    /**
     * @brief Whether this model fits the card, in a word.
     *
     * Three states, and the third exists because two of the ways a model can be unusable are not
     * about the GPU at all: a base model and an architecture this build lacks would fail on any
     * card. Answering those with `no` under a heading that says GPU would blame the hardware for a
     * harness limitation, which is the mirror of the defect that retired the word SUPPORTED --
     * that one blamed the library for a hardware limitation. So `no` means measured and too big,
     * and `not supported` means the question does not arise.
     *
     * A caveat names the quantization argument that makes the answer yes, so the cell is also the
     * command. No context length: what fits is a range, `/context auto` reports it against the full
     * grid, and a single rung quoted here read as a promise the table could not keep -- the ladder's
     * top rung claimed 128K for Gemma where the session actually runs 56320.
     */
    struct RowVerdict
    {
        std::string text;

        /// Whether it fits, for the tint. Unknown leaves the cell uncoloured.
        FootprintVerdict verdict{ FootprintVerdict::Unknown };
    };

    /// The largest ladder rung a deployment fits, or nothing and why.
    struct LadderFit
    {
        /// Measured but not displayed. A single rung quoted in the table read as a promise it could
        /// not keep -- the top rung claimed 128K for Gemma where the session runs 56320, because the
        /// ladder tests memory alone where `resolveAutomaticContext` also requires an unconstrained
        /// prefill chunk. Kept because it is what the search actually found, and because a CONTEXT
        /// column would need exactly this and the chunk test alongside it.
        Mila::Dnn::dim_t context_length{ 0 };

        bool fits{ false };
        std::string unavailable_reason;
    };

    /**
     * @brief A rendered listing, split by how it should be shown.
     *
     * Two kinds of line, and the eye wants them apart: the table is the content the command was
     * run to produce, and the notes are commentary on it.
     */
    export struct ModelListing
    {
        std::vector<std::string> table;
        std::vector<std::string> notes;
    };

    /**
     * @brief The largest context this deployment fits in, or nothing and why not.
     *
     * @param fixed_context_length When the session names its own context, the only length worth
     *        asking about -- "does it run at MY context" has one answer, not a largest. Zero when
     *        context is automatic, where the ladder is walked.
     */
    LadderFit largestFittingContext(
        const std::filesystem::path& weights,
        ModelType family,
        ModelPrecision precision,
        QuantizationMode quantization,
        Mila::Dnn::dim_t fixed_context_length,
        std::size_t available_bytes,
        int device_index )
    {
        LadderFit fit;

        if ( fixed_context_length > 0 )
        {
            const FootprintPrediction prediction = predictFootprint(
                weights, family, precision, quantization, fixed_context_length, device_index );

            if ( !prediction.required )
            {
                fit.unavailable_reason = prediction.unavailable_reason;

                return fit;
            }

            fit.context_length = fixed_context_length;
            fit.fits = !isOverBudget( gradeFootprint( prediction.required, available_bytes ) );

            return fit;
        }

        const std::size_t ceiling = familyTraits( family ).max_context;

        for ( const Mila::Dnn::dim_t candidate : kContextLadder )
        {
            // A rung past what the architecture can address is not a shorter context to fall back
            // to -- the load would fail on the position table, not on memory.
            if ( static_cast<std::size_t>( candidate ) > ceiling )
            {
                continue;
            }

            const FootprintPrediction prediction = predictFootprint(
                weights, family, precision, quantization, candidate, device_index );

            if ( !prediction.required )
            {
                fit.unavailable_reason = prediction.unavailable_reason;

                continue;
            }

            // Weights do not shrink with context, so once they alone are over, no shorter rung can
            // help and every remaining probe is wasted. Same short-circuit the auto scan uses, and
            // it turns the worst case -- a card that cannot hold the model at any length -- into
            // the fastest. The reason is cleared because this is a measured answer, not a failure.
            if ( prediction.required->device_parameter_bytes > available_bytes )
            {
                fit.unavailable_reason.clear();

                return fit;
            }

            if ( !isOverBudget( gradeFootprint( prediction.required, available_bytes ) ) )
            {
                fit.context_length = candidate;
                fit.fits = true;
                fit.unavailable_reason.clear();

                return fit;
            }
        }

        return fit;
    }

    /**
     * @brief What the listing needs to cost a row against this machine, right now.
     *
     * Assembled by the session rather than read here, because both figures are facts about the
     * live session: the context it is running at, and the model it already has resident.
     */
    export struct FootprintBudget
    {
        /// The context every row is answered at, when the session names one. Zero when context is
        /// automatic, where each row is answered at the largest context THAT model would get.
        ///
        /// The distinction is the whole defect this replaced: one auto-derived number priced every
        /// row, so Gemma's 56320 -- affordable only because most of its layers are sliding-window
        /// -- was charged to Llama rows that would never be given it, and three of six carried a
        /// warning that could not happen. A context the user NAMED is different: that one really
        /// does apply to every row, because it is what loading any of them would use.
        Mila::Dnn::dim_t fixed_context_length{ 0 };

        /// The memory a load may claim. The caller decides what that means -- the listing asks
        /// what this device can run, which is its capacity, not what happens to be free now.
        std::size_t available_bytes{ 0 };

        /// The card's own name, for the line beneath the table. Empty when it could not be read,
        /// which drops the name and keeps the capacity -- a verdict has to say what it was measured
        /// against even when it cannot say what that thing is called.
        std::string device_name;

        /// Which card the rows are priced against. Carried alongside its capacity rather than
        /// derived here, because a row is measured by BUILDING the graph on that device -- the
        /// two must name the same card or a row would be sized on one and graded against another.
        int device_index{ 0 };

        /// The loaded model's name, marked in the listing. Empty when none is loaded.
        std::string resident_model;
    };

    /**
     * @brief The card a listing's verdicts were measured against, named once beneath the table.
     *
     * The only line either listing prints beneath its table. Shared because a yes and a no are
     * properties of one specific card, and a reader who cannot see which card is being described
     * has to take the column on trust.
     */
    inline std::string describeDevice( const std::string& name, std::size_t total_bytes )
    {
        return name.empty()
            ? std::format( "GPU Fit: based on your {} VRAM", formatBytes( total_bytes ) )
            : std::format( "GPU Fit: based on your {} with {} VRAM",
                name, formatBytes( total_bytes ) );
    }

    /**
     * @brief Whether this row runs here, and what it takes.
     *
     * Asked through the same predictor and grader the load pre-flight uses, so a row that says it
     * runs and a load that then warns cannot disagree about anything but the machine's state.
     */
    RowVerdict verdictFor(
        const Mila::Distribution::StoredModel& model,
        Mila::Dnn::dim_t fixed_context_length,
        std::size_t available_bytes,
        int device_index )
    {
        RowVerdict row;

        // The INCOMPLETE marker on the row already says what happened, so this does not repeat it.
        if ( !model.complete )
        {
            row.text = "-";

            return row;
        }

        // Chat refuses a base model at load, so probing one would price a load that cannot happen.
        // Said in the column the reader is already looking at rather than left for them to discover
        // by typing /model and getting an essay.
        if ( !model.record.instruct )
        {
            row.text = "not supported";

            return row;
        }

        ModelPrecision precision{ ModelPrecision::BF16 };
        QuantizationMode artifact{ QuantizationMode::None };
        ModelType family{ ModelType::Gemma };

        try
        {
            axesFromVariant( model.record.variant, precision, artifact );
            family = familyFromArchitecture( model.record.architecture );
        }
        catch ( const std::exception& error )
        {
            row.text = "not supported";

            return row;
        }

        // As the artifact is. A pre-quantized model offers only this, so `artifact` rather than
        // None -- for those two are different things, and it is the former /model with no argument
        // loads.
        const LadderFit native = largestFittingContext( model.weights_path, family, precision,
            artifact, fixed_context_length, available_bytes, device_index );

        if ( native.fits )
        {
            row.verdict = FootprintVerdict::Fits;
            row.text = "yes";

            return row;
        }

        std::string reason = native.unavailable_reason;

        // Quantizing on load needs BF16 compute and bytes that are not already quantized, so a
        // pre-quantized or FP32 artifact has nothing further to offer.
        if ( precision == ModelPrecision::BF16 && artifact == QuantizationMode::None )
        {
            // BOTH are tried rather than stopping at the first that works, because they are not
            // interchangeable: fp8 costs less accuracy and fp4 less memory, and which to reach for
            // is the reader's call. A cell naming only the first would make it ours.
            std::vector<std::string_view> forms;

            for ( const QuantizationMode mode : { QuantizationMode::FP8, QuantizationMode::FP4 } )
            {
                const LadderFit fit = largestFittingContext( model.weights_path, family, precision,
                    mode, fixed_context_length, available_bytes, device_index );

                if ( fit.fits )
                {
                    forms.push_back( quantizationName( mode ) );
                }
                else if ( reason.empty() )
                {
                    reason = fit.unavailable_reason;
                }
            }

            if ( !forms.empty() )
            {
                row.verdict = FootprintVerdict::Fits;

                row.text = forms.size() == 1
                    ? std::format( "yes, at {}", forms.front() )
                    : std::format( "yes, at {} or {}", forms.front(), forms.back() );

                return row;
            }
        }

        // Nothing fit, and the two reasons for that are not the same answer: measured-and-too-big is
        // a fact about the card, where a prediction that could not be made is one about this build.
        // Conflating them is what the old single dash did.
        if ( !reason.empty() )
        {
            row.text = "unknown";

            return row;
        }

        // Plain, with no figure: the card's capacity is named once beneath the table, so repeating
        // it on every row that misses would be the same number several times.
        row.verdict = FootprintVerdict::DoesNotFit;
        row.text = "no";

        return row;
    }

    /**
     * @brief What the store holds, one line per model plus a total.
     *
     * Reads only the record tree by default, so it is instant and works with no network, with
     * no device, and in a build with no hub at all. That default matters beyond speed: this
     * also renders --help, which runs before the runtime is initialized.
     *
     * @param budget Engaged adds the fit column, at the price of an artifact-header read and up
     *        to a ladder of constructed graphs per row. No device memory is committed by it. The
     *        session assembles it, because both figures in it are facts about the live session.
     */
    export ModelListing describeInstalledModels(
        std::optional<FootprintBudget> budget = std::nullopt )
    {
        Mila::Distribution::ModelStore store;

        const auto models = store.list();

        ModelListing listing;

        if ( models.empty() )
        {
            // Points at the listing rather than at /install, because this is the first-run state
            // and a user with an empty store has no name to pass to /install yet.
            listing.table.push_back(
                "No models installed. /model list --online lists what can be installed." );

            return listing;
        }

        // Two columns, and the second is the whole reason the command exists. Architecture and
        // license are both gone: the architecture is in the name of every published model and in
        // /model's `Base model:` line, which is better information than a family bucket, and by
        // the time a model is installed its terms have already been taken on. The license is shown
        // where it is a DECISION (--online, before the download) and where it is IDENTITY (/model,
        // beside the attribution a license can require) -- neither of which is this table.
        listing.table.push_back( budget.has_value()
            ? std::format( "  {:<30}{}", "MODEL", "GPU FIT" )
            : std::string( "  MODEL" ) );

        // One guard for every probe the table makes. Constructing a graph logs, and Gemma complains
        // that it cannot prefill efficiently at the top rungs -- a warning about a context nobody
        // asked to run at, which arrived above the table before this was here. A prediction is not
        // a deployment; warnings from an actual LOAD are untouched.
        const ScopedLogSuppression quiet;

        for ( const auto& model : models )
        {
            std::string runs;

            if ( budget.has_value() )
            {
                const RowVerdict row = verdictFor(
                    model, budget->fixed_context_length, budget->available_bytes,
                    budget->device_index );

                // Tinted after formatting and last in the row, so no width calculation has to
                // account for the escape bytes -- which is what the memory columns had to. Colour
                // carries nothing the words do not: the verdict reads the same in mono.
                runs = row.verdict == FootprintVerdict::Fits
                        ? fitsColour() + row.text + reset()
                    : isOverBudget( row.verdict )
                        ? overBudgetColour() + row.text + reset()
                    : row.text;
            }

            const std::string marker =
                ( budget.has_value() && budget->resident_model == model.record.name )
                    ? "* " : "  ";

            // A record whose blobs went missing is shown rather than hidden: a store that
            // silently omits a broken entry cannot be repaired by the person who owns it.
            listing.table.push_back( budget.has_value()
                ? std::format( "{}{:<30}{}{}", marker, model.record.name, runs,
                    model.complete ? "" : "  [INCOMPLETE - blobs missing]" )
                : std::format( "{}{}{}", marker, model.record.name,
                    model.complete ? "" : "  [INCOMPLETE - blobs missing]" ) );
        }

        const auto usage = store.usage();

        // Disk stays in the total and off the rows. The aggregate is refcounted and answers the
        // only question a byte count here serves -- what is this store costing me -- while a
        // per-row size is not what removing that row would return, because blobs are shared.
        listing.table.push_back( std::format( "  {} model(s), {} on disk{}",
            usage.model_count,
            formatBytes( usage.blob_bytes ),
            usage.reclaimable_bytes > 0
                ? std::format( ", {} reclaimable with /rm --prune",
                    formatBytes( usage.reclaimable_bytes ) )
                : std::string{} ) );

        // The card the column is about, and the only line beneath the table. Everything else that
        // stood here -- the context basis, the quantization legend, a per-row reason for an
        // unmeasured verdict -- was the table explaining itself, which is what /help is for.
        if ( budget.has_value() )
        {
            listing.notes.push_back(
                describeDevice( budget->device_name, budget->available_bytes ) );
        }

        return listing;
    }

    /**
     * @brief What is available to install, one line per model.
     *
     * The owner is supplied by the caller and never shown: one publisher makes it a constant
     * rather than a decision, and a name the user cannot act on is noise. Names and tags are
     * authored by whoever owns the repository -- printed as data, interpreted by nothing.
     *
     * **This is the only listing a new user has**, since an empty store makes /model list an empty
     * table -- so the support question has to be answerable here, before a multi-gigabyte transfer
     * rather than after it.
     *
     * @param available_bytes Device capacity, for the one memory verdict that can be reached
     *        honestly from a manifest. Zero screens nothing.
     * @param device_name The card's own name, for the line beneath the table.
     */
    export std::vector<std::string> describeHubModels(
        const std::string& owner,
        std::size_t available_bytes = 0,
        const std::string& device_name = {} )
    {
        std::vector<std::string> lines;

        // A build without a hub says so, rather than reporting an empty listing -- that would
        // claim this publisher has nothing, when the truth is that nothing can be asked.
        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            lines.push_back(
                "This build has no HTTP transport (MILA_ENABLE_LIBCURL=OFF), so no publisher "
                "can be listed. Models already installed are shown by /model list." );

            return lines;
        }

        const auto hub = Mila::Distribution::makeDefaultModelHub();

        const auto models = hub->listModels( owner );

        if ( models.empty() )
        {
            lines.push_back( "No models are available to install." );

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

        struct OnlineDetail
        {
            std::uint64_t bytes{ 0 };

            // From the manifest rather than the listing's `license:` tag, which is the model
            // card's claim. The manifest is what governs once installed, so what is shown before
            // the download is what the store will hold after it.
            std::string license;

            /// "yes", "no", "not supported", or empty when the manifest could not be read.
            std::string_view fit;
        };

        /**
         * What can be settled from a manifest alone, and nothing beyond it.
         *
         * The same three states `/model list` uses, so one word means one thing across both listings:
         * `not supported` for a base model or an architecture this build lacks, which no card would
         * change; `no` and `yes` for the card.
         *
         * Architecture, variant and instruct are decided here with certainty, which is most of why
         * a first install fails. Memory is not: the manifest has no geometry, so there is no
         * footprint to predict, and the download size is a LOWER bound on the device weights rather
         * than an estimate of them -- FP4 unpacks, and the unquantized tables ride along. That
         * bound still buys the one honest memory verdict: a download larger than the whole card
         * cannot possibly fit, so that `no` is certain where the `yes` beside it is provisional.
         */
        const auto fitOf = [available_bytes](
            const Mila::Distribution::ModelManifest& manifest,
            std::uint64_t bytes ) -> std::string_view
            {
                if ( !manifest.instruct )
                {
                    return "not supported";
                }

                try
                {
                    ModelPrecision precision{ ModelPrecision::BF16 };
                    QuantizationMode quantization{ QuantizationMode::None };

                    axesFromVariant( manifest.variant, precision, quantization );
                    familyFromArchitecture( manifest.architecture );
                }
                catch ( const std::exception& )
                {
                    return "not supported";
                }

                return ( available_bytes > 0 && bytes > available_bytes ) ? "no" : "yes";
            };

        // What the transfer costs and what the model is -- both the manifest's answer, and the
        // repository listing knows neither. The size in particular is not the repo's size: a repo
        // also holds README, LICENSE and .gitattributes, and Mila fetches none of them. One small
        // GET per row, on a command the user asked for by name.
        const auto describe =
            [&hub, &owner, &fitOf]( const Mila::Distribution::HubModel& model ) -> OnlineDetail
            {
                if ( !model.hasManifest() )
                {
                    return {};
                }

                try
                {
                    const Mila::Distribution::ModelCoordinate coordinate{
                        owner, model.repository };

                    const auto manifest = Mila::Distribution::parseModelManifest(
                        hub->fetchManifest( coordinate ), model.repository );

                    OnlineDetail detail;
                    detail.license = manifest.license;

                    for ( const auto& file : manifest.files )
                    {
                        detail.bytes += file.bytes;
                    }

                    detail.fit = fitOf( manifest, detail.bytes );

                    return detail;
                }
                catch ( const std::exception& )
                {
                    // A repository whose manifest cannot be read is still worth listing by name.
                    // Losing one row's detail must not cost the whole listing.
                    return {};
                }
            };

        // Every field carries its own trailing pad, which is deliberate: the gated sentence used to
        // live in a trailing STATUS column with no separator of its own, so a verdict that exactly
        // filled the field before it ran straight in ("no, base modelinstalled"). Gating moved to a
        // note, and GPU FIT is sized for its longest value plus two, so nothing can overrun.
        //
        // MODEL is 32 rather than 40 because the longest published name is 25 and the line would
        // otherwise wrap on an 80-column terminal.
        //
        // LICENSE earns a place here and not in /model list, because here it is a decision: this is
        // the listing read while choosing whether to pull a model, which is the last moment before
        // its terms are taken on. ARCHITECTURE is gone for the reason it went from /model list -- it is
        // in the name of every published model, and it decides nothing an uninstalled row can act
        // on. GPU FIT took that space, which is what an uninstalled row CAN act on.
        lines.push_back( std::format( "  {:<32} {:>10}  {:<12}{:<15}{}",
            "MODEL", "DOWNLOAD", "LICENSE", "GPU FIT", "INSTALLED" ) );

        std::vector<std::string> gated;

        for ( const auto& model : models )
        {
            const OnlineDetail detail = describe( model );

            if ( model.gated )
            {
                gated.push_back( model.repository );
            }

            lines.push_back( std::format( "  {:<32} {:>10}  {:<12}{:<15}{}",
                model.repository,
                detail.bytes > 0 ? formatBytes( detail.bytes ) : std::string( "--" ),
                detail.license.empty() ? "--" : detail.license,
                detail.fit.empty() ? "--" : detail.fit,
                isInstalled( model ) ? "yes" : "no" ) );
        }

        // The card the column is about. A yes here is the provisional one -- the manifest has no
        // geometry, so what is checked is that the download alone does not exceed the card -- and
        // naming the card is what lets a reader hold the DOWNLOAD figures against it themselves.
        if ( available_bytes > 0 )
        {
            lines.push_back( "" );
            lines.push_back( "  " + describeDevice( device_name, available_bytes ) );
        }

        // Gating is known from the listing, so a repository behind terms says so here rather than
        // surfacing as a 403 partway through a multi-gigabyte transfer. Named per repository
        // because a blanket sentence would not say which pull is going to fail.
        for ( const auto& repository : gated )
        {
            lines.push_back( std::format(
                "  {} is gated -- accept its terms on huggingface.co before installing.",
                repository ) );
        }

        return lines;
    }

    /**
     * @brief What is known about one model, whether it is installed or only published.
     *
     * The store is asked first, so a model already here answers offline and instantly; only a name
     * the store does not hold costs a manifest fetch. That order is what lets one command serve
     * both populations -- a user asks what a model IS without first having to know whether they
     * have it.
     *
     * The facts only. The model card is published with the model and is not fetched here.
     */
    export std::vector<std::string> describeModel( const std::string& name )
    {
        std::vector<std::string> lines;

        const auto field = []( std::string_view label, std::string_view value )
            {
                return std::format( "  {:<16}{}", label, value );
            };

        Mila::Distribution::ModelStore store;

        if ( const auto canonical = resolveStoredName( name ) )
        {
            for ( const auto& model : store.list() )
            {
                if ( model.record.name != *canonical )
                {
                    continue;
                }

                const auto& record = model.record;

                std::uint64_t bytes = 0;

                for ( const auto& file : record.files )
                {
                    bytes += file.bytes;
                }

                lines.push_back( field( "Model:", record.name ) );

                if ( !record.base_model.empty() )
                {
                    lines.push_back( field( "Base model:", record.base_model ) );
                }

                if ( !record.license.empty() )
                {
                    lines.push_back( field( "License:", record.license ) );
                }

                // The duty a license can impose, discharged wherever the model is presented --
                // which this is, as much as the session status line is.
                if ( const auto attribution = requiredAttributionFor( record.license );
                    !attribution.empty() )
                {
                    lines.push_back( field( "Attribution:", attribution ) );
                }

                lines.push_back( field( "Architecture:",
                    record.architecture.empty() ? "-" : record.architecture ) );
                lines.push_back( field( "Variant:",
                    record.variant.empty() ? "-" : record.variant ) );
                lines.push_back( field( "Instruct:", record.instruct ? "yes" : "no" ) );
                lines.push_back( field( "Installed:", model.complete
                    ? formatBytes( bytes ) : "yes, but its files are missing" ) );
                lines.push_back( field( "Origin:", record.origin() ) );

                return lines;
            }
        }

        // Not here, so ask the publisher. A name that is neither installed nor published costs one
        // failed fetch, which is the right price for a typo on a command typed by hand.
        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            lines.push_back( std::format(
                "'{}' is not installed, and this build has no HTTP transport to ask a "
                "publisher.", name ) );

            return lines;
        }

        const auto hub = Mila::Distribution::makeDefaultModelHub();

        Mila::Distribution::ModelManifest manifest;

        try
        {
            const Mila::Distribution::ModelCoordinate coordinate{
                std::string( Mila::Distribution::kDefaultHubOwner ), name };

            manifest = Mila::Distribution::parseModelManifest(
                hub->fetchManifest( coordinate ), name );
        }
        catch ( const std::exception& )
        {
            lines.push_back( std::format(
                "No model named '{}' is installed or published.", name ) );

            return lines;
        }

        std::uint64_t bytes = 0;

        for ( const auto& file : manifest.files )
        {
            bytes += file.bytes;
        }

        lines.push_back( field( "Model:", manifest.name.empty() ? name : manifest.name ) );

        if ( !manifest.base_model.empty() )
        {
            lines.push_back( field( "Base model:", manifest.base_model ) );
        }

        if ( !manifest.license.empty() )
        {
            lines.push_back( field( "License:", manifest.license ) );
        }

        if ( const auto attribution = requiredAttributionFor( manifest.license );
            !attribution.empty() )
        {
            lines.push_back( field( "Attribution:", attribution ) );
        }

        lines.push_back( field( "Architecture:",
            manifest.architecture.empty() ? "-" : manifest.architecture ) );
        lines.push_back( field( "Variant:",
            manifest.variant.empty() ? "-" : manifest.variant ) );
        lines.push_back( field( "Instruct:", manifest.instruct ? "yes" : "no" ) );
        lines.push_back( field( "Installed:", "no" ) );
        lines.push_back( field( "Download:", formatBytes( bytes ) ) );

        return lines;
    }

    /**
     * @brief Install a published model into the store by name.
     */
    export std::vector<std::string> installModel( const std::string& spec )
    {
        std::vector<std::string> lines;

        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            lines.push_back( std::format(
                "Cannot install '{}': this build has no HTTP transport "
                "(MILA_ENABLE_LIBCURL=OFF).", spec ) );

            return lines;
        }

        Mila::Distribution::ModelStore store;

        bool reported = false;

        // One callback serves every file in a pull, so this tracks which transfer is running as
        // well as how far along it is: a changed total, or a count that goes backwards, is the
        // next file starting rather than progress on this one.
        struct TransferProgress
        {
            std::uint64_t total{ 0 };
            std::uint64_t first_received{ 0 };
            std::chrono::steady_clock::time_point started{};
            int last_percent{ -1 };
            bool active{ false };
        };

        TransferProgress transfer;

        auto progress = [&reported, &transfer](
            std::uint64_t received, std::uint64_t total ) -> bool
            {
                if ( total == 0 )
                {
                    return true;
                }

                const auto now = std::chrono::steady_clock::now();

                if ( !transfer.active || total != transfer.total
                    || received < transfer.first_received )
                {
                    transfer = { total, received, now, -1, true };
                }

                const int percent = static_cast<int>( ( received * 100 ) / total );

                // One redraw per whole percent. Unthrottled this fired on every chunk, which at
                // 6.33 GB is tens of thousands of writes to a line that has 101 distinct states.
                if ( percent == transfer.last_percent && received < total )
                {
                    return true;
                }

                transfer.last_percent = percent;

                // Rate over what this run actually moved. A resumed transfer opens holding a
                // prefix it spent no time on, and counting those bytes makes the estimate a lie.
                const double elapsed =
                    std::chrono::duration<double>( now - transfer.started ).count();
                const std::uint64_t moved = received - transfer.first_received;

                std::string remaining;

                if ( elapsed > 2.0 && moved > 0 && received < total )
                {
                    const double rate = static_cast<double>( moved ) / elapsed;

                    remaining = std::format( "  {} left", formatDuration(
                        static_cast<double>( total - received ) / rate ) );
                }

                const std::string line = std::format( "  {:>3}%  {}{}",
                    percent, formatBytesOf( received, total ), remaining );

                // Erase to end of line rather than padding out to a fixed width. Padding writes
                // past the content, so on a terminal narrower than the pad the line wraps -- and
                // once it has wrapped, \r returns to the start of the last visual row, leaving
                // the earlier rows on screen for the next redraw to sit beside. That is what a
                // resize turned into fragments. Same idiom the renderer uses for the spinner.
                std::cout << "\r" << line << "\x1b[K" << std::flush;

                reported = true;

                return true;
            };

        const auto hub = Mila::Distribution::makeDefaultModelHub( progress );

        Mila::Distribution::ModelResolver resolver( store, *hub );

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

        // Said at the one moment the user is certain to be reading: a model can carry terms,
        // and until now nothing in the product mentioned it. The identifier only -- the text is
        // published with the model, which is the copy the license actually governs.
        if ( !pulled.record.license.empty() )
        {
            lines.push_back( std::format(
                "License: {}. The terms are published with the model.",
                pulled.record.license ) );
        }

        return lines;
    }

    /**
     * @brief Remove an installed model, reclaiming only what nothing else references.
     */
    export std::vector<std::string> removeModel( const std::string& requested_name )
    {
        std::vector<std::string> lines;

        Mila::Distribution::ModelStore store;

        // Folded to the store's own spelling, so a name that lists and a name that removes are the
        // same name on both platforms. See resolveStoredName.
        const std::string name = resolveStoredName( requested_name ).value_or( requested_name );

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
