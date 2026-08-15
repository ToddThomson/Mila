/**
 * @file Chat.ModelCatalog.ixx
 * @brief Resolving a model name against the store, and the store management commands.
 *
 * There is no catalogue. A model is whatever the store holds under its name, so the set Chat can
 * load is discovered at runtime rather than compiled in -- which is the point, since a compiled
 * table could not name a model the user installed after the build.
 */

module;
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
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
                // locate() refuses a record whose blobs are gone, and /models lists that same
                // record -- so without this the message would name the model in its own list
                // of what is installed while claiming it is not.
                if ( model.record.name == name && !model.complete )
                {
                    throw std::runtime_error( std::format(
                        "'{}' is installed but its files are missing, so it cannot be loaded.\n"
                        "Reinstall it with /install {}, or drop the record with /rm {}.",
                        name, name, name ) );
                }

                available += available.empty() ? "" : ", ";
                available += model.record.name;
            }

            if ( available.empty() )
            {
                throw std::runtime_error( std::format(
                    "No model named '{}' is installed, and neither is anything else.\n"
                    "Install one with /install <name>, or from a package you built with "
                    "ExportArtifact --install.", name ) );
            }

            throw std::runtime_error( std::format(
                "No model named '{}' is installed.\nInstalled: {}", name, available ) );
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
     * @brief What a row costs in device memory, and whether the card has room for it.
     *
     * Assembled from the record alone, so it asks the same question the pre-flight will ask
     * when this model is actually loaded -- through the same predictor and the same grader.
     */
    /**
     * @brief One way /model can be asked to load an artifact, and what it would cost.
     *
     * The columns are the command forms, which is the point of showing them: reading across a
     * row is reading the arguments that would work, with the price of each.
     */
    struct RowDeployment
    {
        std::optional<Mila::Dnn::MemoryStats> required;
        FootprintVerdict verdict{ FootprintVerdict::Unknown };

        /// False when the artifact's own bytes refuse this form, which is not the same as a
        /// prediction that could not be made.
        bool applicable{ false };
    };

    /// As installed, at FP8, at FP4 -- matching `/model <name>` with no argument, `fp8`, `fp4`.
    inline constexpr std::size_t kDeploymentCount = 3;

    /**
     * @brief Footnote marks tying a memory column to the command that produces it.
     *
     * The native column carries none: it is the command with no argument, so there is nothing
     * about it to footnote.
     *
     * Written as UTF-8 bytes rather than as characters, matching ConsoleRenderer: the build does
     * not set an execution charset, so a literal glyph would encode by the compiler's codepage.
     * Each is one display column and two bytes, which every width calculation below must treat
     * separately -- std::format pads to byte length.
     */
    inline constexpr std::string_view kColumnMarks[ kDeploymentCount ] = {
        "",
        "\xc2\xb9",  // superscript 1
        "\xc2\xb2"   // superscript 2
    };

    /**
     * @brief Cell tints for the two verdicts worth marking.
     *
     * The red matches ConsoleRenderer::printError, so a cell that will not fit and a message
     * that says so are the same colour. Neither is the sole carrier of its meaning -- the "!"
     * survives a redirected stream and a reader who cannot separate the two hues.
     */
    // Functions rather than inline string constants: a namespace-scope std::string would need
    // dynamic initialization ordered across a module boundary, which buys nothing here.
    inline std::string overBudgetColour() { return fg( 195, 65, 65 ); }
    inline std::string fitsColour() { return fg( 110, 175, 120 ); }

    /// What each marked column is asking /model to do, in the order the columns appear.
    inline constexpr std::string_view kColumnLegend[ kDeploymentCount ] = {
        "",
        "/model <name> fp8   Load model with fp8 dynamic quantization",
        "/model <name> fp4   Load model with fp4 dynamic quantization"
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

    struct RowFootprint
    {
        std::array<RowDeployment, kDeploymentCount> deployments;
    };

    /**
     * @brief Whether a quantization column has anything to say about this artifact.
     *
     * Narrower than applyRequestedQuantization, deliberately. That would accept `fp4` against an
     * already-FP4 artifact, since the request agrees with the bytes -- but the column would then
     * repeat the native figure under a heading implying a choice, and nobody would type an
     * argument asking for what they already have. A pre-quantized artifact offers one deployment
     * and the native column is it.
     *
     * Quantized weights also need BF16 compute, so an FP32 artifact offers nothing either.
     */
    bool deploymentIsApplicable(
        ModelPrecision precision, QuantizationMode artifact, QuantizationMode requested )
    {
        if ( requested == QuantizationMode::None )
        {
            return true;
        }

        return precision == ModelPrecision::BF16 && artifact == QuantizationMode::None;
    }

    /**
     * @brief What the listing needs to cost a row against this machine, right now.
     *
     * Assembled by the session rather than read here, because both figures are facts about the
     * live session: the context it is running at, and the model it already has resident.
     */
    export struct FootprintBudget
    {
        Mila::Dnn::dim_t context_length{ 0 };

        /// The memory a load may claim. The caller decides what that means -- the listing asks
        /// what this device can run, which is its capacity, not what happens to be free now.
        std::size_t available_bytes{ 0 };

        /// The loaded model's name, marked in the listing. Empty when none is loaded.
        std::string resident_model;
    };

    RowFootprint footprintOf(
        const Mila::Distribution::StoredModel& model,
        Mila::Dnn::dim_t context_length,
        std::size_t available_bytes )
    {
        RowFootprint footprint;

        // A record whose blobs are gone has nothing to read a header from, and one this build
        // cannot describe has no graph to construct -- both are an absent number, not an error.
        if ( !model.complete )
        {
            return footprint;
        }

        try
        {
            ModelPrecision precision{ ModelPrecision::BF16 };
            QuantizationMode artifact{ QuantizationMode::None };

            axesFromVariant( model.record.variant, precision, artifact );

            const ModelType family = familyFromArchitecture( model.record.architecture );

            // None means "as the artifact is", so the first column carries the artifact's own
            // quantization rather than no quantization -- for a pre-quantized model those are
            // different things, and it is the former that /model with no argument loads.
            const QuantizationMode requested[ kDeploymentCount ] = {
                QuantizationMode::None, QuantizationMode::FP8, QuantizationMode::FP4 };

            for ( std::size_t column = 0; column < kDeploymentCount; ++column )
            {
                RowDeployment& deployment = footprint.deployments[ column ];

                deployment.applicable =
                    deploymentIsApplicable( precision, artifact, requested[ column ] );

                if ( !deployment.applicable )
                {
                    continue;
                }

                deployment.required = predictFootprint( model.weights_path, family, precision,
                    requested[ column ] == QuantizationMode::None ? artifact : requested[ column ],
                    context_length );

                deployment.verdict = gradeFootprint( deployment.required, available_bytes );
            }
        }
        catch ( const std::exception& )
        {
            return footprint;
        }

        return footprint;
    }

    /**
     * @brief What the store holds, one line per model plus a total.
     *
     * Reads only the record tree by default, so it is instant and works with no network, with
     * no device, and in a build with no hub at all. That default matters beyond speed: this
     * also renders --help, which runs before the runtime is initialized.
     *
     * @param budget Engaged adds the device memory column, at the price of an artifact-header
     *        read and a constructed graph per row. No device memory is committed by it. The
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
            listing.table.push_back( "No models installed. Install one with /install <name>." );

            return listing;
        }

        // Right-aligned on display width rather than by std::format, because a footnote mark is
        // two bytes and one column -- padding the combined string would leave every header a
        // character short of the row beneath it.
        const auto headerCell = []( std::string_view name, std::string_view mark )
            {
                constexpr std::size_t kCellWidth = 9;

                // An unmarked column shows only its name; a mark adds exactly one column.
                const std::size_t shown = name.size() + ( mark.empty() ? 0 : 1 );

                return std::string( kCellWidth > shown ? kCellWidth - shown : 0, ' ' )
                    + std::string( name ) + std::string( mark ) + "  ";
            };

        listing.table.push_back( budget.has_value()
            ? std::format( "  {:<30}{}{}{}{:<13}{}",
                "MODEL",
                headerCell( "NATIVE", kColumnMarks[ 0 ] ),
                headerCell( "FP8", kColumnMarks[ 1 ] ),
                headerCell( "FP4", kColumnMarks[ 2 ] ),
                "ARCHITECTURE",
                "LICENSE" )
            : std::format( "  {:<30} {:<13}{}", "MODEL", "ARCHITECTURE", "LICENSE" ) );

        bool any_over_budget = false;

        for ( const auto& model : models )
        {
            const RowFootprint footprint = budget.has_value()
                ? footprintOf( model, budget->context_length, budget->available_bytes )
                : RowFootprint{};

            std::string memory_fields;

            for ( const RowDeployment& deployment : footprint.deployments )
            {
                if ( !budget.has_value() )
                {
                    break;
                }

                any_over_budget = any_over_budget || isOverBudget( deployment.verdict );

                // Three states, and the difference matters: a form the artifact refuses is not
                // the same as one this build cannot predict. One dash for "not this artifact",
                // two for "no answer" -- a load that would work but goes unmeasured.
                const std::string cost =
                    !deployment.applicable ? std::string( "-" )
                    : deployment.required.has_value()
                        ? formatBytes( practicalDeviceBytes( *deployment.required ) )
                        : std::string( "--" );

                // Padded first, tinted second. std::format counts an escape sequence's bytes as
                // width, so colouring before padding silently shortens the column.
                const std::string cell = std::format( "{:>9}{} ",
                    cost, isOverBudget( deployment.verdict ) ? "!" : " " );

                // Colour carries no information the text does not: the "!" says the same thing
                // for a reader who has no colour, and a cell that says nothing stays uncoloured
                // rather than being tinted for decoration.
                memory_fields +=
                    isOverBudget( deployment.verdict ) ? overBudgetColour() + cell + reset()
                    : deployment.verdict == FootprintVerdict::Fits ? fitsColour() + cell + reset()
                    : cell;
            }

            // A record whose blobs went missing is shown rather than hidden: a store that
            // silently omits a broken entry cannot be repaired by the person who owns it.
            //
            // The license is here because a model can carry terms and nothing else in the
            // product ever says so. It is the identifier only -- the text travels with the
            // model on its hub page, which is the copy the license itself governs.
            listing.table.push_back( std::format( "{}{:<30}{}{:<13}{:<12}{}",
                ( budget.has_value() && budget->resident_model == model.record.name )
                    ? "* " : "  ",
                model.record.name,
                memory_fields.empty() ? std::string( " " ) : memory_fields,
                model.record.architecture.empty() ? "-" : model.record.architecture,
                model.record.license.empty() ? "-" : model.record.license,
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

        // The context the column was costed at: a memory size that does not say at what context
        // is a number nobody can act on, and with no heading above the table this is the only
        // place left to say it. The card's capacity is not repeated here -- the marker says
        // which cells exceed it and the note below carries the figure.
        if ( budget.has_value() )
        {
            listing.notes.push_back( std::format(
                "Memory required with {} context size", budget->context_length ) );
        }

        // One explanation for every marked cell. The marker says which cells; repeating the
        // same sentence per cell would say nothing more.
        if ( any_over_budget )
        {
            listing.notes.push_back( std::format( "! {}",
                doesNotFitExplanation( formatBytes( budget->available_bytes ) ) ) );
        }

        // Last, because it is reference rather than reading: a legend is consulted once and
        // then skipped, where the figures above it are read every time.
        if ( budget.has_value() )
        {
            for ( std::size_t column = 0; column < kDeploymentCount; ++column )
            {
                if ( kColumnMarks[ column ].empty() )
                {
                    continue;
                }

                listing.notes.push_back( std::format(
                    "{} {}", kColumnMarks[ column ], kColumnLegend[ column ] ) );
            }
        }

        return listing;
    }

    /**
     * @brief What is available to install, one line per model.
     *
     * The owner is supplied by the caller and never shown: one publisher makes it a constant
     * rather than a decision, and a name the user cannot act on is noise. Names and tags are
     * authored by whoever owns the repository -- printed as data, interpreted by nothing.
     */
    export std::vector<std::string> describeHubModels( const std::string& owner )
    {
        std::vector<std::string> lines;

        // A build without a hub says so, rather than reporting an empty listing -- that would
        // claim this publisher has nothing, when the truth is that nothing can be asked.
        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            lines.push_back(
                "This build has no HTTP transport (MILA_ENABLE_LIBCURL=OFF), so no publisher "
                "can be listed. Models already installed are shown by /models." );

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
            std::string architecture;

            // From the manifest rather than the listing's `license:` tag, which is the model
            // card's claim. The manifest is what governs once installed, so this column means
            // the same thing as the one /models shows.
            std::string license;
        };

        // What the transfer costs and what the model is -- both the manifest's answer, and the
        // repository listing knows neither. The size in particular is not the repo's size: a repo
        // also holds README, LICENSE and .gitattributes, and Mila fetches none of them. One small
        // GET per row, on a command the user asked for by name.
        const auto describe =
            [&hub, &owner]( const Mila::Distribution::HubModel& model ) -> OnlineDetail
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
                    detail.architecture = manifest.architecture;
                    detail.license = manifest.license;

                    for ( const auto& file : manifest.files )
                    {
                        detail.bytes += file.bytes;
                    }

                    return detail;
                }
                catch ( const std::exception& )
                {
                    // A repository whose manifest cannot be read is still worth listing by name.
                    // Losing one row's detail must not cost the whole listing.
                    return {};
                }
            };

        // STATUS last because it is the only variable-width column: the gated text is a sentence,
        // and anything placed after it would lose its alignment on exactly the rows that matter.
        //
        // LICENSE earns a place here more than it does in /models: by the time a model is
        // installed its terms have already been taken on, and this is the listing a user reads
        // while deciding whether to pull it.
        lines.push_back( std::format( "  {:<40} {:>10}  {:<14}{:<12}{}",
            "MODEL", "DOWNLOAD", "ARCHITECTURE", "LICENSE", "STATUS" ) );

        for ( const auto& model : models )
        {
            const OnlineDetail detail = describe( model );

            // Gating is known from the listing, so a repository behind terms says so here
            // instead of surfacing as a 403 partway through a multi-gigabyte transfer.
            lines.push_back( std::format( "  {:<40} {:>10}  {:<14}{:<12}{}{}",
                model.repository,
                detail.bytes > 0 ? formatBytes( detail.bytes ) : std::string( "--" ),
                detail.architecture.empty() ? "--" : detail.architecture,
                detail.license.empty() ? "--" : detail.license,
                isInstalled( model ) ? "installed " : "",
                model.gated ? "gated - accept terms on huggingface.co" : "" ) );
        }

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
