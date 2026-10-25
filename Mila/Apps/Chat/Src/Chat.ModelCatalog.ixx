/**
 * @file Chat.ModelCatalog.ixx
 * @brief Resolving a model name against the store, and the store management commands.
 *
 * There is no catalogue. A model is whatever the store holds under its name, so the set Chat can
 * load is discovered at runtime rather than compiled in -- which is the point, since a compiled
 * table could not name a model the user installed after the build.
 */

export module Chat.ModelCatalog;

import std;
export import Chat.ResolvedModel;
export import Chat.RefusalAudience;
export import Chat.ModelListing;
export import Chat.FootprintBudget;
export import Chat.PublishedModel;

import Chat.Ansi;
import Chat.Config;
import Chat.FamilyTraits;
import Chat.Footprint;
import Mila;
import nlohmann.json;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using Mila::Distribution::ModelCapabilities;

    /**
     * @brief Where the session remembers the model it was last told to use.
     *
     * Beside the STORE, not beside the executable, for three reasons that agree: the store is what
     * actually holds models, it is the directory a container mounts as a volume so the choice
     * survives `--rm`, and writing here never mutates a tracked file in a developer's checkout.
     *
     * This replaces a compiled-in default model. A fresh install has no model, so naming one
     * reported a specific multi-gigabyte model as "missing" to a user who had never asked for
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
     * @brief What a model can do, as one line: its capabilities, MTP, then any modality besides text this build runs.
     *
     * "base model" when it declares nothing, which is what a model published before capabilities existed reads as.
     */
    std::string describeCapabilities( const ModelCapabilities& capabilities, bool drafts )
    {
        const std::pair<bool, std::string_view> declared[] = {
            { capabilities.instruct, "instruct" },
            { capabilities.thinking, "thinking" },
            { capabilities.tools, "tools" },
            { capabilities.agentic, "agentic" },
            { drafts, "MTP" } };

        std::string line;

        const auto append = [&line]( std::string_view word )
            {
                line += line.empty() ? "" : ", ";
                line += word;
            };

        for ( const auto& [present, word] : declared )
        {
            if ( present )
            {
                append( word );
            }
        }

        for ( const auto& modality : capabilities.modalities )
        {
            if ( isSupportedModality( modality ) )
            {
                append( modality );
            }
        }

        return line.empty() ? std::string( "base model" ) : line;
    }

    /**
     * @brief Resolve a model name against the store.
     *
     * The store is the only source. Nothing here consults a hub, reads a models directory or
     * accepts a path -- a load is not a deliberate act the way an install is, so it must never
     * become a multi-gigabyte transfer. What the name means -- its family, its stored format, its
     * capabilities -- is the library's answer (InstalledModel); what Chat adds is its own refusal of a
     * base model and its own words.
     *
     * @param requested_name The name as typed; the store folds its case.
     * @param requested_quantization Quantize unquantized weights on the way in. Empty loads
     *        the weights as they are.
     * @param audience Whether a refusal is read inside a session or at the shell.
     *
     * @throws std::runtime_error if no model of that name is installed, if this build cannot
     *         load what the record describes, or if the requested quantization contradicts it.
     */
    export ResolvedModel resolveModel(
        const std::string& requested_name,
        std::optional<QuantizationMode> requested_quantization = std::nullopt,
        RefusalAudience audience = RefusalAudience::Session )
    {
        // At the shell the remedy is a command to type. Inside a session it is said in words, because the session
        // serves front ends that do not share commands -- the console's slash commands, the window's menus.
        const bool at_shell = ( audience == RefusalAudience::Shell );

        const std::optional<InstalledModel> installed = InstalledModel::find( requested_name );

        if ( !installed.has_value() )
        {
            // The alternatives are named here rather than pointing at the listing, because this
            // also fires at startup, where the session never opens and no command can be run.
            std::string available;

            for ( const auto& model : InstalledModel::list() )
            {
                available += available.empty() ? "" : ", ";
                available += model.name();
            }

            if ( available.empty() )
            {
                throw std::runtime_error( at_shell
                    ? std::format(
                        "No model named '{}' is installed, and neither is anything else.\n"
                        "Install one with mila-store install <name>, or from a package you built with "
                        "ExportArtifact --install.", requested_name )
                    : std::format(
                        "No model named '{}' is installed, and neither is anything else. Install one first.",
                        requested_name ) );
            }

            // Deliberately does not ask the publisher whether the name exists there. A load is the
            // offline command, and a typo must not become a network wait.
            throw std::runtime_error( at_shell
                ? std::format(
                    "No model named '{}' is installed.\nInstalled: {}\n"
                    "mila-store models --online shows what can be installed.", requested_name, available )
                : std::format( "No model named '{}' is installed.\nInstalled: {}", requested_name, available ) );
        }

        // Every message below names the model as the store holds it rather than as it was typed.
        const std::string& name = installed->name();

        if ( !installed->isComplete() )
        {
            throw std::runtime_error( at_shell
                ? std::format(
                    "'{}' is installed but its files are missing, so it cannot be loaded.\n"
                    "Reinstall it with mila-store install {}, or remove it with mila-store remove {}.",
                    name, name, name )
                : std::format(
                    "'{}' is installed but its files are missing, so it cannot be loaded. "
                    "Install it again, or remove it.", name ) );
        }

        const ModelCapabilities capabilities = installed->capabilities();

        // Chat is an INSTRUCT harness -- turns, templates, a system prompt, history, tool calls.
        // A base model uses none of them, and accommodating one meant switching all four off and
        // explaining why the harness was not behaving like itself. That is a second convention
        // inside one surface, so the model is refused here rather than special-cased throughout.
        if ( !capabilities.instruct )
        {
            throw std::runtime_error( std::format(
                "'{}' is a base model: it continues text rather than answering a conversation, "
                "so it cannot be chatted with.", name ) );
        }

        // The library's own refusal of a family it cannot deploy, before the resident model is released.
        requirePlannable( *installed );

        const std::optional<QuantizationMode> stored = installed->storedWeightQuantization();

        if ( !stored.has_value() )
        {
            throw std::runtime_error( std::format(
                "'{}' stores its weights as '{}', which this build of Mila does not read.",
                name, installed->stored().record.weight_quantization ) );
        }

        ResolvedModel resolved;
        resolved.name = name;
        resolved.weights = installed->weightsPath();
        resolved.tokenizer = installed->tokenizerPath();
        resolved.family = installed->family();
        resolved.instruct = capabilities.instruct;
        resolved.base_model = installed->stored().record.base_model;
        resolved.license = installed->license();
        resolved.quantization = *stored;

        // A format other than the stored one is a quantize-on-load request, checked by the library's own rule
        // while the resident model is still loaded, so a refusal leaves the session on its working model.
        if ( requested_quantization.has_value() && *requested_quantization != *stored )
        {
            requireStoredQuantizationMatches( name, installed->weightsPath().string(),
                *stored == QuantizationMode::None ? std::string{} : weightQuantizationName( *stored ),
                *requested_quantization );

            resolved.quantization = *requested_quantization;
            resolved.quantization_applied_at_load = true;
        }

        resolved.streaming_capable = familyTraits( resolved.family ).streaming_capable;
        resolved.thinking_capable = capabilities.thinking;
        resolved.tools_capable = capabilities.tools;
        resolved.agentic = capabilities.agentic;
        resolved.sampling = Distribution::recommendedSampling( installed->stored().record );

        return resolved;
    }

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
     * harness limitation. So `no` means planned and too big, and `not supported` means the question
     * does not arise.
     *
     * A caveat names the quantization argument that makes the answer yes, so the cell is also the
     * command. No context length: what fits is a range, and `/context auto` reports it.
     */
    struct RowVerdict
    {
        std::string text;

        /// Whether it fits, for the tint. Unknown leaves the cell uncoloured.
        FootprintVerdict verdict{ FootprintVerdict::Unknown };
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
     * The planner's answer against the card's whole memory (Deployment.md 8.1), so the row and the load it
     * describes are one decision -- the context ladder this replaced claimed 128K for Gemma where the session
     * ran 56320.
     */
    RowVerdict verdictFor(
        const InstalledModel& model,
        Mila::Dnn::dim_t fixed_context_length,
        int device_index )
    {
        RowVerdict row;

        // The INCOMPLETE marker on the row already says what happened, so this does not repeat it.
        if ( !model.isComplete() )
        {
            row.text = "-";

            return row;
        }

        // Chat refuses a base model and a family it does not run at load, so planning one would price a load
        // that cannot happen.
        const ModelType family = model.family();

        if ( !model.capabilities().instruct
            || ( family != ModelType::Gemma && family != ModelType::Llama && family != ModelType::Qwen ) )
        {
            row.text = "not supported";

            return row;
        }

        DeploymentRequest request;
        request.withDevice( DeviceId{ DeviceType::Cuda, device_index } ).withWholeDeviceMemory();

        if ( fixed_context_length > 0 )
        {
            request.withContextLength( fixed_context_length );
        } else {
            request.withAutomaticContextLength( DeploymentRequest::kContextStep,
                static_cast<Mila::Dnn::dim_t>( familyTraits( family ).max_context ) );
        }

        const std::optional<std::vector<QuantizationMode>> fitting =
            fittingWeightQuantizations<DeviceType::Cuda>( model, request );

        // Nothing fit, and the two reasons for that are not the same answer: planned-and-too-big is
        // a fact about the card, where a plan that could not be made is one about this build.
        if ( !fitting.has_value() )
        {
            row.text = "unknown";

            return row;
        }

        if ( fitting->empty() )
        {
            // Plain, with no figure: the card's capacity is named once beneath the table, so repeating
            // it on every row that misses would be the same number several times.
            row.verdict = FootprintVerdict::DoesNotFit;
            row.text = "no";

            return row;
        }

        row.verdict = FootprintVerdict::Fits;

        // The stored format leads, so a model that runs as it is reads "yes". Otherwise BOTH quantized forms are
        // named, because fp8 costs less accuracy and fp4 less memory, and which to reach for is the reader's call.
        if ( fitting->front() == *model.storedWeightQuantization() )
        {
            row.text = "yes";
        } else {
            row.text = fitting->size() == 1
                ? std::format( "yes, at {}", quantizationName( fitting->front() ) )
                : std::format( "yes, at {} or {}", quantizationName( fitting->front() ),
                    quantizationName( fitting->back() ) );
        }

        return row;
    }

    /**
     * @brief What the store holds, one line per model plus a total.
     *
     * Reads only the record tree by default, so it is instant and works with no network, with
     * no device, and in a build with no hub at all. That default matters beyond speed: this
     * also renders --help, which runs before the runtime is initialized.
     *
     * @param budget Engaged adds the fit column, at the price of a weights-header read and a few
     *        planned deployments per row. No device memory is committed by it. The session assembles
     *        it, because both figures in it are facts about the live session.
     */
    export ModelListing describeInstalledModels(
        std::optional<FootprintBudget> budget = std::nullopt )
    {
        Mila::Distribution::ModelStore store;

        const std::vector<InstalledModel> models = InstalledModel::list( store );

        ModelListing listing;

        if ( models.empty() )
        {
            // Points at the listing rather than at /model install, because this is the first-run
            // state and a user with an empty store has no name to pass to /model install yet.
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
                    model, budget->fixed_context_length, budget->device_index );

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
                ( budget.has_value() && budget->resident_model == model.name() )
                    ? "* " : "  ";

            // A record whose blobs went missing is shown rather than hidden: a store that
            // silently omits a broken entry cannot be repaired by the person who owns it.
            listing.table.push_back( budget.has_value()
                ? std::format( "{}{:<30}{}{}", marker, model.name(), runs,
                    model.isComplete() ? "" : "  [INCOMPLETE - blobs missing]" )
                : std::format( "{}{}{}", marker, model.name(),
                    model.isComplete() ? "" : "  [INCOMPLETE - blobs missing]" ) );
        }

        const auto usage = store.usage();

        // Disk stays in the total and off the rows. The aggregate is refcounted and answers the
        // only question a byte count here serves -- what is this store costing me -- while a
        // per-row size is not what removing that row would return, because blobs are shared.
        listing.table.push_back( std::format( "  {} model(s), {} on disk",
            usage.model_count,
            formatBytes( usage.blob_bytes ) ) );

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
     * @brief Every model a publisher offers, with what its manifest says.
     *
     * One small GET per row for the manifest, on a request the person made by name: the listing API knows neither
     * what the transfer costs nor what the model is. A repository whose manifest cannot be read is still listed by
     * name, with manifest_read false.
     *
     * @throws std::exception when the publisher cannot be listed at all.
     */
    export std::vector<PublishedModel> listPublishedModels( const std::string& owner )
    {
        std::vector<PublishedModel> published;

        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            return published;
        }

        const auto hub = Mila::Distribution::makeDefaultModelHub();

        const auto models = hub->listModels( owner );

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

        for ( const auto& model : models )
        {
            PublishedModel row;
            row.name = model.repository;
            row.gated = model.gated;
            row.installed = isInstalled( model );

            if ( model.hasManifest() )
            {
                try
                {
                    const Mila::Distribution::ModelCoordinate coordinate{ owner, model.repository };

                    const auto manifest = Mila::Distribution::parseModelManifest(
                        hub->fetchManifest( coordinate ), model.repository );

                    for ( const auto& file : manifest.files )
                    {
                        row.download_bytes += file.bytes;
                    }

                    // Architecture, weight format and instruct are decided here with certainty, which is most of
                    // why a first install fails.
                    const ModelType family = modelTypeFromString( manifest.architecture );
                    const auto quantization = parseWeightQuantization( manifest.weight_quantization );

                    row.license = manifest.license;
                    row.quantization = quantization.has_value()
                        ? std::string( quantizationName( *quantization ) )
                        : manifest.weight_quantization;
                    row.supported = manifest.capabilities.instruct
                        && ( family == ModelType::Gemma || family == ModelType::Llama || family == ModelType::Qwen )
                        && quantization.has_value();
                    row.manifest_read = true;
                }
                catch ( const std::exception& )
                {
                    // Losing one row's detail must not cost the whole listing.
                }
            }

            published.push_back( std::move( row ) );
        }

        return published;
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

        const std::vector<PublishedModel> models = listPublishedModels( owner );

        if ( models.empty() )
        {
            lines.push_back( "No models are available to install." );

            return lines;
        }

        /**
         * What can be settled from a manifest alone, and nothing beyond it.
         *
         * The same three states `/model list` uses, so one word means one thing across both listings:
         * `not supported` for a base model or an architecture this build lacks, which no card would
         * change; `no` and `yes` for the card.
         *
         * Memory is not settled: the manifest has no geometry, so there is no footprint to predict, and the
         * download size is a LOWER bound on the device weights rather than an estimate of them -- FP4 unpacks,
         * and the unquantized tables ride along. That bound still buys the one honest memory verdict: a download
         * larger than the whole card cannot possibly fit, so that `no` is certain where the `yes` beside it is
         * provisional.
         */
        const auto fitOf = [available_bytes]( const PublishedModel& model ) -> std::string_view
            {
                if ( !model.manifest_read )
                {
                    return {};
                }

                if ( !model.supported )
                {
                    return "not supported";
                }

                return ( available_bytes > 0 && model.download_bytes > available_bytes ) ? "no" : "yes";
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
            const std::string_view fit = fitOf( model );

            if ( model.gated )
            {
                gated.push_back( model.name );
            }

            lines.push_back( std::format( "  {:<32} {:>10}  {:<12}{:<15}{}",
                model.name,
                model.download_bytes > 0 ? formatBytes( model.download_bytes ) : std::string( "--" ),
                model.license.empty() ? "--" : model.license,
                fit.empty() ? "--" : fit,
                model.installed ? "yes" : "no" ) );
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

        if ( const std::optional<InstalledModel> model = InstalledModel::find( name ) )
        {
            const auto& record = model->stored().record;

            std::uint64_t bytes = 0;

            for ( const auto& file : record.files )
            {
                bytes += file.bytes;
            }

            lines.push_back( field( "Model:", model->name() ) );

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
            if ( const auto attribution = model->attribution(); !attribution.empty() )
            {
                lines.push_back( field( "Attribution:", attribution ) );
            }

            lines.push_back( field( "Architecture:",
                record.architecture.empty() ? "-" : record.architecture ) );
            lines.push_back( field( "Variant:",
                record.variant.empty() ? "-" : record.variant ) );
            lines.push_back( field( "Capabilities:",
                describeCapabilities( model->capabilities(), model->drafter().has_value() ) ) );
            lines.push_back( field( "Installed:", model->isComplete()
                ? formatBytes( bytes ) : "yes, but its files are missing" ) );
            lines.push_back( field( "Origin:", record.origin() ) );

            return lines;
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

        if ( const auto attribution = requiredAttribution( manifest.license );
            !attribution.empty() )
        {
            lines.push_back( field( "Attribution:", attribution ) );
        }

        lines.push_back( field( "Architecture:",
            manifest.architecture.empty() ? "-" : manifest.architecture ) );
        lines.push_back( field( "Variant:",
            manifest.variant.empty() ? "-" : manifest.variant ) );
        // MTP needs the installed drafter and this build's support for it, so a published model reports only what
        // its manifest declares.
        lines.push_back( field( "Capabilities:", describeCapabilities( manifest.capabilities, false ) ) );
        lines.push_back( field( "Installed:", "no" ) );
        lines.push_back( field( "Download:", formatBytes( bytes ) ) );

        return lines;
    }

    /**
     * @brief Install a published model into the store by name.
     *
     * @param progress Called with each file's bytes received so far and its size, as often as the transfer
     *        reports; a front end decides how much of that to show. One call serves every file of the install.
     */
    export std::vector<std::string> installModel(
        const std::string& spec,
        const std::function<void( std::uint64_t received, std::uint64_t total )>& progress )
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

        auto forward = [&progress]( std::uint64_t received, std::uint64_t total ) -> bool
            {
                if ( total > 0 )
                {
                    progress( received, total );
                }

                return true;
            };

        const auto hub = Mila::Distribution::makeDefaultModelHub( forward );

        Mila::Distribution::ModelResolver resolver( store, *hub );

        const auto pulled = resolver.pull(
            spec, std::string( Mila::Distribution::kDefaultHubOwner ) );

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

        // The store's own spelling, for the messages: the store folds case, so any spelling removes it.
        const std::optional<InstalledModel> installed = InstalledModel::find( requested_name, store );
        const std::string name = installed ? installed->name() : requested_name;

        const auto report = store.remove( name );

        if ( report.records_removed == 0 && report.retained.empty() )
        {
            lines.push_back( std::format( "{} is not installed.", name ) );

            return lines;
        }

        if ( report.records_removed == 0 )
        {
            lines.push_back( std::format(
                "{} was not removed: some of its files are in use. Close whatever has it loaded, "
                "then remove it again.", name ) );
        } else {
            lines.push_back( std::format( "Removed {} -- {} blob(s), {} reclaimed.",
                name, report.blobs_removed, formatBytes( report.bytes_reclaimed ) ) );
        }

        if ( !report.unreadable_records.empty() )
        {
            lines.push_back(
                "  Its files were kept, because these records cannot be read:" );

            for ( const auto& unreadable : report.unreadable_records )
            {
                lines.push_back( std::format( "    {}", unreadable ) );
            }
        } else if ( report.records_removed == 1 && report.blobs_removed == 0 ) {
            // Shared blobs survive by design; saying so pre-empts "why did that free so little".
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
