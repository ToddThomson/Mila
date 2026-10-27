/**
 * @file Chat.Session.ixx
 * @brief One chat session over one loaded model, driven by a front end and reporting through SessionEvents.
 *
 * Every instruct model runs through one Mila::AI::Agent, which renders and reads it through its own template and
 * runs the calls it makes; the session words what the agent reports and keeps the measurements.
 */

export module Chat.Session;

import std;
export import Chat.Config;
export import Chat.ContextFloor;
export import Chat.Message;
export import Chat.RoundStats;
export import Chat.SessionEvents;
export import Chat.SystemPrompt;
export import Chat.ThinkingEffort;
import Chat.FamilyTraits;
import Chat.Footprint;
import Chat.Json;
import Chat.ModelCatalog;
import Chat.ParsedResponse;
import Chat.Settings;
import Chat.TextChannel;
import Chat.ToolDecision;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Data;

    /**
     * @brief The model, the conversation and the settings, with every request a front end can make of them.
     *
     * It never draws and never reads input: what it has to say goes out through SessionEvents, and what a person
     * asks arrives as a call. Sentences about an operation that does work -- a load, a reload, an install -- are
     * the session's; a front end words its own commands, their usage and the settings it shows.
     */
    export class ChatSession
    {
    public:

        /**
         * @brief Construct a session from a fully-populated ChatConfig.
         *
         * Loads the system prompt only; the model is loaded by open() or loadConfiguredModel(), so a front end can
         * show something first.
         *
         * @throws std::runtime_error on system prompt load failure.
         */
        ChatSession( ChatConfig config, SessionEvents& events )
            : config_( std::move( config ) ), events_( events )
        {
            loadSystemPrompt();

            // Set once: it goes with the session to every model it is given to.
            held_session_.onAsk( [this]( const Mila::AI::ToolQuestion& question ) { return askAboutCall( question ); } );
            held_session_.withSystemPrompt( system_prompt_config_.system_prompt );
        }

        ChatSession( const ChatSession& ) = delete;
        ChatSession& operator=( const ChatSession& ) = delete;

        /**
         * @brief Register a handler for a named tool.
         *
         * The handler receives the tool's arguments as a JSON object string
         * and returns a plain string result that is fed back to the model as
         * a Tool-role message. The name must match a ToolDefinition::name
         * loaded from the system prompt file; unregistered tool calls are
         * reported and return an error string to the model rather than throwing.
         */
        void registerTool( std::string name, std::function<std::string( const std::string& )> handler )
        {
            registerTool( std::move( name ), {},
                [handler = std::move( handler )]( const std::string& arguments, const Mila::AI::ToolGrant& )
                {
                    return handler( arguments );
                } );
        }

        /**
         * @brief Register a handler for a named tool that touches files or the network, as its needs declare.
         *
         * Each call is given the session's grant narrowed to the needs (useGrant), and the handler opens files and
         * makes connections through it.
         */
        void registerTool( std::string name, Mila::AI::ToolNeeds needs,
            std::function<std::string( const std::string&, const Mila::AI::ToolGrant& )> handler )
        {
            const auto [handler_entry, added] = tool_handlers_.emplace(
                std::move( name ), std::pair{ needs, std::move( handler ) } );

            // Offered from the next response, whether or not a model is resident yet.
            if ( added )
            {
                if ( const auto tool = agentTool( handler_entry->first ) )
                {
                    session().withTool( *tool );
                }
            }
        }

        /**
         * @brief Load the configured model, reporting a failure into the session rather than out of it.
         *
         * The commands that fix a failed load are all inside the session, which is perfectly able to run with
         * nothing loaded. Leaving instead is what once turned a readable "context_length must be greater than
         * zero" into an abort.
         *
         * @return False when nothing is resident afterwards.
         */
        bool open()
        {
            if ( config_.model_name.empty() )
            {
                return false;
            }

            try
            {
                loadConfiguredModel();

                return true;
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format( "Could not load {}: {}", config_.model_name, error.what() ) );

                releaseModel();

                // An empty name IS the no-model state, tested in both places that ask.
                config_.model_name.clear();

                return false;
            }
        }

        /**
         * @brief Load the configured tokenizer and model.
         *
         * @throws std::exception when the model cannot be loaded; the caller decides how that is said.
         */
        void loadConfiguredModel()
        {
            loadActiveModel();
        }

        /**
         * @brief Answer one user turn, through as many tool rounds as it takes, up to Chat's four.
         *
         * @return False, doing nothing, when no model is resident: every generation path dereferences the model.
         */
        bool respond( std::string text )
        {
            if ( !modelIsResident() )
            {
                return false;
            }

            {
                // A fresh source per turn, so a stop asked for during one reply never ends the next.
                std::lock_guard<std::mutex> lock( stop_mutex_ );
                stop_src_ = std::stop_source{};
            }

            // Every reply streams: the template's reader routes each piece to its channel as it arrives.
            events_.turnBegan( true );

            std::string response;
            response.reserve( 4096 );

            generateResponse( text, response );

            events_.turnEnded( turn_display_.value_or( ParsedResponse{} ), response );

            return true;
        }

        /**
         * @brief Stop the reply in flight; it ends as cancelled. Callable from any thread.
         *
         * The library ends a generate with GenerateStatus::ClientCancelled on the next decode step. A tool call
         * not yet run is not run.
         */
        void stop()
        {
            std::lock_guard<std::mutex> lock( stop_mutex_ );
            stop_src_.request_stop();
        }

        /// Clear the conversation, keeping the system prompt.
        void newConversation()
        {
            clearHistory();
        }

        /**
         * @brief How far each response may run from the next on; with none, Chat's own: its tool rounds and a reply
         *        of max_new_tokens across them.
         */
        void useLimits( std::optional<Mila::AI::AgentLimits> limits )
        {
            limits_ = std::move( limits );
        }

        /// The rules every tool call passes from the next call on; with none set, every call is asked about.
        void useToolRules( Mila::AI::ToolRules rules )
        {
            session().withRules( std::move( rules ) );
        }

        /// The most a tool call may reach from the next call on -- folders and the network; with none, nothing.
        void useGrant( Mila::AI::ToolGrant grant )
        {
            session().withGrant( std::move( grant ) );
        }

        /**
         * @brief Use another system prompt file, and the tools it describes, from the next response on.
         *
         * The model stays resident: the session is given the file's prompt and offers only the tools the file
         * describes and a handler runs. The history is kept; a front end that changes prompt starts a conversation.
         *
         * @throws std::runtime_error when the file cannot be read as a system prompt.
         */
        void useSystemPrompt( const std::filesystem::path& path )
        {
            system_prompt_config_ = SystemPromptLoader::load( path );
            config_.system_prompt_path = path;

            session().withSystemPrompt( system_prompt_config_.system_prompt );
            offerTools();
        }

        /// Stop offering a tool from the next response on, as though its handler had never been registered.
        void unregisterTool( const std::string& name )
        {
            if ( tool_handlers_.erase( name ) > 0 )
            {
                offerTools();
            }
        }

        /**
         * @brief Replace the conversation with one held earlier, keeping this session's system prompt.
         *
         * The next turn prefills it whole. Tools allowed in the conversation it replaces are asked about again.
         *
         * @param turns The conversation as history() held it, without its system turn.
         */
        void openConversation( std::vector<ChatMessage> turns )
        {
            std::erase_if( turns, []( const ChatMessage& turn ) { return turn.role == MessageRole::System; } );

            session().openConversation( std::move( turns ) );
        }

        /**
         * @brief Reseed the sampler. The sampler belongs to the model, so there is nothing to seed until one is resident.
         *
         * @return False when no model is resident.
         */
        bool seed( std::uint64_t seed )
        {
            if ( !modelIsResident() )
            {
                return false;
            }

            agent_->model().model().seedSampler( seed );

            return true;
        }

        /**
         * @brief Set the display-detail level and sync the live log level.
         *
         * Detail All raises the console log level to Info (showing load dumps and
         * INFO logging); lower levels keep it quiet at Warning.
         */
        void setDetail( DetailLevel level )
        {
            config_.detail = level;

            Logging::Logger::defaultLogger().setLevel( level == DetailLevel::All
                ? Logging::LogLevel::Info
                : Logging::LogLevel::Warning );
        }

        /**
         * @brief Switch the reasoning channel. Toggling it does not reload weights.
         *
         * @return False when the model has no reasoning channel: a preference cannot give a model a channel it
         *         was not trained with, and reporting one as enabled is how a session once advertised an effort
         *         level for Llama.
         */
        bool setThinking( bool on )
        {
            if ( !config_.thinking_capable )
            {
                return false;
            }

            config_.show_thinking = on;
            applyTemplateVariables();

            return true;
        }

        /// @return False outside 1-5, or where the template's thinking has no levels.
        bool setThinkingEffort( int level )
        {
            if ( !templateHasEffortLevels() || level < 1 || level > 5 )
            {
                return false;
            }

            config_.thinking_effort = level;
            applyTemplateVariables();

            return true;
        }

        // The sampling knobs, each read per generate call. Empty returns a knob to the model's own setting; the
        // bounds are the sampler's, and a front end refuses a value outside them before it gets here.
        void setTemperature( std::optional<float> value ) { config_.temperature = value; }
        void setTopK( std::optional<int> value ) { config_.top_k = value; }
        void setTopP( std::optional<float> value ) { config_.top_p = value; }
        void setPresencePenalty( std::optional<float> value ) { config_.presence_penalty = value; }

        /**
         * @brief `/model load <name> [quantization]`: load a model, clearing the conversation.
         *
         * Quantizing on load is a deployment choice, not an identity: it lets BF16 weights too large for the card
         * run anyway. Pre-quantized weights are a different model with its own name, and refuse it.
         */
        void loadModel( std::string_view requested_name, std::optional<QuantizationMode> requested_quantization )
        {
            // Folded before the comparison below, so `/model load LLAMA-3.2-3B-INSTRUCT-FP4`
            // against that same model resident is recognised rather than reloaded.
            const std::string name = foldModelName( requested_name );

            if ( isCurrentModel( name, requested_quantization ) )
            {
                events_.info( std::format( "{} is already loaded.", modelName() ) );

                return;
            }

            try
            {
                switchModel( name, requested_quantization );
            }
            catch ( const std::exception& error )
            {
                // The session keeps its working model: resolution happens before the current one
                // is released.
                events_.error( error.what() );
            }
        }

        /// Install a published model; a session that opened with nothing resident then loads it.
        void installModel( const std::string& name )
        {
            bool installed = false;

            try
            {
                const auto progress = [this]( std::uint64_t received, std::uint64_t total )
                    {
                        events_.installProgress( received, total );
                    };

                for ( const auto& line : Mila::ChatApp::installModel( name, progress ) )
                {
                    events_.info( line );
                }

                installed = true;
            }
            catch ( const std::exception& error )
            {
                // A failed install must leave the session on its working model, so this reports
                // and returns rather than propagating.
                events_.error( std::format( "Install failed: {}", error.what() ) );
            }

            // Bootstrap: a session that opened with nothing resident wants the model it just
            // installed. Reported separately from the install because a load that fails here has
            // not failed the install, and saying so would send the user to fix the wrong thing.
            if ( installed && !modelIsResident() )
            {
                try
                {
                    switchModel( name );
                }
                catch ( const std::exception& error )
                {
                    events_.error( std::format(
                        "Installed, but loading it failed: {}", error.what() ) );
                }
            }
        }

        void removeModel( const std::string& name )
        {
            try
            {
                for ( const auto& line : Mila::ChatApp::removeModel( name ) )
                {
                    events_.info( line );
                }
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format( "Remove failed: {}", error.what() ) );
            }
        }

        /**
         * @brief Set the context to an explicit length and rebuild the model there.
         *
         * Refused rather than clamped below the floor: a context too short for the turn loads
         * perfectly well and then truncates every round, which is exactly the failure Gemma's
         * compiled 512 default produced for anyone with no config file.
         *
         * The number is passed through exactly. Rounding a requested 8000 up to the 1024 grid the
         * auto scan reports on would be the silent device override ChatConfiguration.md section 6
         * rules out: a user who writes 8192 gets 8192, and one who writes 8000 gets 8000.
         */
        void setContextLength( std::size_t length )
        {
            const FamilyTraits traits = familyTraits( config_.model_type );

            if ( modelIsResident() && length > traits.max_context )
            {
                events_.error( std::format(
                    "{} addresses {} positions at most.", modelName(), traits.max_context ) );

                return;
            }

            const ContextFloor floor = contextFloor();

            if ( floor.minimum > 0 && length < floor.minimum )
            {
                events_.error( std::format(
                    "{} is below this session's minimum of {}: the transcript renders to {} "
                    "tokens{}, and an answer needs room after it.",
                    length, floor.minimum, floor.transcript,
                    floor.reasoning > 0
                        ? std::format( " with {} budgeted for reasoning", floor.reasoning )
                        : std::string{} ) );

                events_.info(
                    "  A new conversation has no transcript to hold, and a lower thinking effort budgets less "
                    "for reasoning." );

                return;
            }

            reloadAtContext( length, false );
        }

        /**
         * @brief Measure the largest context that fits this card and rebuild there.
         */
        void setAutomaticContext()
        {
            if ( !modelIsResident() )
            {
                // Nothing to measure: auto is a question about a model's footprint, and the flag
                // is what makes the next load ask it.
                config_.context_is_automatic = true;
                config_.configured_context_length = 0;
                config_.context_origin =
                    std::string( layerName( SettingsLayer::SessionOverride ) );

                events_.info(
                    "Context set to auto; it is measured when a model loads." );

                return;
            }

            // Released before the load plans, not after: the planner reads the device's free memory,
            // so planning with this model resident would describe a card that still holds it.
            const std::string name = modelName();

            releaseModel();

            const std::size_t previous_length = config_.context_length;
            const bool previous_automatic = config_.context_is_automatic;
            const std::size_t previous_configured = config_.configured_context_length;
            const std::string previous_origin = config_.context_origin;

            config_.context_is_automatic = true;
            config_.configured_context_length = 0;
            config_.context_origin = std::string( layerName( SettingsLayer::SessionOverride ) );

            try
            {
                loadActiveModel();

                events_.info( std::format(
                    "Context {} is now re-measured on each load.", config_.context_length ) );

                return;
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format( "Could not reload {}: {}", name, error.what() ) );
            }

            // The model was released to plan, so put it back at the context the session already had
            // rather than leaving the session empty.
            config_.context_length = previous_length;
            config_.context_is_automatic = previous_automatic;
            config_.configured_context_length = previous_configured;
            config_.context_origin = previous_origin;

            try
            {
                loadActiveModel();
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format(
                    "Could not reload {} at context {}: {}", name, previous_length, error.what() ) );

                releaseModel();
                config_.model_name.clear();
            }
        }

        // ── What a front end reads ────────────────────────────────────────────

        const ChatConfig& config() const noexcept { return config_; }

        const std::string& modelName() const noexcept { return config_.model_name; }

        /// True once weights are actually on the device, which the session config cannot say --
        /// it names the model that will be loaded as readily as the one that is.
        bool modelIsResident() const
        {
            return agent_ != nullptr;
        }

        /**
         * @brief True when the named model is the one already resident in memory, so a
         *        switch would be a no-op reload.
         */
        bool isCurrentModel(
            const std::string& name, std::optional<QuantizationMode> requested ) const
        {
            // The name identifies the weights exactly -- it is unique across the store, which
            // the family/size/precision triple never was. Quantization still counts, because
            // one set of weights can be deployed at more than one.
            return modelIsResident()
                && config_.model_name == name
                && ( !requested.has_value() || *requested == config_.quantization_mode );
        }

        /// The store's spelling of a name a person typed, or the name as typed when the store does not hold it.
        static std::string foldModelName( std::string_view requested_name )
        {
            return InstalledModel::find( requested_name )
                .transform( []( const InstalledModel& model ) { return model.name(); } )
                .value_or( std::string( requested_name ) );
        }

        /// The plan the resident model was loaded with, rather than a second derivation of it.
        const std::optional<DeploymentPlan>& deploymentPlan() const noexcept { return deployment_plan_; }

        /// The tools the system prompt describes, registered or not.
        const std::vector<ToolDefinition>& tools() const noexcept { return system_prompt_config_.tools; }

        /// The conversation, without the system prompt; held across a reload, when no model is resident.
        std::span<const ChatMessage> history() const noexcept
        {
            return session().conversation();
        }

        /// The most recent turn's rounds, in the order they ran.
        const std::vector<RoundStats>& lastTurnRounds() const noexcept { return last_turn_rounds_; }

        /**
         * @brief How the last round of the turn ended, or Success when nothing has run.
         *
         * The last round is the one that produced the answer: a tool round trip finishes its
         * first round early by design, and reporting that as the turn's outcome would call a
         * complete answer truncated.
         */
        GenerateStatus finishStatus() const
        {
            return last_turn_rounds_.empty()
                ? GenerateStatus::Success
                : last_turn_rounds_.back().finish_status;
        }

        /// False where the model's template has no effort levels: Gemma's has a thinking switch alone, Llama's neither.
        bool templateHasEffortLevels() const
        {
            if ( !agent_ )
            {
                return true;
            }

            return std::ranges::any_of( agent_->model().protocol()->templateVariables(),
                []( const auto& variable ) { return variable.name == "reasoning_effort"; } );
        }

        /**
         * @brief The smallest context this session could hold a turn in.
         *
         * Derived rather than compiled, because every part is something the session already knows:
         * what the transcript renders to, what the reasoning budget will claim, and room to answer.
         * A constant 512 or 1024 would be a figure the user has to take on faith, and would be
         * wrong in both directions -- too small for a Gemma turn at high effort, too large for a
         * short-context model on a card with no room for more.
         */
        ContextFloor contextFloor() const
        {
            ContextFloor floor;

            if ( !agent_ )
            {
                return floor;
            }

            // The transcript as the next turn renders it, so the system prompt and the template's own control
            // tokens are both counted -- the two a hand-written floor would have forgotten. That turn has a user
            // message, which is also what makes the history renderable at all: Qwen's template refuses a
            // transcript with no user turn.
            floor.transcript = agent_->model().tokenizer().encode( agent_->prompt().text ).size();

            floor.reasoning = ( config_.show_thinking && config_.thinking_capable )
                ? static_cast<std::size_t>( thinkingEffort( config_.thinking_effort ).budget )
                : 0;

            floor.minimum = floor.transcript + floor.reasoning + kAnswerHeadroomTokens;

            return floor;
        }

    private:

        /// Rounds in one turn: the four Chat has run since it first ran tools, a number no measurement has set.
        static constexpr int kToolRounds = 4;

        /// The resident agent's session, or the one held while nothing is resident.
        Mila::AI::Session& session() noexcept
        {
            return agent_ ? agent_->session() : held_session_;
        }

        const Mila::AI::Session& session() const noexcept
        {
            return agent_ ? agent_->session() : held_session_;
        }

        /// The session, conversation and all, is kept across a reload, and while nothing is resident.
        void releaseModel()
        {
            if ( agent_ )
            {
                held_session_ = std::move( *agent_ ).release();
                agent_.reset();
            }

            deployment_plan_.reset();
        }

        bool stopRequested() const
        {
            std::lock_guard<std::mutex> lock( stop_mutex_ );

            return stop_src_.stop_requested();
        }

        static Mila::AI::ToolAnswer toolAnswer( ToolDecision decision ) noexcept
        {
            switch ( decision )
            {
                case ToolDecision::AllowOnce:
                    return Mila::AI::ToolAnswer::Once;

                case ToolDecision::AllowConversation:
                    return Mila::AI::ToolAnswer::ForConversation;

                case ToolDecision::Deny:
                    break;
            }

            return Mila::AI::ToolAnswer::Deny;
        }

        /**
         * @brief The tools worth telling the model about: those with a registered handler.
         *
         * Describing an unhandled tool primes the model to emit a call it will never get a
         * result for.
         */
        std::vector<ToolDefinition> activeTools() const
        {
            std::vector<ToolDefinition> active;

            for ( const auto& tool : system_prompt_config_.tools )
            {
                if ( tool_handlers_.contains( tool.name ) )
                    active.push_back( tool );
            }

            return active;
        }

        /// The session offers exactly the tools the system prompt describes and a handler runs.
        void offerTools()
        {
            Mila::AI::Session& offering = session();

            offering.withoutTools();

            for ( const auto& tool : activeTools() )
            {
                offering.withTool( *agentTool( tool.name ) );
            }
        }

        /// A tool the system prompt describes and a handler runs, as the agent offers it; nothing for any other name.
        std::optional<Mila::AI::Tool> agentTool( const std::string& name ) const
        {
            const auto definition = std::ranges::find( system_prompt_config_.tools, name, &ToolDefinition::name );
            const auto handler = tool_handlers_.find( name );

            if ( definition == system_prompt_config_.tools.end() || handler == tool_handlers_.end() )
            {
                return std::nullopt;
            }

            Mila::AI::Tool tool{
                definition->name,
                definition->description,
                parametersJson( *definition ),
                [run = handler->second.second]( std::string_view arguments, const Mila::AI::ToolGrant& grant, std::stop_token )
                {
                    return run( std::string( arguments ), grant );
                } };

            tool.needs = handler->second.first;

            return tool;
        }

        /// A tool's arguments as the JSON Schema object a request writes, keys in that order.
        static std::string parametersJson( const ToolDefinition& tool )
        {
            nlohmann::ordered_json properties = nlohmann::ordered_json::object();

            for ( const auto& [name, property] : tool.parameters.properties )
            {
                properties[ name ] = { { "type", property.type }, { "description", property.description } };
            }

            nlohmann::ordered_json parameters = nlohmann::ordered_json::object();
            parameters[ "type" ] = tool.parameters.type;
            parameters[ "properties" ] = std::move( properties );
            parameters[ "required" ] = tool.parameters.required;

            return parameters.dump();
        }

        /// A round's reasoning and text join the turn's display, in the order the rounds ran.
        void appendToTurnDisplay( const Mila::Dnn::Conversation::Reply& reply )
        {
            ParsedResponse& display = turn_display_.has_value() ? *turn_display_ : turn_display_.emplace();

            const auto append = []( std::string& into, const std::string& part )
                {
                    if ( part.empty() )
                    {
                        return;
                    }

                    into += into.empty() ? "" : "\n\n";
                    into += part;
                };

            append( display.thinking, reply.reasoning );
            append( display.answer, reply.content );
        }

        /**
         * @brief A round as the console and the desktop app measure it: when its first token came, how fast the
         *        rest followed, and where it stalled.
         *
         * Timed here, in the consumer, from the agent's events: the agent reports that a token arrived, never how
         * long it took.
         */
        class RoundTimer
        {
        public:

            void begin( std::optional<dim_t> budget )
            {
                start_ = std::chrono::high_resolution_clock::now();
                first_ = start_;
                last_ = start_;
                count_ = 0;
                budget_ = budget;
                gaps_ms_.clear();
                gaps_ms_.reserve( 4096 );
            }

            void token()
            {
                const auto now = std::chrono::high_resolution_clock::now();

                if ( count_ == 0 )
                    first_ = now;
                else
                    gaps_ms_.push_back( std::chrono::duration<float, std::milli>( now - last_ ).count() );

                last_ = now;
                ++count_;
            }

            RoundStats stats( GenerateStatus status ) const
            {
                RoundStats round_stats;
                round_stats.finish_status = status;
                round_stats.token_budget = budget_ ? static_cast<std::size_t>( *budget_ ) : 0;
                round_stats.tokens_generated = count_;
                round_stats.prefill_time_ms = std::chrono::duration<float, std::milli>( first_ - start_ ).count();

                const int decode_tokens = count_ > 0 ? count_ - 1 : 0;
                const float decode_ms = std::chrono::duration<float, std::milli>( last_ - first_ ).count();

                round_stats.decode_tokens_per_second = ( decode_ms > 0.0f && decode_tokens > 0 )
                    ? static_cast<float>( decode_tokens ) / ( decode_ms / 1000.0f )
                    : 0.0f;

                if ( !gaps_ms_.empty() )
                {
                    std::vector<float> ordered( gaps_ms_ );
                    const std::size_t mid = ordered.size() / 2;
                    std::nth_element( ordered.begin(), ordered.begin() + mid, ordered.end() );
                    round_stats.median_gap_ms = ordered[ mid ];

                    const std::size_t p99_index = ( ordered.size() * 99 ) / 100;
                    std::nth_element( ordered.begin(), ordered.begin() + p99_index, ordered.end() );
                    round_stats.p99_gap_ms = ordered[ p99_index ];

                    const auto max_it = std::max_element( gaps_ms_.begin(), gaps_ms_.end() );
                    round_stats.max_gap_ms = *max_it;
                    // +1: gap i sits between token i and token i+1 of the round.
                    round_stats.max_gap_token_index = static_cast<int>( max_it - gaps_ms_.begin() ) + 1;

                    const float stall_floor = std::max( 2.5f * round_stats.median_gap_ms, 40.0f );

                    for ( const float gap : gaps_ms_ )
                    {
                        if ( gap > stall_floor )
                        {
                            ++round_stats.stall_count;
                            round_stats.stall_total_ms += gap;
                        }
                    }
                }

                return round_stats;
            }

        private:

            std::chrono::high_resolution_clock::time_point start_;
            std::chrono::high_resolution_clock::time_point first_;
            std::chrono::high_resolution_clock::time_point last_;
            int count_{ 0 };
            std::optional<dim_t> budget_;
            std::vector<float> gaps_ms_;
        };

        /**
         * @brief Answer one message through the agent, reporting what it does as this session's events.
         *
         * @param response The model's output verbatim, every round's, with the results written between them as the
         *                 template writes them -- what the console's stream check compares the display against.
         */
        void generateResponse( const std::string& text, std::string& response )
        {
            turn_display_.reset();

            // Before the room check below, so a turn with no room to generate reports no rounds rather than the last
            // turn's.
            last_turn_rounds_.clear();

            const std::stop_token stop_token = [this]
                {
                    std::lock_guard<std::mutex> lock( stop_mutex_ );

                    return stop_src_.get_token();
                }();

            // The user's own knobs; the agent sets each one left empty at the publisher's for the thinking in effect.
            agent_->session().withSampling(
                Mila::AI::SamplingOverride{ config_.temperature, config_.top_k, config_.top_p, config_.presence_penalty } );
            agent_->session().withLimits( limits_.value_or(
                Mila::AI::AgentLimits{ .rounds = kToolRounds, .reply_tokens = static_cast<dim_t>( config_.max_new_tokens ) } ) );

            const Mila::Dnn::Conversation::Protocol& protocol = *agent_->model().protocol();
            RoundTimer timer;

            const Mila::AI::Response answered = agent_->respond( text,
                [&]( const Mila::AI::AgentEvent& event )
                {
                    using Kind = Mila::AI::AgentEvent::Kind;

                    switch ( event.kind )
                    {
                        case Kind::RoundBegan:
                            events_.roundBegan();
                            timer.begin( event.budget );
                            break;

                        case Kind::Token:
                            timer.token();
                            events_.tokenGenerated();
                            response += event.text;
                            break;

                        case Kind::ReasoningOpened:
                            events_.thoughtBreak();
                            break;

                        case Kind::Reasoning:
                            events_.text( TextChannel::Thought, event.text );
                            break;

                        case Kind::ReasoningClosed:
                            events_.thoughtClosed();
                            break;

                        case Kind::Answer:
                            events_.text( TextChannel::Answer, event.text );
                            break;

                        case Kind::ToolCallOpened:
                            events_.toolCallOpened();
                            break;

                        case Kind::ToolCalled:
                            events_.toolCalled( *event.call, event.text, event.is_error );
                            response += protocol.formatToolResult( event.call->name, event.text );
                            break;

                        case Kind::MalformedToolCall:
                            // A call the template's grammar could not read is said, never dropped: the model meant
                            // to call something.
                            events_.error( std::format( "[tool call not understood: {}]", event.text ) );
                            break;

                        case Kind::RoundEnded:
                            endRound( timer, event );
                            break;
                    }
                },
                stop_token );

            if ( answered.rounds == 0 && answered.status == GenerateStatus::ContextOverflow )
            {
                events_.info( std::format(
                    "The prompt is {} tokens and the context is {}, so there is no room to "
                    "answer. Start a new conversation, or choose a longer context.",
                    agent_->model().tokenizer().encode( agent_->prompt( text ).text ).size(), config_.context_length ) );
            }
        }

        /// A round's measurements, and its reasoning and text for the turn's display.
        void endRound( const RoundTimer& timer, const Mila::AI::AgentEvent& event )
        {
            const Mila::Dnn::Conversation::Reply& reply = *event.reply;

            RoundStats round_stats = timer.stats( event.status );

            round_stats.reasoning_tokens = reply.reasoning.empty()
                ? 0
                : static_cast<int>( agent_->model().tokenizer().encode( reply.reasoning ).size() );

            // The agent answers a round's calls when the round finished them, as it says the reason in status.
            const bool answers_calls = !reply.tool_calls.empty() && !activeTools().empty()
                && event.status == GenerateStatus::Success && !stopRequested();

            if ( answers_calls )
            {
                round_stats.ended_in_tool_call = true;

                for ( const auto& call : reply.tool_calls )
                {
                    round_stats.tool_name += round_stats.tool_name.empty() ? "" : ", ";
                    round_stats.tool_name += call.name;
                }
            }

            events_.roundEnded( round_stats );
            appendToTurnDisplay( reply );
            last_turn_rounds_.push_back( std::move( round_stats ) );
        }

        /// The agent asks the person through this session's events, which a front end answers.
        Mila::AI::ToolAnswer askAboutCall( const Mila::AI::ToolQuestion& question )
        {
            return toolAnswer( events_.approveToolCall(
                ToolCall{ std::string( question.name ), std::string( question.arguments ) } ) );
        }

        /**
         * @brief The session's settings as the template's own variables.
         *
         * Thinking is the template's switch. Effort is set only where the template has levels -- Qwen 3.8's
         * three -- and Gemma 4 has none, so no layer writes one for it (ModelHandle.md 10.6).
         *
         * Tools go in the system message where the template can place them there. Llama 3.1's template puts
         * them in the first user message by default, under an instruction to answer with a call, and Llama 3.1
         * 8B then called a tool for every question; with them in the system message and a system prompt saying
         * when to call, it answered 10 of 10 questions that needed none (measured 2026-10-25).
         */
        std::map<std::string, std::string> templateVariables() const
        {
            std::map<std::string, std::string> variables;
            const bool offers_tools = !activeTools().empty();

            for ( const auto& variable : agent_->model().protocol()->templateVariables() )
            {
                if ( variable.name == "enable_thinking" )
                {
                    variables[ variable.name ] = config_.show_thinking ? "true" : "false";
                }
                else if ( variable.name == "reasoning_effort" && config_.show_thinking )
                {
                    variables[ variable.name ] = qwenEffortName(
                        Mila::Dnn::Qwen::reasoningEffortFromScale( config_.thinking_effort ) );
                }
                else if ( variable.name == "tools_in_user_message" && offers_tools )
                {
                    variables[ variable.name ] = "false";
                }
            }

            return variables;
        }

        void applyTemplateVariables()
        {
            if ( agent_ )
            {
                agent_->withTemplateVariables( templateVariables() );
            }
        }

        static std::string qwenEffortName( Mila::Dnn::Qwen::ReasoningEffort effort )
        {
            switch ( effort )
            {
                case Mila::Dnn::Qwen::ReasoningEffort::Low:
                    return "low";

                case Mila::Dnn::Qwen::ReasoningEffort::Medium:
                    return "medium";

                case Mila::Dnn::Qwen::ReasoningEffort::Xhigh:
                    return "xhigh";
            }

            return "xhigh";
        }

        void switchModel(
            const std::string& name,
            std::optional<QuantizationMode> requested_quantization = std::nullopt )
        {
            const ModelType prev_type = config_.model_type;

            // A session that opened with nothing resident is loading, not switching, and the
            // closing message is the only part that can tell the difference.
            const bool had_model = modelIsResident();

            // Resolved before the current model is released, so a name that is not installed
            // leaves the session on its working model.
            const ResolvedModel resolved = resolveModel( name, requested_quantization );

            config_.model_name        = resolved.name;
            config_.model_type        = resolved.family;
            config_.is_instruct       = resolved.instruct;
            config_.base_model        = resolved.base_model;
            config_.license           = resolved.license;

            // Thinking is the model's capability, so it re-derives on every switch rather than
            // carrying a preference across a family that has no reasoning channel.
            config_.thinking_capable  = resolved.thinking_capable;
            config_.show_thinking     = resolved.thinking_capable;
            config_.recommended_sampling = resolved.sampling;
            config_.quantization_mode = resolved.quantization;
            config_.quantization_applied_at_load = resolved.quantization_applied_at_load;
            config_.model_path        = resolved.weights;
            config_.tokenizer_path    = resolved.tokenizer;

            // Preserve context_length across same-architecture switches. On an architecture change
            // the live value cannot carry (the ceilings differ), so fall back to what the merged
            // configuration asked for, clamped to what the new architecture can address, and only
            // to the new family's default when no layer above the defaults named a context.
            //
            // A zero is not a live value to preserve, and the architecture test alone does not
            // catch it: a session that opened with NO model holds the default model_type, so
            // loading a model of that same family skipped this and carried the zero into the load.
            // That is the container failure -- no session config, then a Llama, then
            // "context_length must be greater than zero". See ChatConfiguration.md section 2.
            const FamilyTraits traits = familyTraits( config_.model_type );

            // Destroy the current model before allocating the replacement.
            // This returns VRAM to the CUDA pool before the new model is loaded,
            // avoiding a transient old+new peak that overflows the VRAM budget
            // and forces WDDM to spill into shared system memory.
            //
            // It also has to happen before the load plans: the planner reads the device's FREE
            // memory, so a plan made while the outgoing model is resident describes a card that
            // still holds it.
            releaseModel();

            // Under auto the load plans the context for the model being loaded, rather than carrying
            // a number derived for the one being replaced -- the defect configured_context_length
            // exists to prevent, in the one shape that field cannot express.
            if ( !config_.context_is_automatic
                && ( prev_type != config_.model_type || config_.context_length == 0 ) )
            {
                const std::size_t configured = config_.configured_context_length;

                config_.context_length = configured == 0
                    ? traits.default_context
                    : ( configured < traits.max_context ? configured : traits.max_context );
            }

            loadActiveModel();
            clearHistory();

            // Written only after the load succeeds, so a name that failed to load is never the
            // one the next session opens with.
            writeLastChosenModel( config_.model_name );

            events_.info( had_model
                ? "Model switched. Conversation history cleared."
                : "Model loaded." );

            // Said on a switch as well as at startup: under auto the context is re-measured for
            // the model just loaded, and between two families that is a change of two orders of
            // magnitude. Silent would be the same unaccountable number this replaced.
            if ( config_.context_is_automatic )
            {
                events_.info( std::format( "Context {} (auto).", config_.context_length ) );
            }
        }

        /**
         * @brief Rebuild the resident model at a new context, keeping the conversation.
         *
         * Context sizes the KV cache and the activation workspaces at build time and there is no
         * in-place resize, so this is switchModel's path with the same weights. It differs in the
         * one way the user notices: a switch clears history because the tokenizer and the template
         * change, and neither changes here, so the transcript survives and re-prefills next turn.
         *
         * Same exposure switchModel has, deliberately rather than by omission: the outgoing model
         * is released before the replacement is built, so a load that fails leaves nothing
         * resident and reports into the session. The pre-flight inside loadActiveModel is what
         * makes that unlikely; a bespoke restore-and-reload path here would be a second recovery
         * convention for one command.
         */
        void reloadAtContext( std::size_t length, bool automatic )
        {
            if ( length == config_.context_length )
            {
                if ( automatic == config_.context_is_automatic )
                {
                    events_.info( std::format( "Context is already {}.", length ) );

                    return;
                }

                // A change of flag alone must not rebuild: the buffers are already this size. This
                // is `/context <the number auto chose>`, which pins a measured value, and its
                // inverse -- and a multi-second reload to flip a bool would be indefensible.
                config_.context_is_automatic = automatic;
                config_.configured_context_length = automatic ? 0 : length;
                config_.context_origin =
                    std::string( layerName( SettingsLayer::SessionOverride ) );

                events_.info( automatic
                    ? std::format( "Context {} is now re-measured on each load.", length )
                    : std::format( "Context {} is now pinned.", length ) );

                return;
            }

            const std::string name = modelName();

            // Every field the attempt is about to overwrite, so a failure leaves the session
            // describing the context it would return to rather than the one that did not load.
            const std::size_t previous_length = config_.context_length;
            const bool previous_automatic = config_.context_is_automatic;
            const std::size_t previous_configured = config_.configured_context_length;
            const std::string previous_origin = config_.context_origin;

            config_.context_length = length;
            config_.context_is_automatic = automatic;

            // What was ASKED for, so a later model switch carries this rather than dropping to the
            // next family's default. Cleared under auto, where the answer is re-measured per model.
            config_.configured_context_length = automatic ? 0 : length;
            config_.context_origin = std::string( layerName( SettingsLayer::SessionOverride ) );

            try
            {
                // Released before the replacement is built, so the two never both hold VRAM.
                releaseModel();

                loadActiveModel();
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format(
                    "Could not reload {} at context {}: {}", name, length, error.what() ) );

                releaseModel();

                config_.context_length = previous_length;
                config_.context_is_automatic = previous_automatic;
                config_.configured_context_length = previous_configured;
                config_.context_origin = previous_origin;

                // An empty name IS the no-model state, which is what the session now is. The
                // transcript is left alone: it is what the reload was protecting.
                config_.model_name.clear();

                events_.info( std::format(
                    "Nothing is loaded. Loading {} again returns it at context {}.",
                    name, previous_length ) );

                return;
            }

            events_.info( automatic
                ? std::format( "Context {} (auto). Conversation kept.", length )
                : std::format( "Context {}. Conversation kept.", length ) );
        }

        /**
         * @brief Load the active model as an agent.
         *
         * Reported as a load from start to end, so a front end can show progress across the multi-second
         * weight load.
         */
        void loadActiveModel()
        {
            // Checked against the key a user can edit, before anything is constructed. The
            // library's own guard is correct and names LanguageModelConfig, which is a type the
            // reader of a session file has never heard of and cannot change. Under auto there is no
            // number yet: the load plans one.
            if ( !config_.context_is_automatic && config_.context_length == 0 )
            {
                throw std::invalid_argument(
                    "context_length is zero. Set a positive 'context_length' in the session "
                    "config, or remove the key to take the model's own default." );
            }

            // The quantization is quoted only when it was chosen at load. A pre-quantized model
            // already carries it in its name, and "gemma-4-12b-it-fp4 (fp4)" says one fact twice.
            events_.loadBegan( config_.quantization_applied_at_load
                ? std::format( "Loading {} ({})",
                    modelName(), quantizationName( config_.quantization_mode ) )
                : std::format( "Loading {}", modelName() ) );

            // Ended here rather than in each caller's catch: the one place that begins it owns ending it.
            try
            {
                loadModel();
            }
            catch ( ... )
            {
                events_.loadEnded();

                throw;
            }

            events_.loadEnded();
        }

        /**
         * @brief The request this session's load sends: the model's formats, and the context either fixed
         *        or left to the planner below the family's ceiling.
         */
        DeploymentRequest activeDeploymentRequest() const
        {
            DeploymentRequest request =
                deploymentRequestFor( config_.quantization_mode, config_.quantization_applied_at_load, config_.device_index,
                    config_.kv_cache );

            if ( config_.context_is_automatic )
            {
                request.withAutomaticContextLength(
                    DeploymentRequest::kContextStep,
                    static_cast<dim_t>( familyTraits( config_.model_type ).max_context ) );
            }
            else
            {
                request.withContextLength( static_cast<dim_t>( config_.context_length ) );
            }

            return request;
        }

        /**
         * @brief A refusal in this session's words: what does not fit, against what, and what to try.
         *
         * The library names the reason and the numbers; the sentence is Chat's (Deployment.md section 6).
         */
        std::string describeRefusal( const DeploymentRefusal& refusal ) const
        {
            const std::string free = formatBytes( refusal.reading().free_bytes );

            switch ( refusal.reason() )
            {
                case DeploymentRefusal::Reason::WeightsExceedDevice:
                    return std::format(
                        "its weights need {} and this GPU has {} free, so no context length fits.{}",
                        formatBytes( refusal.footprint().device_parameter_bytes ), free,
                        config_.quantization_mode == QuantizationMode::None
                            ? std::string( " Quantized to fp4 as it loads, it needs less." ) : std::string{} );

                case DeploymentRefusal::Reason::FixedContextDoesNotFit:
                    return std::format(
                        "context {} needs {} and this GPU has {} free. An automatic context picks the longest that fits.",
                        refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::NothingAboveTheFloorFits:
                    return std::format(
                        "even context {} needs {} and this GPU has {} free.",
                        refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::DeviceDoesNotReportMemory:
                    return "this GPU does not report its memory, so no context length can be measured for it. "
                           "Choose one yourself.";
            }

            return refusal.toString();
        }

        /**
         * @brief Open the model by its name, offer it the tools, and carry the conversation held across a reload.
         *
         * Takes the context length the plan resolved.
         */
        void loadModel()
        {
            try
            {
                // The held session -- prompt, tools, rules, grant, asker and conversation -- goes to the agent only
                // once the model has opened, so a load that fails leaves it here.
                agent_ = std::make_unique<Mila::AI::Agent>(
                    Mila::AI::Agent::create( config_.model_name, activeDeploymentRequest(), std::move( held_session_ ) ) );

                if ( config_.detail == DetailLevel::All )
                {
                    events_.log( agent_->model().model().toString() );
                    events_.log( agent_->model().model().getMemoryStats().toString() + "\n" );
                }

                deployment_plan_ = agent_->plan();
                config_.context_length = static_cast<std::size_t>( deployment_plan_->contextLength() );

                applyTemplateVariables();
            }
            catch ( const DeploymentRefusedError& error )
            {
                deployment_plan_.reset();

                throw std::runtime_error( describeRefusal( error.refusal() ) );
            }
        }

        /**
         * @brief Load the system prompt and tool definitions from file.
         *
         * No-op when system_prompt_path is not set in config. File existence is validated by main() before
         * construction, so a missing file here is a logic error.
         */
        void loadSystemPrompt()
        {
            if ( !config_.system_prompt_path.has_value() )
                return;

            try
            {
                system_prompt_config_ = SystemPromptLoader::load(
                    *config_.system_prompt_path );
            }
            catch ( const std::exception& e )
            {
                std::cerr << "Error loading system prompt: " << e.what() << "\n";
                throw;
            }
        }

        void clearHistory()
        {
            // Ends the tools allowed for the conversation with it: conversation.new, conversation.open and model.load
            // (MilaApp.md 4.5).
            session().newConversation();
        }

        ChatConfig config_;
        SessionEvents& events_;

        // The plan the resident model was loaded with: what /context and the one-shot JSON report,
        // rather than a second derivation of it.
        std::optional<DeploymentPlan> deployment_plan_;
        SystemPromptConfig system_prompt_config_;
        // Each tool's declared needs, and its handler.
        std::unordered_map<std::string, std::pair<Mila::AI::ToolNeeds,
            std::function<std::string( const std::string&, const Mila::AI::ToolGrant& )>>> tool_handlers_;

        // Set by a front end that runs its own limits; Chat's own when empty.
        std::optional<Mila::AI::AgentLimits> limits_;

        // The resident model and its session; empty when nothing is resident.
        std::unique_ptr<Mila::AI::Agent> agent_;

        // The session while no agent holds it: before the first load, and between releasing a model and loading
        // the next. Moved from while an agent holds it.
        Mila::AI::Session held_session_;

        // Replaced at the start of each turn and read by stop() from any thread, hence the lock.
        mutable std::mutex stop_mutex_;
        std::stop_source stop_src_;

        // Every round's reasoning and text of the turn generateResponse() just ran, for the display.
        std::optional<ParsedResponse> turn_display_;

        std::vector<RoundStats> last_turn_rounds_;
    };
}
