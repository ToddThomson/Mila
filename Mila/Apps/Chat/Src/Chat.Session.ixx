/**
 * @file Chat.Session.ixx
 * @brief One chat session over one loaded model, driven by a front end and reporting through SessionEvents.
 *
 * Supports the instruct families -- Llama, Gemma and Qwen, at BF16 compute -- each rendered and read through
 * the model's own template in the library (Conversation::Protocol).
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

    using LlamaModelBF16Type = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using GemmaModelBF16Type = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using QwenModelBF16Type  = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

    using ModelVariant = std::variant<
        std::unique_ptr<LlamaModelBF16Type>,
        std::unique_ptr<GemmaModelBF16Type>,
        std::unique_ptr<QwenModelBF16Type>
    >;

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
        }

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
            tool_handlers_.emplace( std::move( name ), std::move( handler ) );
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

                std::visit( []( auto& model ) { model.reset(); }, model_ );

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
         * @brief Answer one user turn, through as many tool rounds as it takes.
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

            history_.push_back( { MessageRole::User, std::move( text ) } );

            // Every reply streams: the template's reader routes each piece to its channel as it arrives.
            events_.turnBegan( true );

            std::string response;
            response.reserve( 4096 );

            generateResponse( response );
            finishTurn( response );

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
         * @brief Replace the conversation with one held earlier, keeping this session's system prompt.
         *
         * The next turn prefills it whole. Tools allowed in the conversation it replaces are asked about again.
         *
         * @param turns The conversation as history() held it, without its system turn.
         */
        void openConversation( std::vector<ChatMessage> turns )
        {
            clearHistory();

            for ( auto& turn : turns )
            {
                if ( turn.role != MessageRole::System )
                {
                    history_.push_back( std::move( turn ) );
                }
            }
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

            std::visit( [&]( auto& m ) { m->seedSampler( seed ); }, model_ );

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

            std::visit( []( auto& model ) { model.reset(); }, model_ );

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

                std::visit( []( auto& model ) { model.reset(); }, model_ );
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
            return !std::visit( []( const auto& model ) { return model == nullptr; }, model_ );
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

        const std::vector<ChatMessage>& history() const noexcept { return history_; }

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
            if ( !protocol_ )
            {
                return true;
            }

            return std::ranges::any_of( protocol_->templateVariables(),
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

            if ( !tokenizer_ )
            {
                return floor;
            }

            // The transcript as the next turn renders it, so the system prompt and the template's own control
            // tokens are both counted -- the two a hand-written floor would have forgotten. That turn has a user
            // message, which is also what makes the history renderable at all: Qwen's template refuses a
            // transcript with no user turn.
            std::vector<ChatMessage> next_turn = history_;
            next_turn.push_back( { MessageRole::User, std::string{} } );

            floor.transcript = encodePrompt( renderPrompt( next_turn ) ).size();

            floor.reasoning = ( config_.show_thinking && config_.thinking_capable )
                ? static_cast<std::size_t>( thinkingEffort( config_.thinking_effort ).budget )
                : 0;

            floor.minimum = floor.transcript + floor.reasoning + kAnswerHeadroomTokens;

            return floor;
        }

    private:

        /**
         * @brief Say the turn ended, as the template's reader read it round by round, and store its answer.
         *
         * @param raw The model's output verbatim, every round's, with the results the session wrote between them.
         */
        void finishTurn( const std::string& raw )
        {
            const ParsedResponse parsed = turn_display_.value_or( ParsedResponse{} );

            events_.turnEnded( parsed, raw );
            pushAnswer();
        }

        /**
         * @brief The turn's answer joins the history.
         *
         * It is the last round's reply with its reasoning, which the template decides whether to render again; a
         * turn whose last round dispatched calls has already joined the history as those calls and their results.
         */
        void pushAnswer()
        {
            if ( turn_answer_.has_value() )
            {
                history_.push_back( { MessageRole::Assistant, turn_answer_->content, {}, {}, turn_answer_->reasoning } );
            }
        }

        bool stopRequested() const
        {
            std::lock_guard<std::mutex> lock( stop_mutex_ );

            return stop_src_.stop_requested();
        }

        /**
         * @brief Ask whether a call may run, run it or decline it, and put its result in the history.
         *
         * A tool allowed for the conversation is not asked about again until the conversation ends, and a call of a
         * tool nobody registered is never asked about, since nothing would run: the model reads that it has none.
         *
         * @return False when the reply was stopped while the call waited: the call is not run, and the turn ends.
         */
        bool answerToolCall( const ToolCall& call )
        {
            ToolDecision decision = ToolDecision::AllowOnce;

            if ( tool_handlers_.contains( call.name ) && !tools_allowed_for_conversation_.contains( call.name ) )
            {
                decision = events_.approveToolCall( call );
            }

            if ( stopRequested() )
            {
                // The history keeps a result for every call it holds, so the next prompt renders; the person
                // stopped the reply, so no event reports one.
                history_.push_back( { MessageRole::Tool, std::string( kStoppedResult ), {}, call.name } );

                return false;
            }

            if ( decision == ToolDecision::AllowConversation )
            {
                tools_allowed_for_conversation_.insert( call.name );
            }

            std::string result;
            bool is_error = false;

            if ( decision == ToolDecision::Deny )
            {
                result = std::string( kDeclinedResult );
                is_error = true;
            }
            else
            {
                std::tie( result, is_error ) = dispatchTool( call );
            }

            events_.toolCalled( call, result, is_error );

            history_.push_back( { MessageRole::Tool, result, {}, call.name } );
            pending_tool_results_ += protocol_->formatToolResult( call.name, result );

            return true;
        }

        static constexpr std::string_view kDeclinedResult = "The user declined to run this tool call.";
        static constexpr std::string_view kStoppedResult = "The user stopped the reply before this tool call ran.";

        /**
         * @brief Invoke the registered handler for a tool call.
         *
         * Returns an error string when no handler is registered for the call name so
         * the model receives feedback rather than silently receiving an empty result.
         *
         * @return The handler's result, or a formatted error and true when there was none or it threw.
         */
        std::pair<std::string, bool> dispatchTool( const ToolCall& call )
        {
            const auto it = tool_handlers_.find( call.name );

            if ( it == tool_handlers_.end() )
            {
                return { std::format( "Error: no handler registered for tool '{}'", call.name ), true };
            }

            try
            {
                return { it->second( call.arguments ), false };
            }
            catch ( const std::exception& e )
            {
                return { std::format( "Error: tool '{}' handler threw: {}", call.name, e.what() ), true };
            }
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

        /// A call the template's grammar could not read is said, never dropped: the model meant to call something.
        void reportMalformedCalls( const Mila::Dnn::Conversation::Reply& reply )
        {
            for ( const auto& call : reply.malformed_tool_calls )
            {
                events_.error( std::format( "[tool call not understood: {}]", call ) );
            }
        }

        /// What the reader settled, as the session's events. A call is reported once it has been run.
        void reportReplyEvents( std::span<const Mila::Dnn::Conversation::ReplyEvent> events )
        {
            using Kind = Mila::Dnn::Conversation::ReplyEvent::Kind;

            for ( const auto& event : events )
            {
                switch ( event.kind )
                {
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

                    case Kind::ToolCallClosed:
                        break;
                }
            }
        }

        void generateResponse( std::string& response )
        {
            turn_display_.reset();
            turn_answer_.reset();

            // Before the room check below, so a turn with no room to generate reports no rounds rather than the last
            // turn's.
            last_turn_rounds_.clear();

            Mila::Dnn::Conversation::Prompt prompt = renderPrompt( history_ );
            std::vector<std::int32_t> input_tokens = encodePrompt( prompt );

            const std::size_t budget = generationBudget( input_tokens.size() );

            if ( budget == 0 )
            {
                events_.info( std::format(
                    "The prompt is {} tokens and the context is {}, so there is no room to "
                    "answer. Start a new conversation, or choose a longer context.",
                    input_tokens.size(), config_.context_length ) );

                return;
            }

            GenerateParams gen_params;
            gen_params.max_new_tokens = static_cast<int>( budget );
            gen_params.sampling = sessionSampling( config_ );

            // A round ends where the model hands over to its tools, having written every call it
            // needs: Gemma 4 at <|tool_response>, in its stop set, and Qwen 3.8 by ending its turn.
            // Each call is dispatched, the calls and their results join the history, and the
            // template renders the history again -- for Gemma, as the same model turn resumed.
            constexpr int kMaxToolRounds = 4;

            const std::stop_token stop_token = [this]
                {
                    std::lock_guard<std::mutex> lock( stop_mutex_ );

                    return stop_src_.get_token();
                }();

            for ( int round = 0; round < kMaxToolRounds; ++round )
            {
                events_.roundBegan();

                // Read against the prompt it continues, which may leave the reply inside the reasoning channel.
                Mila::Dnn::Conversation::ReplyReader reader = protocol_->readReply( prompt );

                // Per-round timing: each round has its own prefill (a full re-prefill on
                // rounds after a tool call) so the numbers stay meaningful across the turn.
                const auto round_start = std::chrono::high_resolution_clock::now();
                auto first_token_time = round_start;
                auto last_token_time = round_start;
                int round_token_count = 0;

                std::vector<float> token_gaps_ms;
                token_gaps_ms.reserve( 4096 );

                GenerateStatus round_status = GenerateStatus::Success;

                std::visit(
                    [&]( auto& m )
                    {
                        round_status = m->generate(
                            input_tokens,
                            [&]( std::int32_t tok )
                            {
                                const auto now = std::chrono::high_resolution_clock::now();
                                if ( round_token_count == 0 )
                                    first_token_time = now;
                                else
                                    token_gaps_ms.push_back(
                                        std::chrono::duration<float, std::milli>( now - last_token_time ).count() );
                                last_token_time = now;
                                ++round_token_count;

                                events_.tokenGenerated();

                                const std::string piece = tokenizer_->decode(
                                    std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                                response += piece;
                                reportReplyEvents( reader.feed( piece ) );
                            },
                            gen_params,
                            stop_token );
                    },
                    model_ );

                // The round's end settles what the reader still held, and reads a call left open as written.
                reportReplyEvents( reader.finish() );
                const Mila::Dnn::Conversation::Reply& reply = reader.reply();

                RoundStats round_stats;
                round_stats.finish_status = round_status;
                round_stats.token_budget = budget;
                round_stats.tokens_generated = round_token_count;
                round_stats.prefill_time_ms =
                    std::chrono::duration<float, std::milli>( first_token_time - round_start ).count();
                const int decode_tokens = round_token_count > 0 ? round_token_count - 1 : 0;
                const float decode_ms =
                    std::chrono::duration<float, std::milli>( last_token_time - first_token_time ).count();
                round_stats.decode_tokens_per_second =
                    ( decode_ms > 0.0f && decode_tokens > 0 )
                    ? static_cast<float>( decode_tokens ) / ( decode_ms / 1000.0f )
                    : 0.0f;

                if ( !token_gaps_ms.empty() )
                {
                    std::vector<float> ordered( token_gaps_ms );
                    const std::size_t mid = ordered.size() / 2;
                    std::nth_element( ordered.begin(), ordered.begin() + mid, ordered.end() );
                    round_stats.median_gap_ms = ordered[ mid ];

                    const std::size_t p99_index = ( ordered.size() * 99 ) / 100;
                    std::nth_element( ordered.begin(), ordered.begin() + p99_index, ordered.end() );
                    round_stats.p99_gap_ms = ordered[ p99_index ];

                    const auto max_it = std::max_element( token_gaps_ms.begin(), token_gaps_ms.end() );
                    round_stats.max_gap_ms = *max_it;
                    // +1: gap i sits between token i and token i+1 of the round.
                    round_stats.max_gap_token_index =
                        static_cast<int>( max_it - token_gaps_ms.begin() ) + 1;

                    const float stall_floor =
                        std::max( 2.5f * round_stats.median_gap_ms, 40.0f );

                    for ( const float gap : token_gaps_ms )
                    {
                        if ( gap > stall_floor )
                        {
                            ++round_stats.stall_count;
                            round_stats.stall_total_ms += gap;
                        }
                    }
                }

                round_stats.reasoning_tokens = reply.reasoning.empty()
                    ? 0
                    : static_cast<int>( tokenizer_->encode( reply.reasoning ).size() );

                // A round cut short by the budget, the context or a stop is not dispatched: a call it ends in
                // is not one the model finished.
                const bool dispatches = !reply.tool_calls.empty() && !tool_handlers_.empty()
                    && round_status == GenerateStatus::Success && !stopRequested();

                if ( dispatches )
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
                reportMalformedCalls( reply );

                if ( !dispatches )
                {
                    turn_answer_ = reply;
                    last_turn_rounds_.push_back( std::move( round_stats ) );
                    break;
                }

                last_turn_rounds_.push_back( std::move( round_stats ) );

                history_.push_back( { MessageRole::Assistant, reply.content, reply.tool_calls, {}, reply.reasoning } );

                bool stopped = false;
                pending_tool_results_.clear();

                for ( const auto& call : reply.tool_calls )
                {
                    if ( !answerToolCall( call ) )
                    {
                        stopped = true;
                        break;
                    }
                }

                response += pending_tool_results_;

                if ( stopped )
                {
                    break;
                }

                prompt = renderPrompt( history_ );
                input_tokens = encodePrompt( prompt );
            }
        }

        std::vector<std::int32_t> encodePrompt( const Mila::Dnn::Conversation::Prompt& prompt ) const
        {
            const auto token_ids = tokenizer_->encode( prompt.text );

            return std::vector<std::int32_t>( token_ids.begin(), token_ids.end() );
        }

        /**
         * @brief A history as the model's template renders it, primed for the reply.
         *
         * There is no base-model branch: `resolveModel` refuses a non-instruct model, so history is
         * always something the model was trained to read.
         */
        Mila::Dnn::Conversation::Prompt renderPrompt( const std::vector<ChatMessage>& history ) const
        {
            return protocol_->formatPrompt( history, promptOptions() );
        }

        /**
         * @brief The session's settings as the template's own variables, and the tools offered.
         *
         * Thinking is the template's switch. Effort is set only where the template has levels -- Qwen 3.8's
         * three -- and Gemma 4 has none, so no layer writes one for it (ModelHandle.md 10.6).
         */
        Mila::Dnn::Conversation::PromptOptions promptOptions() const
        {
            Mila::Dnn::Conversation::PromptOptions options;
            const std::vector<ToolDefinition> tools = activeTools();

            if ( !tools.empty() )
            {
                options.tools_json = toolsJson( tools );
            }

            for ( const auto& variable : protocol_->templateVariables() )
            {
                if ( variable.name == "enable_thinking" )
                {
                    options.template_variables[ variable.name ] = config_.show_thinking ? "true" : "false";
                }
                else if ( variable.name == "reasoning_effort" && config_.show_thinking )
                {
                    options.template_variables[ variable.name ] = qwenEffortName(
                        Mila::Dnn::Qwen::reasoningEffortFromScale( config_.thinking_effort ) );
                }
            }

            return options;
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

        /// The tools as OpenAI-style function schemas, keys in the order a request writes them.
        static std::string toolsJson( const std::vector<ToolDefinition>& tools )
        {
            nlohmann::ordered_json array = nlohmann::ordered_json::array();

            for ( const auto& tool : tools )
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

                nlohmann::ordered_json function = nlohmann::ordered_json::object();
                function[ "name" ] = tool.name;
                function[ "description" ] = tool.description;
                function[ "parameters" ] = std::move( parameters );

                nlohmann::ordered_json entry = nlohmann::ordered_json::object();
                entry[ "type" ] = "function";
                entry[ "function" ] = std::move( function );
                array.push_back( std::move( entry ) );
            }

            return array.dump();
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
            std::visit( []( auto& m ) { m.reset(); }, model_ );

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
                std::visit( []( auto& model ) { model.reset(); }, model_ );

                loadActiveModel();
            }
            catch ( const std::exception& error )
            {
                events_.error( std::format(
                    "Could not reload {} at context {}: {}", name, length, error.what() ) );

                std::visit( []( auto& model ) { model.reset(); }, model_ );

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
         * @brief Load the active tokenizer and model.
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
                initializeTokenizer();
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
         * @brief Tokens this model may generate, bounded by the context the prompt leaves free.
         *
         * The prompt already occupies context, so a budget measured from zero would ask for
         * more than can fit. This was once load-bearing: GPT-2's positional embeddings are
         * *learned*, exactly context_length of them, so position 1024 was an out-of-bounds
         * lookup -- observed at 1005 generated tokens on a 1024 context. The models now guard
         * the decode position themselves and return GenerateStatus::ContextOverflow, so this
         * is no longer what stands between a base model and a crash.
         *
         * It stays because asking for a budget that cannot fit is a worse question than asking
         * for one that can: the round ends on `length` at a number the user chose, rather than
         * on `context_limit` at one they did not.
         */
        std::size_t generationBudget( std::size_t prompt_tokens ) const
        {
            const std::size_t remaining = config_.context_length > prompt_tokens
                ? config_.context_length - prompt_tokens
                : 0;

            return std::min( config_.max_new_tokens, remaining );
        }

        void initializeTokenizer()
        {
            try
            {
                Logging::Logger::info( std::format( "Loading tokenizer from: {}", config_.tokenizer_path.string() ) );

                const std::optional<InstalledModel> installed = InstalledModel::find( config_.model_name );

                if ( !installed )
                {
                    throw std::runtime_error( std::format( "'{}' is no longer installed", config_.model_name ) );
                }

                tokenizer_ = installed->loadTokenizer();

                Logging::Logger::info( std::format( "Tokenizer loaded. Vocab size: {}", tokenizer_->getVocabSize() ) );

                // The model's own chat template, chosen by the library from what the manifest declares; the
                // catalogue has already refused a model it cannot choose one for.
                const auto& record = installed->stored().record;

                protocol_ = Mila::Dnn::chatProtocolFor( record.architecture, record.architecture_version );
            }
            catch ( const std::exception& e )
            {
                Logging::Logger::error( std::format( "Failed to load tokenizer: {}", e.what() ) );

                throw;
            }
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

        void loadModel()
        {
            const DeploymentRequest request = activeDeploymentRequest();

            // Loads, keeps the model and its plan, and takes the context length the plan resolved.
            auto adopt = [&]( auto&& model )
                {
                    if ( config_.detail == DetailLevel::All )
                    {
                        events_.log( model->toString() );
                        events_.log( model->getMemoryStats().toString() + "\n" );
                    }

                    deployment_plan_ = model->getDeploymentPlan();
                    config_.context_length = static_cast<std::size_t>( deployment_plan_->contextLength() );
                    model_ = std::move( model );
                };

            try
            {
                switch ( config_.model_type )
                {
                    case ModelType::Llama:
                        adopt( LlamaModelBF16Type::load( config_.model_path, request ) );
                        break;

                    case ModelType::Gemma:
                        adopt( GemmaModelBF16Type::load( config_.model_path, request ) );
                        break;

                    case ModelType::Qwen:
                        adopt( QwenModelBF16Type::load( config_.model_path, request ) );
                        break;

                    default:
                        throw std::runtime_error( std::format( "'{}' is not a family Chat runs", config_.model_name ) );
                }
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
            history_.clear();

            // Ends with the conversation: conversation.new, conversation.open and model.load (ChatDesktop.md 4.5).
            tools_allowed_for_conversation_.clear();

            if ( system_prompt_config_.system_prompt.empty() )
                return;

            // The template places the tools itself, from promptOptions().
            history_.push_back( { MessageRole::System, system_prompt_config_.system_prompt } );
        }

        ChatConfig config_;
        SessionEvents& events_;

        // The plan the resident model was loaded with: what /context and the one-shot JSON report,
        // rather than a second derivation of it.
        std::optional<DeploymentPlan> deployment_plan_;
        ModelVariant model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage> history_;
        std::unordered_map<std::string, std::function<std::string( const std::string& )>> tool_handlers_;
        std::unordered_set<std::string> tools_allowed_for_conversation_;

        // Replaced at the start of each turn and read by stop() from any thread, hence the lock.
        mutable std::mutex stop_mutex_;
        std::stop_source stop_src_;

        // The loaded model's chat template: it renders every prompt and reads every reply.
        std::unique_ptr<Mila::Dnn::Conversation::Protocol> protocol_;

        // The turn generateResponse() just ran, read through protocol_ round by round: every round's reasoning
        // and text, for the display, and the last round's reply when it made no call, which history keeps.
        std::optional<ParsedResponse> turn_display_;
        std::optional<Mila::Dnn::Conversation::Reply> turn_answer_;

        // A round's tool results as the template writes them, appended to the raw response once the round's
        // calls have all been answered.
        std::string pending_tool_results_;

        std::vector<RoundStats> last_turn_rounds_;
    };
}
