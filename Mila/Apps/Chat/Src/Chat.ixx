/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Supports the instruct families -- Llama, Gemma and Qwen, at BF16 compute. Gemma and Qwen render and parse
 * through the model's own template in the library (Conversation::Protocol); Llama through Chat's
 * MessageFormatter and ToolCallParser until its packages declare which template they carry.
 */

export module Mila.Chat;

import std;
export import Chat.Config;
export import Chat.ModelCatalog;
export import Chat.Message;
export import Chat.MessageFormatter;
export import Chat.SystemPrompt;
export import Chat.ToolCallParser;
import Chat.ChannelParser;
import Chat.FamilyTraits;
import Chat.Footprint;
import Chat.Json;
import Chat.Renderer;
import Chat.RichText;

// For the layer vocabulary only: /context reports which layer set the value, and naming the layer
// in one place is what keeps "this session" from being spelled two ways.
import Chat.Settings;
import Chat.StreamingDisplay;

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

    export class Chat
    {
    public:

        /**
         * @brief Construct a Chat session from a fully-populated ChatConfig.
         *
         * Loads the system prompt only; the tokenizer and model are loaded by run()
         * after the welcome banner so the multi-second weight load runs under the
         * progress spinner. Throws on system prompt load failure.
         *
         * @param config Session configuration.
         * @throws std::runtime_error on system prompt load failure.
         */
        explicit Chat( ChatConfig config )
            : config_( std::move( config ) )
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
         * logged and return an error string to the model rather than throwing.
         *
         * @param name    Tool name matching the ToolDefinition in the system prompt.
         * @param handler Callable taking a JSON arguments string, returning a result string.
         */
        void registerTool( std::string name, std::function<std::string( const std::string& )> handler )
        {
            tool_handlers_.emplace( std::move( name ), std::move( handler ) );
        }

        /**
         * @brief Answer one prompt and return, without opening a session.
         *
         * Standard output carries the answer and nothing else -- no banner, no spinner, no
         * session status -- so that `mila-chat -p "..." > answer.txt` yields an answer rather
         * than an answer wearing a welcome box.
         *
         * @param answer_out The real standard output. main.cpp has already pointed std::cout at
         *        standard error, because the diagnostics to be kept out of the answer include
         *        the library's own log records and those are not written through this class.
         *        Everything reaching std::cout below is therefore already diverted.
         *
         * @return An exit code from the contract in ChatConfiguration.md section 9.
         */
        int runOnce( const std::string& prompt, bool as_json, std::ostream& answer_out )
        {
            one_shot_ = true;
            renderer_.setQuiet( true );

            if ( config_.model_name.empty() )
            {
                std::cerr << "No model is loaded. Name one with --model, "
                             "or install one with 'mila-store install <name>'.\n";

                return 4;
            }

            try
            {
                loadActiveModel();
            }
            catch ( const std::exception& error )
            {
                std::cerr << std::format( "Could not load {}: {}\n",
                    config_.model_name, error.what() );

                return 5;
            }

            clearHistory();
            history_.push_back( { MessageRole::User, prompt } );

            std::string response;
            response.reserve( 4096 );

            generateResponse( response );
            handleResponse( response );

            // handleResponse pushes the parsed answer as the final Assistant turn, so the last
            // entry is the answer after any tool round trip -- which is what a caller asked for,
            // rather than the raw text with its channels still in it.
            const std::string answer = history_.empty() ? std::string{} : history_.back().content;

            if ( as_json )
            {
                emitOneShotJson( answer, answer_out );
            }
            else
            {
                answer_out << answer << '\n';

                // A truncated answer is still an answer, so it is not a failure code. It is said
                // once on stderr, where it cannot corrupt the thing being piped.
                if ( finishStatus() == GenerateStatus::MaxNewTokensReached )
                {
                    std::cerr << "Note: response hit the token cap without a stop token "
                                 "(finish: length).\n";
                }
            }

            return 0;
        }

        void run()
        {
            printBanner();

            if ( config_.model_name.empty() )
            {
                // Said once, here: which model was configured and what the store does hold is a
                // startup fact, and it stops being true as soon as anything is installed.
                if ( !config_.no_model_reason.empty() )
                {
                    renderer_.printInfo( config_.no_model_reason );
                }

                reportNoModel();
            }
            else
            {
                // A load that fails is reported INTO the session, not out of it -- the commands
                // that fix it (/model and its subcommands) are all inside. Leaving here is what
                // turned a readable "context_length must be greater than zero" into an abort,
                // and the session is perfectly able to run with nothing loaded.
                try
                {
                    loadActiveModel();
                }
                catch ( const std::exception& error )
                {
                    renderer_.printError( std::format(
                        "Could not load {}: {}", config_.model_name, error.what() ) );

                    std::visit( []( auto& model ) { model.reset(); }, model_ );

                    // An empty name IS the no-model state, tested in both places that ask.
                    config_.model_name.clear();

                    reportNoModel();
                }
            }

            printSessionStatus();

            clearHistory();

            std::string user_input;

            while ( true )
            {
                renderer_.printUserPrompt();

                // End of input ends the session, exactly as /exit does. getline clears the
                // string and sets failbit on EOF, so without this the empty-input continue
                // below spins the loop forever against a stream that will never yield again.
                if ( !std::getline( std::cin, user_input ) )
                {
                    renderer_.endUserPrompt();
                    break;
                }

                if ( user_input.empty() )
                    continue;

                if ( user_input.starts_with( '/' ) )
                {
                    std::string_view cmd{ user_input };
                    cmd.remove_prefix( 1 );

                    if ( cmd == "exit" )
                    {
                        break;
                    }
                    if ( cmd == "clear" )
                    {
                        clearHistory();
                        renderer_.printInfo( "Conversation history cleared." );
                        continue;
                    }
                    if ( cmd == "help" )
                    {
                        printHelp();
                        continue;
                    }
                    if ( cmd == "stats" )
                    {
                        showTurnStats();
                        continue;
                    }
                    if ( cmd == "seed" || cmd.starts_with( "seed " ) )
                    {
                        if ( cmd == "seed" )
                        {
                            renderer_.printInfo( "Usage: /seed <n>  (reseed the sampler; identical seed + prompt + history => identical tokens)." );
                            continue;
                        }

                        const std::string_view arg = cmd.substr( 5 );
                        std::uint64_t seed = 0;
                        const auto result = std::from_chars( arg.data(), arg.data() + arg.size(), seed );

                        if ( result.ec != std::errc{} )
                        {
                            renderer_.printInfo( "Usage: /seed <n>  (n = non-negative integer)." );
                            continue;
                        }

                        // The sampler belongs to the model, so there is nothing to seed until
                        // one is resident.
                        if ( !modelIsResident() )
                        {
                            reportNoModel();
                            continue;
                        }

                        std::visit( [&]( auto& m ) { m->seedSampler( seed ); }, model_ );
                        renderer_.printInfo( std::format( "Sampler reseeded ({}).", seed ) );
                        continue;
                    }
                    if ( cmd == "verbose" || cmd.starts_with( "verbose " ) )
                    {
                        if ( cmd == "verbose" )
                        {
                            renderer_.printInfo( std::format(
                                "Detail: {}. Set with /verbose <off|thoughts|all>.",
                                detailLevelName( config_.detail ) ) );
                            continue;
                        }

                        const auto level = parseDetailLevel( cmd.substr( 8 ) );

                        if ( !level )
                        {
                            renderer_.printInfo( "Usage: /verbose <off|thoughts|all>." );
                            continue;
                        }

                        setDetailLevel( *level );
                        continue;
                    }
                    if ( cmd == "effort" || cmd.starts_with( "effort " ) )
                    {
                        if ( !templateHasEffortLevels() )
                        {
                            renderer_.printInfo( std::format(
                                "{}'s thinking has no effort levels: it is on or off (/thinking).", modelName() ) );
                            continue;
                        }

                        if ( cmd == "effort" )
                        {
                            const ThinkingEffort& e = thinkingEffort( config_.thinking_effort );
                            renderer_.printInfo( std::format(
                                "Thinking effort: {} ({}, ~{} tokens). Set with /effort <1-5>.",
                                config_.thinking_effort, e.name, e.budget ) );
                            continue;
                        }

                        const std::string_view arg = cmd.substr( 7 );
                        int level = 0;
                        const auto result = std::from_chars( arg.data(), arg.data() + arg.size(), level );

                        if ( result.ec != std::errc{} || level < 1 || level > 5 )
                        {
                            renderer_.printInfo( "Usage: /effort <1-5>  (1 = minimal, 3 = balanced, 5 = exhaustive)." );
                            continue;
                        }

                        config_.thinking_effort = level;
                        const ThinkingEffort& e = thinkingEffort( level );
                        renderer_.printInfo( std::format(
                            "Thinking effort set to {} ({}, ~{} tokens).", level, e.name, e.budget ) );
                        continue;
                    }

                    if ( cmd == "context" || cmd.starts_with( "context " ) )
                    {
                        const std::vector<std::string_view> args =
                            cmd == "context" ? std::vector<std::string_view>{}
                                             : splitWhitespace( cmd.substr( 8 ) );

                        if ( args.empty() )
                        {
                            reportContext();
                            continue;
                        }

                        if ( args.front() == "auto" )
                        {
                            applyAutomaticContext();
                            continue;
                        }

                        const std::string_view value = args.front();
                        std::size_t length = 0;
                        const auto result = std::from_chars(
                            value.data(), value.data() + value.size(), length );

                        if ( result.ec != std::errc{}
                            || result.ptr != value.data() + value.size() || length == 0 )
                        {
                            renderer_.printInfo(
                                "Usage: /context <n>|auto  (n = tokens; auto measures the largest "
                                "that fits this card)." );
                            continue;
                        }

                        applyContextLength( length );
                        continue;
                    }

                    if ( cmd == "set" || cmd.starts_with( "set " ) )
                    {
                        const std::vector<std::string_view> args =
                            cmd == "set" ? std::vector<std::string_view>{}
                                         : splitWhitespace( cmd.substr( 4 ) );

                        if ( args.empty() )
                        {
                            reportSamplingSettings();
                            continue;
                        }

                        if ( args.size() != 2 )
                        {
                            renderer_.printInfo(
                                "Usage: /set <temperature|top_k|top_p|presence_penalty> <value|model>." );
                            continue;
                        }

                        applySamplingSetting( args[ 0 ], args[ 1 ] );
                        continue;
                    }

                    if ( cmd == "thinking" || cmd.starts_with( "thinking " ) )
                    {
                        const std::vector<std::string_view> args =
                            cmd == "thinking" ? std::vector<std::string_view>{}
                                              : splitWhitespace( cmd.substr( 9 ) );

                        applyThinking( args );
                        continue;
                    }

                    if ( cmd == "model" || cmd.starts_with( "model " ) )
                    {
                        const std::vector<std::string_view> args =
                            cmd == "model" ? std::vector<std::string_view>{}
                                           : splitWhitespace( cmd.substr( 6 ) );

                        // A bare argument is a model NAME, not a verb, so the reserved words are a
                        // closed set and everything else describes. That is what makes /model
                        // <name> report rather than load: a load takes seconds, discards the KV
                        // cache and is spelled out, so a half-remembered name costs a lookup
                        // instead of a multi-gigabyte reload. It also means the noun form reads
                        // the same way everywhere -- /model and /model <name> are one verb applied
                        // to two objects, matching /context, /effort and /verbose.
                        //
                        // A model named `list` is unreachable. Repository names make that
                        // implausible, and the alternative is a prefix nobody would type.
                        if ( args.empty() )
                        {
                            printModelInfo();
                        }
                        else if ( args.front() == "list" )
                        {
                            listModelsCommand( std::span( args ).subspan( 1 ) );
                        }
                        else if ( args.front() == "load" )
                        {
                            loadModelCommand( std::span( args ).subspan( 1 ) );
                        }
                        else if ( args.front() == "install" )
                        {
                            installModelCommand( std::span( args ).subspan( 1 ) );
                        }
                        else if ( args.front() == "remove" )
                        {
                            removeModelCommand( std::span( args ).subspan( 1 ) );
                        }
                        else
                        {
                            describeModelCommand( args );
                        }

                        continue;
                    }

                    renderer_.printInfo( std::format(
                        "Unknown command: {}. Type /help for available commands.", user_input ) );
                    continue;
                }

                // Every generation path dereferences model_, so a turn with nothing resident is
                // a crash rather than an empty answer. The commands that fix it still work.
                if ( !modelIsResident() )
                {
                    reportNoModel();
                    continue;
                }

                history_.push_back( { MessageRole::User, user_input } );

                std::string response;
                response.reserve( 4096 );

                generateResponse( response );

                handleResponse( response );
            }
        }

    private:

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

        /**
         * @brief The one-shot answer as a JSON object on standard output.
         *
         * finish_reason is the field that earns this format: a response cut off at the token
         * cap reads as a complete answer in plain text and is self-announcing here.
         */
        void emitOneShotJson( const std::string& answer, std::ostream& answer_out ) const
        {
            int tokens = 0;
            float prefill_ms = 0.0f;

            for ( const RoundStats& round : last_turn_rounds_ )
            {
                tokens += round.tokens_generated;
                prefill_ms += round.prefill_time_ms;
            }

            nlohmann::json payload;
            payload[ "content" ] = answer;
            payload[ "model" ] = config_.model_name;
            payload[ "context_length" ] = config_.context_length;

            // The scripted half of the provenance the startup line prints. A caller that asked for
            // no context and got 83968 has the same right to know why as a reader of the banner.
            payload[ "context_source" ] = config_.context_is_automatic ? "auto" : "configured";

            // What the plan was decided against and what it traded, so a script can reproduce the
            // choice: the free memory is the input that moves between runs on a card driving a display.
            if ( config_.context_is_automatic && deployment_plan_ )
            {
                const DeploymentPlan& plan = *deployment_plan_;

                payload[ "context_measurement" ] = {
                    { "device", config_.device_index },
                    { "device_total_bytes", plan.reading().total_bytes },
                    { "device_free_bytes", plan.reading().free_bytes },
                    { "prefill_chunk_rows", plan.prefillChunkRows() },
                    { "unconstrained_chunk_rows", plan.prefillChunking().unconstrained_chunk_rows },
                    { "bounded_by_prefill", plan.contextLimitedBy() == DeploymentPlan::ContextLimit::FullPrefillChunk },
                    { "context_limit", std::string( DeploymentPlan::nameOf( plan.contextLimitedBy() ) ) } };
            }

            payload[ "tokens_generated" ] = tokens;

            // The interactive session reads this off /stats; a scripted caller measuring a prompt
            // has only wall time without it, which carries the model load and the tokenizer too.
            payload[ "prefill_ms" ] = prefill_ms;
            payload[ "rounds" ] = last_turn_rounds_.size();
            payload[ "finish_reason" ] = finishReasonName( finishStatus() );

            answer_out << payload.dump( 2 ) << '\n';
        }

        /**
         * @brief Wire names for the generation outcome, in the vocabulary a caller expects.
         */
        static const char* finishReasonName( GenerateStatus status )
        {
            switch ( status )
            {
                case GenerateStatus::MaxNewTokensReached: return "length";
                case GenerateStatus::ContextOverflow:     return "context_limit";
                default:                                  return "stop";
            }
        }

        /**
         * @brief Route a completed generation response through tool call handling.
         *
         * When ToolCallParser detects a <|python_tag|> block the response is treated
         * as a tool call: the call is dispatched, the result is pushed as a Tool-role
         * message, and generation runs again so the model sees the result and produces
         * a final answer. Plain text responses are streamed and pushed as an Assistant turn.
         *
         * A tool call that names an unregistered handler pushes an error result back to
         * the model rather than throwing, allowing the model to respond gracefully.
         *
         * @param response Raw text from the most recent generateResponse() call.
         */
        void handleResponse( const std::string& response )
        {
            // A model whose template the library renders had its tool rounds inside generateResponse(),
            // read by its own grammar; the Llama ToolCallParser below is Chat's own.
            if ( tool_handlers_.empty() || protocol_ )
            {
                emitAssistantResponse( response );
                return;
            }

            std::optional<ToolCall> tool_call;

            try
            {
                tool_call = ToolCallParser::parse( response );
            }
            catch ( const std::runtime_error& e )
            {
                renderer_.printError( std::format( "[tool call parse error: {}]", e.what() ) );
                emitAssistantResponse( response );
                return;
            }

            if ( !tool_call.has_value() )
            {
                emitAssistantResponse( response );
                return;
            }

            ChatMessage assistant_turn;
            assistant_turn.role = MessageRole::Assistant;
            assistant_turn.tool_calls.push_back( *tool_call );
            history_.push_back( std::move( assistant_turn ) );

            const std::string tool_result = dispatchTool( *tool_call );

            emitToolCall( *tool_call, tool_result );

            ChatMessage tool_turn;
            tool_turn.role = MessageRole::Tool;
            tool_turn.content = tool_result;
            history_.push_back( std::move( tool_turn ) );

            std::string final_response;
            final_response.reserve( 512 );

            generateResponse( final_response );

            emitAssistantResponse( final_response );
        }

        /**
         * @brief Set the display-detail level and sync the live log level.
         *
         * Detail All raises the console log level to Info (showing load dumps and
         * INFO logging); lower levels keep it quiet at Warning. Warns when reasoning
         * is requested while thinking mode is off, since nothing would be shown.
         */
        void setDetailLevel( DetailLevel level )
        {
            config_.detail = level;

            // "all" raises the live log level so INFO logging and load dumps appear.
            Logging::Logger::defaultLogger().setLevel( level == DetailLevel::All
                ? Logging::LogLevel::Info
                : Logging::LogLevel::Warning );

            std::string message = std::format( "Detail set to {}.", detailLevelName( level ) );

            // The reasoning channel only exists when thinking mode is active. Named from the
            // resident model rather than a literal: model names come from the store, so any
            // spelled-out example goes stale the moment the store holds something else.
            if ( level >= DetailLevel::Thoughts && !config_.show_thinking )
                message += std::format(
                    " (Thinking mode is off — enable it with /thinking on to see reasoning.)",
                    modelName() );

            renderer_.printInfo( message );
        }

        /**
         * @brief Strip tokens, split channels, render (or validate), and store an
         *        assistant turn.
         *
         * The Gemma reasoning channel is rendered only at detail level Thoughts or
         * above, and the verbatim raw output at All; the final answer is always
         * rendered and is the only text pushed to history so the reasoning never
         * re-enters the next formatted prompt. For models that emit no reasoning
         * channel the parse is a pass-through and the whole response is the answer.
         *
         * When the turn streamed live, the buffered pipeline is not re-rendered;
         * instead its output is the oracle the streamed transcript is checked
         * against (gate 1 of the streaming display).
         */
        void emitAssistantResponse( const std::string& raw )
        {
            if ( config_.detail == DetailLevel::All )
                renderer_.printRaw( raw );

            // A turn the template's grammar read round by round is shown as it read it; Llama's raw
            // text is split here, its tool spans stripped first so no call syntax reaches the answer.
            const ParsedResponse parsed = turn_display_.has_value()
                ? *turn_display_
                : ChannelParser::parse( stripSpecialTokens( stripToolExchangeSpans( raw ) ), activeChannel() );

            // One shot renders nothing: the caller gets the answer on standard output in the
            // format it asked for, and a painted block would be in the middle of it.
            if ( one_shot_ )
            {
                stream_display_.reset();
                pushAnswer( parsed );

                return;
            }

            if ( stream_display_ != nullptr && renderer_.streamHasOutput() )
            {
                validateStreamedDisplay( parsed );
            }
            else
            {
                if ( config_.detail >= DetailLevel::Thoughts && !parsed.thinking.empty() )
                    renderer_.printThinking( parsed.thinking );

                renderer_.printMilaResponse( parsed.answer );
            }

            stream_display_.reset();
            pushAnswer( parsed );
        }

        /**
         * @brief The turn's answer joins the history.
         *
         * Read through the template, it is the last round's reply with its reasoning, which the template decides
         * whether to render again; a turn whose last round dispatched calls has already joined the history as those
         * calls and their results. Llama's is the answer Chat split out of the raw text.
         */
        void pushAnswer( const ParsedResponse& parsed )
        {
            if ( !protocol_ )
            {
                history_.push_back( { MessageRole::Assistant, parsed.answer } );

                return;
            }

            if ( turn_answer_.has_value() )
            {
                history_.push_back( { MessageRole::Assistant, turn_answer_->content, {}, {}, turn_answer_->reasoning } );
            }
        }

        /**
         * @brief Gate 1 of the streaming display: the streamed transcript must
         *        equal the buffered render of the same response.
         *
         * Line-exact comparison against the shared formatRich + wordWrap pipeline
         * (built at the wrap width the stream captured); when the stream had
         * forced visual breaks (tool trace lines interleaving the block) the
         * comparison falls back to whitespace-normalized text. Any divergence is
         * a bug in the incremental path and is reported loudly.
         */
        void validateStreamedDisplay( const ParsedResponse& parsed ) const
        {
            const int width = renderer_.streamWrapWidth();

            validateStreamedChannel( "answer", renderer_.streamedAnswerLines(),
                RichText::wordWrap( RichText::formatRich( parsed.answer ), width ) );

            if ( config_.detail >= DetailLevel::Thoughts )
                validateStreamedChannel( "thinking", renderer_.streamedThinkingLines(),
                    RichText::wordWrap( RichText::formatRich( parsed.thinking ), width ) );
        }

        void validateStreamedChannel( std::string_view channel,
            const std::vector<std::string>& streamed,
            const std::vector<std::string>& oracle ) const
        {
            if ( !renderer_.streamForcedBreak() )
            {
                if ( streamed == oracle )
                    return;

                std::size_t line = 0;
                const std::size_t common = std::min( streamed.size(), oracle.size() );

                while ( line < common && streamed[ line ] == oracle[ line ] )
                    ++line;

                renderer_.printInfo( std::format(
                    "[stream validator] {} display diverged from the buffered render "
                    "(first difference at line {}; {} streamed / {} buffered lines).",
                    channel, line + 1, streamed.size(), oracle.size() ) );
                return;
            }

            if ( normalizeWhitespace( joinLines( streamed ) ) != normalizeWhitespace( joinLines( oracle ) ) )
                renderer_.printInfo( std::format(
                    "[stream validator] {} display diverged from the buffered render.",
                    channel ) );
        }

        static std::string joinLines( const std::vector<std::string>& lines )
        {
            std::string joined;

            for ( const auto& line : lines )
            {
                joined += line;
                joined += '\n';
            }

            return joined;
        }

        /// Collapse whitespace runs to single spaces and trim the ends, so text
        /// comparison survives the extra line breaks a forced suspend introduces.
        static std::string normalizeWhitespace( std::string_view text )
        {
            std::string out;
            out.reserve( text.size() );
            bool in_whitespace = true;

            for ( const char c : text )
            {
                if ( c == ' ' || c == '\t' || c == '\n' || c == '\r' )
                {
                    in_whitespace = true;
                    continue;
                }

                if ( in_whitespace && !out.empty() )
                    out += ' ';

                in_whitespace = false;
                out += c;
            }

            return out;
        }

        /**
         * @brief The markers the resident family wraps its reasoning in.
         *
         * A family with no reasoning channel is given Gemma's, which finds nothing in its
         * output and returns the whole response as the answer -- the pass-through the parser
         * already contracts for. Naming a "none" pair would be a third case behaving as the
         * first.
         *
         * Qwen's pair carries the session's thinking state, because for that family the state
         * changes what a response MEANS: the primer opens the reasoning span when thinking is
         * on, so text before a close marker is reasoning, and when it is off the very same text
         * with no markers at all is the answer. The parser cannot tell those apart on its own.
         */
        ChannelDelimiters activeChannel() const
        {
            if ( config_.model_type != ModelType::Qwen )
            {
                return kGemmaChannel;
            }

            ChannelDelimiters channel = kQwenThinkSpan;
            channel.primer_opens_channel = config_.show_thinking;

            return channel;
        }

        /**
         * @brief Remove tool call and tool response spans from a response before it is
         *        channel-split and displayed.
         *
         * These are protocol-internal (the harness already consumed and dispatched them);
         * showing the raw call/response syntax as answer prose is noise. An unterminated span
         * (generation stopped mid-span) truncates to end-of-string.
         *
         * Every family's pair is stripped rather than only the resident one's: the strings do
         * not collide, and a list keyed on the model would be a second place to update when a
         * family is added.
         */
        static std::string stripToolExchangeSpans( const std::string& text )
        {
            static constexpr std::pair<std::string_view, std::string_view> kSpans[] = {
                { "<|tool_call>", "<tool_call|>" },
                { "<|tool_response>", "<tool_response|>" },
                // Qwen 3.8.
                { "<tool_call>", "</tool_call>" },
                { "<tool_response>", "</tool_response>" },
            };

            std::string result = text;

            for ( const auto& [open, close] : kSpans )
            {
                std::string::size_type pos;

                while ( (pos = result.find( open )) != std::string::npos )
                {
                    const auto close_pos = result.find( close, pos + open.size() );

                    if ( close_pos == std::string::npos )
                    {
                        result.erase( pos );
                        break;
                    }

                    result.erase( pos, close_pos + close.size() - pos );
                }
            }

            return result;
        }

        /**
         * @brief Remove control tokens from a generated response before storing it in the
         *        conversation history.
         *
         * The streaming decoder may include a turn terminator at the tail of the generated
         * text. Storing these verbatim causes them to be re-emitted literally into the next
         * formatted prompt, corrupting the token boundary structure and confusing the model
         * on subsequent turns.
         *
         * Every family's tokens are listed, not the resident one's: the spellings do not
         * collide, and one list is one place to update.
         */
        static std::string stripSpecialTokens( const std::string& text )
        {
            static constexpr std::string_view kTokens[] = {
                "<|eot_id|>", "<|eom_id|>", "<|python_tag|>",
                "<|begin_of_text|>", "<|end_of_text|>",
                // Gemma 4 instruct special tokens.
                "<end_of_turn>", "<start_of_turn>", "<bos>", "<eos>", "<pad>",
                // Gemma 4 control tokens. The <|channel>/<channel|> markers are
                // deliberately omitted: ChannelParser consumes them before this runs.
                "<|turn>", "<turn|>", "<|think|>",
                "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
                "<|tool_response>", "<tool_response|>",
                "<|image>", "<image|>", "<|image|>",
                "<|audio>", "<audio|>", "<|audio|>", "<|video|>",
                // Qwen 3.8 ChatML and tool markers. <think>/</think> are deliberately omitted
                // for the same reason Gemma's channel pair is: ChannelParser consumes them
                // before this runs, and removing them here would erase the boundary it splits on.
                "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>"
            };

            std::string result = text;

            for ( const auto token : kTokens )
            {
                std::string::size_type pos = 0;

                while ( (pos = result.find( token, pos )) != std::string::npos )
                    result.erase( pos, token.size() );
            }

            // Trim trailing whitespace left after token removal.
            const auto last = result.find_last_not_of( " \t\n\r" );

            if ( last != std::string::npos )
                result.erase( last + 1 );
            else
                result.clear();

            return result;
        }

        /**
         * @brief Invoke the registered handler for a tool call and return its result.
         *
         * Returns an error string when no handler is registered for the call name so
         * the model receives feedback rather than silently receiving an empty result.
         *
         * @param call Parsed tool call from ToolCallParser::parse().
         * @return     Handler result string, or a formatted error on missing handler.
         */
        std::string dispatchTool( const ToolCall& call )
        {
            const auto it = tool_handlers_.find( call.name );

            if ( it == tool_handlers_.end() )
            {
                const std::string error = std::format(
                    "Error: no handler registered for tool '{}'", call.name );
                std::cerr << "\n[" << error << "]\n";
                return error;
            }

            try
            {
                return it->second( call.arguments );
            }
            catch ( const std::exception& e )
            {
                const std::string error = std::format(
                    "Error: tool '{}' handler threw: {}", call.name, e.what() );
                std::cerr << "\n[" << error << "]\n";
                return error;
            }
        }

        /**
         * @brief Render a dispatched tool call as an inline agentic-trace line.
         *
         * The human-readable intent (from the tool's summary template) always shows; the
         * raw result payload only at detail level All. Shared by the Gemma and Llama paths.
         */
        void emitToolCall( const ToolCall& call, const std::string& result )
        {
            // ANSI bold on/off around each substituted argument value so the dynamic parts
            // stand out within the trace line; bold-off (not full reset) preserves the line
            // color the renderer applies around the whole summary.
            const std::string summary = formatToolSummary(
                system_prompt_config_.tools, call.name, call.arguments, "\x1b[1m", "\x1b[22m" );

            renderer_.printToolCall( summary, result, config_.detail == DetailLevel::All );
        }

        /**
         * @brief Serialize tool definitions to a JSON array string for inclusion in
         *        the system prompt so the model knows which tools are available.
         *
         * @param tools Tool definitions loaded from the system prompt file.
         * @return      JSON array string describing all tools.
         */
        /**
         * @brief The tools worth telling the model about: those with a registered handler.
         *
         * Describing an unhandled tool primes the model to emit a call it will never get a
         * result for. Read from two places -- the tools a template renders (promptOptions())
         * and Llama's system turn -- so the filter lives here rather than in whichever of them
         * happens to run first.
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

        static std::string serializeTools( const std::vector<ToolDefinition>& tools )
        {
            nlohmann::json arr = nlohmann::json::array();

            for ( const auto& tool : tools )
            {
                nlohmann::json props = nlohmann::json::object();

                for ( const auto& [pname, prop] : tool.parameters.properties )
                {
                    props[ pname ] = {
                        { "type", prop.type },
                        { "description", prop.description }
                    };
                }

                arr.push_back( {
                    { "name", tool.name },
                    { "description", tool.description },
                    { "parameters", {
                        { "type", tool.parameters.type },
                        { "properties", props },
                        { "required", tool.parameters.required }
                    }}
                    } );
            }

            return arr.dump( 2 );
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
                if ( stream_display_ )
                    stream_display_->suspendForTrace();

                renderer_.printError( std::format( "[tool call not understood: {}]", call ) );
            }
        }

        void generateResponse( std::string& response )
        {
            turn_display_.reset();
            turn_answer_.reset();

            Mila::Dnn::Conversation::Prompt prompt = renderPrompt();
            std::vector<std::int32_t> input_tokens = encodePrompt( prompt );

            const std::size_t budget = generationBudget( input_tokens.size() );

            if ( budget == 0 )
            {
                renderer_.printInfo( std::format(
                    "The prompt is {} tokens and the context is {}, so there is no room to "
                    "generate. Clear the history or raise context_length.",
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

            last_turn_rounds_.clear();

            // Streaming display (Gemma-first): tokens render live while the buffered
            // response string keeps accumulating as history and as the display's
            // validation oracle (see emitAssistantResponse). One pipeline spans all
            // tool rounds of the turn.
            stream_display_.reset();

            // Not in one shot: streaming paints tokens onto standard output as they arrive,
            // which is the whole value at a prompt and pure corruption of a piped answer.
            if ( config_.streaming_capable && gemma_stream_tokens_.has_value() && !one_shot_ )
                stream_display_ = std::make_unique<StreamingResponseDisplay>(
                    renderer_, *gemma_stream_tokens_, config_.detail );

            for ( int round = 0; round < kMaxToolRounds; ++round )
            {
                // Live token count on the spinner line: ticking = generating, frozen = hung.
                std::atomic<int> live_token_count{ 0 };

                renderer_.startSpinner( {}, &live_token_count );
                stop_src_ = std::stop_source{};
                const std::size_t round_begin = response.size();

                // The template left this round's reply inside the reasoning channel (a resume with
                // thinking on), so the display starts there too.
                if ( stream_display_ && prompt.reply_begins_in_reasoning
                    && stream_display_->beginRoundInReasoning() == StreamingResponseDisplay::TokenAction::ThinkingStarted )
                {
                    renderer_.startSpinner( "Thinking...", &live_token_count );
                }

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
                                live_token_count.fetch_add( 1, std::memory_order_relaxed );

                                const std::string piece = tokenizer_->decode(
                                    std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                                response += piece;

                                if ( stream_display_ )
                                {
                                    const auto action = stream_display_->onToken( tok, piece );

                                    // The display suspended for the suppressed tool span:
                                    // the spinner owns the line until the round ends.
                                    if ( action == StreamingResponseDisplay::TokenAction::ToolCallOpened )
                                        renderer_.startSpinner( "tool call", &live_token_count );
                                    // Hidden reasoning (detail off): label the spinner so the
                                    // user knows Mila is thinking. The first answer word stops it.
                                    else if ( action == StreamingResponseDisplay::TokenAction::ThinkingStarted )
                                        renderer_.startSpinner( "Thinking...", &live_token_count );
                                }
                            },
                            gen_params,
                            stop_src_.get_token() );
                    },
                    model_ );

                renderer_.stopSpinner();

                // Round boundary: surrender any incomplete UTF-8 tail to the display.
                if ( stream_display_ )
                    stream_display_->onRoundEnd();

                RoundStats round_stats;
                round_stats.finish_status = round_status;
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

                // A capped round is indistinguishable from a hang while the buffered
                // response builds (pegged GPU, silent spinner) -- say so explicitly.
                // One shot says it on standard error instead, or in finish_reason -- this line
                // goes to standard output and would land inside the answer being piped.
                if ( ( round_status == GenerateStatus::MaxNewTokensReached
                    || round_status == GenerateStatus::ContextOverflow ) && !one_shot_ )
                {
                    if ( stream_display_ )
                        stream_display_->suspendForTrace();

                    renderer_.printInfo( round_status == GenerateStatus::MaxNewTokensReached
                        ? std::format(
                            "Response hit the {}-token cap without a stop token (finish: length).",
                            budget )
                        : "Response stopped at the context limit (finish: context_limit)." );
                }

                // Llama's reply is read after the turn, by handleResponse().
                if ( !protocol_ )
                {
                    last_turn_rounds_.push_back( std::move( round_stats ) );
                    break;
                }

                const Mila::Dnn::Conversation::Reply reply =
                    protocol_->parseReply( std::string_view( response ).substr( round_begin ), prompt );

                appendToTurnDisplay( reply );
                reportMalformedCalls( reply );

                // A round cut short by the budget or the context is not dispatched: a call it ends in
                // is not one the model finished.
                if ( reply.tool_calls.empty() || tool_handlers_.empty() || round_status != GenerateStatus::Success )
                {
                    turn_answer_ = reply;
                    last_turn_rounds_.push_back( std::move( round_stats ) );
                    break;
                }

                round_stats.ended_in_tool_call = true;

                for ( const auto& call : reply.tool_calls )
                {
                    round_stats.tool_name += round_stats.tool_name.empty() ? "" : ", ";
                    round_stats.tool_name += call.name;
                }

                last_turn_rounds_.push_back( std::move( round_stats ) );

                // The trace lines print between streamed blocks; make sure the
                // display's visual line is closed first.
                if ( stream_display_ )
                    stream_display_->suspendForTrace();

                history_.push_back( { MessageRole::Assistant, reply.content, reply.tool_calls, {}, reply.reasoning } );

                for ( const auto& call : reply.tool_calls )
                {
                    const std::string tool_result = dispatchTool( call );

                    emitToolCall( call, tool_result );

                    history_.push_back( { MessageRole::Tool, tool_result, {}, call.name } );
                    response += protocol_->formatToolResult( call.name, tool_result );
                }

                prompt = renderPrompt();
                input_tokens = encodePrompt( prompt );
            }

            if ( stream_display_ )
                stream_display_->finish();
        }

        /**
         * @brief Build the token sequence for the current generation step.
         *
         * Each family renders the full structured history into its own instruct template. There
         * is no base-model branch: `resolveModel` refuses a non-instruct model, so history is
         * always something the model was trained to read.
         *
         * @return Token ids ready to pass to generateAsync().
         */
        std::vector<std::int32_t> buildInputTokens() const
        {
            return encodePrompt( renderPrompt() );
        }

        std::vector<std::int32_t> encodePrompt( const Mila::Dnn::Conversation::Prompt& prompt ) const
        {
            const auto token_ids = tokenizer_->encode( prompt.text );

            return std::vector<std::int32_t>( token_ids.begin(), token_ids.end() );
        }

        /// The history as the model's template renders it, primed for the reply.
        Mila::Dnn::Conversation::Prompt renderPrompt() const
        {
            if ( !protocol_ )
            {
                return { MessageFormatter::format( history_ ), false };
            }

            return protocol_->formatPrompt( history_, promptOptions() );
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

        /// False where the model's template has a thinking switch and no levels; Llama's, which Chat renders, has neither.
        bool templateHasEffortLevels() const
        {
            if ( !protocol_ )
            {
                return true;
            }

            return std::ranges::any_of( protocol_->templateVariables(),
                []( const auto& variable ) { return variable.name == "reasoning_effort"; } );
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

        /**
         * @brief A point on the thinking-effort scale: its name and the reasoning tokens a context should leave room for.
         *
         * The model reads the level only where its template has levels: Qwen 3.8's reasoning_effort. Gemma 4's
         * thinking is a switch, so its prompt carries no effort at all.
         */
        struct ThinkingEffort
        {
            int              budget;
            std::string_view name;
        };

        static constexpr ThinkingEffort kThinkingEfforts[ 5 ] = {
            { 128,  "minimal" },
            { 256,  "low" },
            { 512,  "balanced" },
            { 1024, "high" },
            { 2048, "exhaustive" },
        };

        /// Resolve a 1..5 effort level (clamped) to its scale entry.
        static const ThinkingEffort& thinkingEffort( int level )
        {
            return kThinkingEfforts[ std::clamp( level, 1, 5 ) - 1 ];
        }

        static std::vector<std::string_view> splitWhitespace( std::string_view text )
        {
            std::vector<std::string_view> tokens;

            std::size_t i = 0;

            while ( i < text.size() )
            {
                while ( i < text.size() && text[ i ] == ' ' )
                    ++i;

                if ( i >= text.size() )
                    break;

                const std::size_t start = i;

                while ( i < text.size() && text[ i ] != ' ' )
                    ++i;

                tokens.push_back( text.substr( start, i - start ) );
            }

            return tokens;
        }

        /**
         * @brief True when the named model is the one already resident in memory, so a
         *        switch would be a no-op reload.
         */
        bool isCurrentModel(
            const std::string& name, std::optional<QuantizationMode> requested ) const
        {
            const bool loaded = !std::visit( []( const auto& m ) { return m == nullptr; }, model_ );

            // The name identifies the weights exactly -- it is unique across the store, which
            // the family/size/precision triple never was. Quantization still counts, because
            // one set of weights can be deployed at more than one.
            return loaded
                && config_.model_name == name
                && ( !requested.has_value() || *requested == config_.quantization_mode );
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
            config_.streaming_capable = resolved.streaming_capable;

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

            renderer_.printInfo( had_model
                ? "Model switched. Conversation history cleared."
                : "Model loaded." );

            // Said on a switch as well as at startup: under auto the context is re-measured for
            // the model just loaded, and between two families that is a change of two orders of
            // magnitude. Silent would be the same unaccountable number this replaced.
            if ( config_.context_is_automatic )
            {
                renderer_.printInfo( std::format( "Context {} (auto).", config_.context_length ) );
            }
        }

        /**
         * @brief `/model list [--online]`.
         *
         * Installed is the default because it is the offline, instant answer, and because it is
         * the only one that says what can actually be loaded. The owner is hidden: --online means
         * the one Mila publishes into. An explicit owner still works, as the escape hatch for a
         * second publisher, but nothing advertises it.
         */
        void listModelsCommand( std::span<const std::string_view> args )
        {
            try
            {
                const auto hub_owner = [&]() -> std::string
                    {
                        const std::string argument( args.front() );

                        return ( argument == "--online" || argument == "online" )
                            ? std::string( Mila::Distribution::kDefaultHubOwner )
                            : argument;
                    };

                // Against the card's TOTAL memory rather than what is free.
                //
                // The listing answers "which of these could this machine run", and that is a
                // property of the card, not of this second. Free memory was tried and is wrong
                // here twice over: the resident model's own report understates what releasing it
                // returns (it excludes the 6-13% residual Gate B measured), and whatever the
                // desktop holds gets charged to every candidate. Both push the same way, and a 3B
                // was marked as not fitting on a card with room for three. The live picture
                // belongs on /model, which measures it directly.
                const DeviceMemoryInfo memory = queryDeviceMemory( config_.device_index );

                std::optional<FootprintBudget> budget;

                // No device to ask means no column: a listing claiming "0 MB" would be stating a
                // measurement it does not have.
                if ( memory.total_bytes > 0 )
                {
                    FootprintBudget costed;

                    // Zero under auto, which asks the listing to answer each row at the largest
                    // context THAT model would get. Pricing every row at the resident model's
                    // auto-derived number is what made three of six rows warn falsely: Gemma
                    // affords 56320 because most of its layers are sliding-window, and no Llama
                    // would ever be given it. A context the user NAMED is different -- it really
                    // does apply to every row, because it is what loading any of them would use.
                    costed.fixed_context_length = config_.context_is_automatic
                        ? dim_t{ 0 }
                        : static_cast<dim_t>( config_.context_length );

                    costed.available_bytes = memory.total_bytes;
                    costed.device_name = queryDeviceName( config_.device_index );
                    costed.device_index = config_.device_index;
                    costed.resident_model =
                        modelIsResident() ? modelName() : std::string{};

                    budget = std::move( costed );
                }

                if ( args.empty() )
                {
                    const ModelListing listing = describeInstalledModels( budget );

                    // Plain, like /help and /model: a table is the content the command was run
                    // to produce.
                    for ( const auto& line : listing.table )
                    {
                        std::cout << line << "\n";
                    }

                    std::cout << "\n";

                    for ( const auto& line : listing.notes )
                    {
                        renderer_.printInfo( "  " + line );
                    }

                    return;
                }

                // Capacity, matching the installed listing: the question is what this card can
                // run, not what is free while a model is resident.
                for ( const auto& line : describeHubModels(
                    hub_owner(), memory.total_bytes,
                    queryDeviceName( config_.device_index ) ) )
                {
                    std::cout << line << "\n";
                }
            }
            catch ( const std::exception& error )
            {
                renderer_.printError( std::format(
                    "Could not list models: {}", error.what() ) );
            }
        }

        /**
         * @brief `/model load <name> [none|fp8|fp4|q4_0]`.
         *
         * Spelled out rather than reached by a bare name, because it is the one command here that
         * costs seconds and megabytes: it releases the resident weights, builds a new graph and
         * clears the conversation.
         */
        void loadModelCommand( std::span<const std::string_view> args )
        {
            if ( args.empty() )
            {
                renderer_.printInfo(
                    "Usage: /model load <name> [none|fp8|fp4|q4_0]." );

                return;
            }

            // Quantizing on load is a deployment choice, not an identity: it lets BF16 weights
            // too large for the card run anyway. Pre-quantized weights are a different model
            // with its own name, and refuse this.
            std::optional<QuantizationMode> requested_quantization;

            for ( std::size_t index = 1; index < args.size(); ++index )
            {
                const auto parsed = parseQuantization( args[ index ] );

                if ( !parsed )
                {
                    renderer_.printInfo( std::format(
                        "Unknown option '{}'. Use none, fp8, fp4 or q4_0.", args[ index ] ) );

                    return;
                }

                requested_quantization = *parsed;
            }

            // Folded before the comparison below, so `/model load LLAMA-3.2-3B-INSTRUCT-FP4`
            // against that same model resident is recognised rather than reloaded.
            const std::string name =
                InstalledModel::find( args.front() ).transform( []( const InstalledModel& model ) { return model.name(); } )
                    .value_or( std::string( args.front() ) );

            if ( isCurrentModel( name, requested_quantization ) )
            {
                renderer_.printInfo( std::format( "{} is already loaded.", modelName() ) );

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
                renderer_.printInfo( error.what() );
            }
        }

        /// `/model install <name>`.
        void installModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model install <name> -- one name, as /model list --online shows it." );

                return;
            }

            const std::string name( args.front() );

            bool installed = false;

            try
            {
                for ( const auto& line : installModel( name ) )
                {
                    renderer_.printInfo( line );
                }

                installed = true;
            }
            catch ( const std::exception& error )
            {
                // A failed install must leave the session on its working model, so this reports
                // and returns to the prompt rather than propagating.
                renderer_.printInfo( std::format( "Install failed: {}", error.what() ) );
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
                    renderer_.printInfo( std::format(
                        "Installed, but loading it failed: {}", error.what() ) );
                }
            }
        }

        /// `/model remove <name>`.
        void removeModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model remove <name> -- one name, as /model list shows it." );

                return;
            }

            try
            {
                for ( const auto& line : removeModel( std::string( args.front() ) ) )
                {
                    renderer_.printInfo( line );
                }
            }
            catch ( const std::exception& error )
            {
                renderer_.printInfo( std::format( "Remove failed: {}", error.what() ) );
            }
        }

        /// `/model <name>` -- the facts about one model, installed or only published.
        void describeModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model <name>, or /model list|load|install|remove." );

                return;
            }

            const std::string name =
                InstalledModel::find( args.front() ).transform( []( const InstalledModel& model ) { return model.name(); } )
                    .value_or( std::string( args.front() ) );

            // The resident model answers from the session instead, because the live deployment --
            // the context it was built at, what it is actually holding -- is strictly more than
            // the record can say, and it is the same model either way.
            if ( isCurrentModel( name, std::nullopt ) )
            {
                printModelInfo();

                return;
            }

            try
            {
                for ( const auto& line : describeModel( name ) )
                {
                    std::cout << line << "\n";
                }
            }
            catch ( const std::exception& error )
            {
                renderer_.printError( std::format(
                    "Could not describe {}: {}", name, error.what() ) );
            }
        }

        /**
         * @brief `/thinking [on|off]`.
         *
         * Its own command rather than an argument to a load. It rode on `/model <name> thinking`,
         * which needed a no-reload fast path to avoid a multi-gigabyte round trip for a boolean --
         * and once loading became explicit, `/model load <name> thinking` would have been a load
         * that deliberately does not load.
         */
        void applyThinking( std::span<const std::string_view> args )
        {
            // A preference cannot give a model a channel it was not trained with, and reporting
            // one as enabled is how the session came to advertise an effort level for Llama.
            if ( !config_.thinking_capable )
            {
                renderer_.printInfo( "This model has no reasoning channel." );

                return;
            }

            if ( args.empty() )
            {
                renderer_.printInfo( std::format(
                    "Thinking: {}.", config_.show_thinking ? "on" : "off" ) );

                return;
            }

            if ( args.size() != 1 || ( args.front() != "on" && args.front() != "off" ) )
            {
                renderer_.printInfo( "Usage: /thinking [on|off]." );

                return;
            }

            config_.show_thinking = ( args.front() == "on" );

            renderer_.printInfo( std::format(
                "Thinking {}.", config_.show_thinking ? "on" : "off" ) );
        }

        /// Tokens an answer needs after the transcript and the reasoning budget. A policy number,
        /// and the only compiled part of the context floor -- the rest the session measures.
        static constexpr std::size_t kAnswerHeadroomTokens = 512;

        /**
         * @brief The smallest context this session could hold a turn in, and what makes it up.
         *
         * The parts are carried, not just the sum, because a refusal has to be arguable: "8192 is
         * below the minimum" invites the question this answers.
         */
        struct ContextFloor
        {
            /// Zero when there was no tokenizer to measure with, which reads as "no floor".
            std::size_t minimum{ 0 };

            std::size_t transcript{ 0 };

            /// Zero unless thinking is both switched on and something this model can do.
            std::size_t reasoning{ 0 };
        };

        /**
         * @brief Measure the floor.
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

            // The RENDERED transcript, so the system prompt and the template's own control tokens
            // are both counted -- the two a hand-written floor would have forgotten.
            floor.transcript = buildInputTokens().size();

            floor.reasoning = ( config_.show_thinking && config_.thinking_capable )
                ? static_cast<std::size_t>( thinkingEffort( config_.thinking_effort ).budget )
                : 0;

            floor.minimum = floor.transcript + floor.reasoning + kAnswerHeadroomTokens;

            return floor;
        }

        /**
         * @brief Report the context, where it came from, and what auto would choose now.
         *
         * The last of those is what makes the command self-teaching: the number is on screen
         * before the user commits to it, so /context auto is never a leap in the dark.
         */
        void reportContext()
        {
            const std::string basis = config_.context_is_automatic
                ? std::string( "auto" )
                : ( config_.context_origin.empty()
                    ? std::string( layerName( SettingsLayer::FamilyInvariants ) )
                    : config_.context_origin );

            std::cout << std::format( "  {:<16}{}\n", "Context window:", config_.context_length );
            std::cout << std::format( "  {:<16}{}\n", "Set by:", basis );
            std::cout << std::format( "  {:<16}{}\n", "KV cache:", kvCacheName( config_.kv_cache ) );

            const ContextFloor floor = contextFloor();

            if ( floor.minimum > 0 )
            {
                std::cout << std::format( "  {:<16}{} (transcript {}, reasoning {}, answer {})\n",
                    "Minimum here:", floor.minimum, floor.transcript, floor.reasoning,
                    kAnswerHeadroomTokens );
            }

            if ( !modelIsResident() )
            {
                return;
            }

            // From the plan this model's own load ran, not a fresh one: a plan made with the model
            // resident would describe a card that still holds it.
            if ( !deployment_plan_
                || deployment_plan_->contextLimitedBy() == DeploymentPlan::ContextLimit::FixedByCaller )
            {
                std::cout << std::format( "  {:<16}not measured (set /context auto to measure it)\n",
                    "Largest fit:" );

                return;
            }

            // Why it stopped short is worth one clause here, where the reader asked about context
            // specifically -- it is the one place the prefill bound is not noise.
            std::cout << std::format( "  {:<16}{}{}\n",
                "Largest fit:", deployment_plan_->contextLength(),
                deployment_plan_->contextLimitedBy() == DeploymentPlan::ContextLimit::FullPrefillChunk
                    ? "  (held back to keep a full prefill chunk)" : "" );
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
        void applyContextLength( std::size_t length )
        {
            const FamilyTraits traits = familyTraits( config_.model_type );

            if ( modelIsResident() && length > traits.max_context )
            {
                renderer_.printError( std::format(
                    "{} addresses {} positions at most.", modelName(), traits.max_context ) );

                return;
            }

            const ContextFloor floor = contextFloor();

            if ( floor.minimum > 0 && length < floor.minimum )
            {
                renderer_.printError( std::format(
                    "{} is below this session's minimum of {}: the transcript renders to {} "
                    "tokens{}, and an answer needs room after it.",
                    length, floor.minimum, floor.transcript,
                    floor.reasoning > 0
                        ? std::format( " with {} budgeted for reasoning", floor.reasoning )
                        : std::string{} ) );

                renderer_.printInfo(
                    "  /clear drops the transcript; /effort lowers the reasoning budget." );

                return;
            }

            reloadAtContext( length, false );
        }

        /**
         * @brief Measure the largest context that fits this card and rebuild there.
         */
        void applyAutomaticContext()
        {
            if ( !modelIsResident() )
            {
                // Nothing to measure: auto is a question about a model's footprint, and the flag
                // is what makes the next load ask it.
                config_.context_is_automatic = true;
                config_.configured_context_length = 0;
                config_.context_origin =
                    std::string( layerName( SettingsLayer::SessionOverride ) );

                renderer_.printInfo(
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

                renderer_.printInfo( std::format(
                    "Context {} is now re-measured on each load.", config_.context_length ) );

                return;
            }
            catch ( const std::exception& error )
            {
                renderer_.printError( std::format( "Could not reload {}: {}", name, error.what() ) );
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
                renderer_.printError( std::format(
                    "Could not reload {} at context {}: {}", name, previous_length, error.what() ) );

                std::visit( []( auto& model ) { model.reset(); }, model_ );
                config_.model_name.clear();
            }
        }

        /**
         * @brief Rebuild the resident model at a new context, keeping the conversation.
         *
         * Context sizes the KV cache and the activation workspaces at build time and there is no
         * in-place resize, so this is switchModel's path with the same weights. It differs in the
         * one way the user notices: /model clears history because the tokenizer and the template
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
                    renderer_.printInfo( std::format( "Context is already {}.", length ) );

                    return;
                }

                // A change of flag alone must not rebuild: the buffers are already this size. This
                // is `/context <the number auto chose>`, which pins a measured value, and its
                // inverse -- and a multi-second reload to flip a bool would be indefensible.
                config_.context_is_automatic = automatic;
                config_.configured_context_length = automatic ? 0 : length;
                config_.context_origin =
                    std::string( layerName( SettingsLayer::SessionOverride ) );

                renderer_.printInfo( automatic
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
                renderer_.printError( std::format(
                    "Could not reload {} at context {}: {}", name, length, error.what() ) );

                std::visit( []( auto& model ) { model.reset(); }, model_ );

                config_.context_length = previous_length;
                config_.context_is_automatic = previous_automatic;
                config_.configured_context_length = previous_configured;
                config_.context_origin = previous_origin;

                // An empty name IS the no-model state, which is what the session now is. The
                // transcript is left alone: it is what the reload was protecting.
                config_.model_name.clear();

                renderer_.printInfo( std::format(
                    "Nothing is loaded. /model {} loads it again at context {}.",
                    name, previous_length ) );

                return;
            }

            renderer_.printInfo( automatic
                ? std::format( "Context {} (auto). Conversation kept.", length )
                : std::format( "Context {}. Conversation kept.", length ) );
        }

        /// The sampling the next reply draws with, and which knobs are the user's rather than the model's.
        void reportSamplingSettings() const
        {
            const Dnn::SamplingParams sampling = sessionSampling( config_ );

            const auto origin = []( bool yours ) { return yours ? "  (yours)" : ""; };

            std::cout << std::format( "  {:<18}{}{}\n", "temperature:", sampling.temperature,
                origin( config_.temperature.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "top_k:", sampling.top_k,
                origin( config_.top_k.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "top_p:", sampling.top_p,
                origin( config_.top_p.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "presence_penalty:", sampling.presence_penalty,
                origin( config_.presence_penalty.has_value() ) );

            renderer_.printInfo(
                "  Set with /set <key> <value>, or /set <key> model for the model's own. temperature 0 is "
                "greedy, top_k 0 and top_p 1 each disable that filter." );
        }

        /**
         * @brief Set one sampling knob.
         *
         * The bounds are the sampler's rather than a matter of taste: a negative temperature or a
         * top_p outside [0,1] is not an adventurous setting, it is a value nothing downstream has
         * a meaning for. No reload -- these are read per generate call, unlike context.
         */
        void applySamplingSetting( std::string_view key, std::string_view value )
        {
            // Whole-field parses only. from_chars stops at the first character it cannot use, so
            // without the end test "0.8abc" and "40x" would both be accepted silently.
            const auto parseFloat = [value]( float& out ) -> bool
                {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), out );

                    return result.ec == std::errc{}
                        && result.ptr == value.data() + value.size();
                };

            const auto parseInt = [value]( int& out ) -> bool
                {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), out );

                    return result.ec == std::errc{}
                        && result.ptr == value.data() + value.size();
                };

            if ( value == "model" )
            {
                if ( key == "temperature" )
                {
                    config_.temperature.reset();
                }
                else if ( key == "top_k" )
                {
                    config_.top_k.reset();
                }
                else if ( key == "top_p" )
                {
                    config_.top_p.reset();
                }
                else if ( key == "presence_penalty" )
                {
                    config_.presence_penalty.reset();
                }
                else
                {
                    renderer_.printInfo( std::format(
                        "'{}' is not a setting. Use temperature, top_k, top_p or presence_penalty.", key ) );

                    return;
                }

                renderer_.printInfo( std::format( "{} follows the model.", key ) );

                return;
            }

            if ( key == "temperature" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < 0.0f || parsed > 5.0f )
                {
                    renderer_.printInfo( "Usage: /set temperature <0..5>  (0 is greedy)." );

                    return;
                }

                config_.temperature = parsed;
                renderer_.printInfo( std::format( "temperature set to {}.", parsed ) );

                return;
            }

            if ( key == "top_k" )
            {
                int parsed = 0;

                if ( !parseInt( parsed ) || parsed < 0 )
                {
                    renderer_.printInfo( "Usage: /set top_k <n>  (0 disables it)." );

                    return;
                }

                config_.top_k = parsed;
                renderer_.printInfo( std::format( "top_k set to {}.", parsed ) );

                return;
            }

            if ( key == "top_p" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < 0.0f || parsed > 1.0f )
                {
                    renderer_.printInfo( "Usage: /set top_p <0..1>  (1 disables it)." );

                    return;
                }

                config_.top_p = parsed;
                renderer_.printInfo( std::format( "top_p set to {}.", parsed ) );

                return;
            }

            if ( key == "presence_penalty" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < -2.0f || parsed > 2.0f )
                {
                    renderer_.printInfo( "Usage: /set presence_penalty <-2..2>  (0 disables it)." );

                    return;
                }

                config_.presence_penalty = parsed;
                renderer_.printInfo( std::format( "presence_penalty set to {}.", parsed ) );

                return;
            }

            renderer_.printInfo( std::format(
                "'{}' is not a setting. Use temperature, top_k, top_p or presence_penalty.", key ) );
        }

        /// True once weights are actually on the device, which the session config cannot say --
        /// it names the model that will be loaded as readily as the one that is.
        bool modelIsResident() const
        {
            return !std::visit( []( const auto& model ) { return model == nullptr; }, model_ );
        }

        /**
         * @brief Load the active tokenizer and model, under the progress spinner.
         *
         * At detail level All the spinner is skipped so the INFO logging and the
         * model / memory dumps remain readable; otherwise the multi-second weight
         * load runs under the same braille spinner used for turns.
         */
        /**
         * @brief Say what is missing and what to type, without ending the session.
         *
         * A store with nothing in it is the state every new user starts in, so this is the first
         * thing Mila says to them rather than an error. The reason from resolveModel is carried
         * through because it distinguishes an empty store from a name that is merely misspelled,
         * and those want different next commands.
         */
        /**
         * @brief Say what to type next. Carries no reason: startup already gave it.
         *
         * The reason names the configured model and lists the store, which is worth saying once.
         * Repeating it on every prompt buries the one line that changes anything -- and it goes
         * stale the moment something is installed, while this does not.
         */
        void reportNoModel() const
        {
            renderer_.printInfo(
                "No model is loaded. /model list --online shows what can be installed, "
                "/model install <name> installs one, and /model list shows what is here." );
        }

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

            if ( config_.detail == DetailLevel::All )
            {
                // No spinner: its redraws and the log lines fight for the same line.
                initializeTokenizer();
                loadModel();

                return;
            }

            // The quantization is quoted only when it was chosen at load. A pre-quantized model
            // already carries it in its name, and "gemma-4-12b-it-fp4 (fp4)" says one fact twice.
            renderer_.startSpinner( config_.quantization_applied_at_load
                ? std::format( "Loading {} ({})",
                    modelName(), quantizationName( config_.quantization_mode ) )
                : std::format( "Loading {}", modelName() ) );

            // Stopped here rather than in each caller's catch: a load that throws left the
            // spinner redrawing over the error message that explained it, and the hidden
            // cursor never came back. The one place that starts it owns stopping it.
            try
            {
                initializeTokenizer();
                loadModel();
            }
            catch ( ... )
            {
                renderer_.stopSpinner();

                throw;
            }

            renderer_.stopSpinner();
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

        void printModelInfo() const
        {
            // Without a model the fields below are defaults, not facts -- printing them would
            // report a precision and a context for weights that are not there.
            if ( config_.model_name.empty() )
            {
                reportNoModel();

                return;
            }

            const ThinkingEffort& effort = thinkingEffort( config_.thinking_effort );

            std::cout << std::format( "  Model:          {}\n", modelName() );

            // Lineage before the deployment axes: a redistributed model's attribution belongs
            // beside its identity, not below its settings.
            if ( !config_.base_model.empty() )
            {
                std::cout << std::format( "  Base model:     {}\n", config_.base_model );
            }

            if ( !config_.license.empty() )
            {
                std::cout << std::format( "  License:        {}\n", config_.license );
            }

            const std::string_view attribution = requiredAttribution( config_.license );

            if ( !attribution.empty() )
            {
                std::cout << std::format( "  Attribution:    {}\n", attribution );
            }

            // Whether it was derived, not the derivation. "auto, 11.99 GB device, held to a full
            // 1024-row prefill chunk" answers a question nobody in a chat session is asking; that
            // it was not a number they chose is the part that changes how they read it. The full
            // account still reaches a caller through the -p JSON payload's context_source.
            const std::string context_line = config_.context_is_automatic
                ? std::format( "{} (auto)", config_.context_length )
                : std::format( "{}", config_.context_length );

            std::cout << std::format(
                "  Quantization:   {}\n"
                "  Context window: {}\n"
                "  Instruct:       {}\n",
                quantizationName( config_.quantization_mode ),
                context_line,
                config_.is_instruct ? "yes" : "no" );

            // Reported only by a model that has the channel. An effort level beside a model with
            // no reasoning mode describes a budget for tokens it will never emit.
            if ( config_.thinking_capable )
            {
                std::cout << std::format( "  Thinking:       {}\n", config_.show_thinking ? "on" : "off" );

                if ( templateHasEffortLevels() )
                {
                    std::cout << std::format( "  Effort:         {} ({}, ~{} tokens)\n",
                        config_.thinking_effort, effort.name, effort.budget );
                }
            }

            std::cout << std::format( "  Detail:         {}\n",
                detailLevelName( config_.detail ) );

            // Measured off the device rather than taken from the model's own report: what is
            // actually resident includes the allocator rounding and lazily grown scratch that
            // a footprint cannot model, and that residual is the whole reason to look here.
            const DeviceMemoryInfo memory = queryDeviceMemory( config_.device_index );

            if ( memory.total_bytes > 0 )
            {
                std::cout << std::format( "  VRAM:           {}\n",
                    formatBytesOf( memory.total_bytes - memory.free_bytes, memory.total_bytes ) );
            }
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

                // The model's own chat template, chosen by the library from what the manifest declares. Llama's
                // packages do not declare the version that tells Meta's two templates apart until they are
                // republished (BACKLOG, "Llama's prompt and tool grammar live in Chat and the inference server"),
                // so Chat still renders and parses Llama itself.
                const auto& record = installed->stored().record;

                protocol_ = config_.model_type == ModelType::Llama
                    ? nullptr
                    : Mila::Dnn::chatProtocolFor( record.architecture, record.architecture_version );

                // Cache the Gemma control-token ids: the set drives the streaming display's
                // channel router. Only
                // genuine single-token vocab entries (per GemmaChatProtocol.md) are trusted;
                // a multi-token encode means the checkpoint lacks the registered special
                // token. Streaming requires all four routing ids or it stays buffered.
                gemma_stream_tokens_.reset();

                if ( config_.model_type == ModelType::Gemma )
                {
                    const auto channel_open = probeSingleTokenId( "<|channel>" );
                    const auto channel_close = probeSingleTokenId( "<channel|>" );
                    const auto tool_call_open = probeSingleTokenId( "<|tool_call>" );
                    const auto tool_call_close = probeSingleTokenId( "<tool_call|>" );

                    if ( channel_open && channel_close && tool_call_open && tool_call_close )
                    {
                        GemmaStreamTokens tokens;
                        tokens.channel_open = *channel_open;
                        tokens.channel_close = *channel_close;
                        tokens.tool_call_open = *tool_call_open;
                        tokens.tool_call_close = *tool_call_close;

                        // Control tokens with no display form: the buffered pipeline strips
                        // their text post-hoc (stripSpecialTokens); the router drops them live.
                        static constexpr std::string_view kSuppressedControlTokens[] = {
                            "<|turn>", "<turn|>", "<|think|>", "<|tool>", "<tool|>",
                            "<|tool_response>", "<tool_response|>",
                            "<end_of_turn>", "<start_of_turn>", "<bos>", "<eos>", "<pad>",
                            "<|image>", "<image|>", "<|image|>",
                            "<|audio>", "<audio|>", "<|audio|>", "<|video|>"
                        };

                        for ( const auto control : kSuppressedControlTokens )
                        {
                            if ( const auto id = probeSingleTokenId( control ) )
                                tokens.suppressed.push_back( *id );
                        }

                        gemma_stream_tokens_ = std::move( tokens );
                    }
                    else if ( config_.streaming_capable )
                    {
                        Logging::Logger::warning(
                            "Streaming display disabled: Gemma control tokens not in the vocabulary." );
                    }
                }
            }
            catch ( const std::exception& e )
            {
                Logging::Logger::error( std::format( "Failed to load tokenizer: {}", e.what() ) );

                throw;
            }
        }

        /**
         * @brief Resolve text to its vocabulary id when it encodes as a single
         *        (special) token; nullopt when the checkpoint lacks the
         *        registered special.
         */
        std::optional<std::int32_t> probeSingleTokenId( std::string_view text ) const
        {
            const auto ids = tokenizer_->encode( std::string( text ) );

            return (ids.size() == 1)
                ? std::optional<std::int32_t>( static_cast<std::int32_t>( ids[ 0 ] ) )
                : std::nullopt;
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
                            ? std::format( " /model {} fp4 needs less.", modelName() ) : std::string{} );

                case DeploymentRefusal::Reason::FixedContextDoesNotFit:
                    return std::format(
                        "context {} needs {} and this GPU has {} free. /context auto picks the longest that fits.",
                        refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::NothingAboveTheFloorFits:
                    return std::format(
                        "even context {} needs {} and this GPU has {} free.",
                        refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::DeviceDoesNotReportMemory:
                    return "this GPU does not report its memory, so no context length can be chosen for it. "
                           "Set one with /context <length>.";
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
                        std::cout << model->toString();
                        std::cout << model->getMemoryStats().toString() << "\n";
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
         * No-op when system_prompt_path is not set in config. On success,
         * system_prompt_config_ is populated and available to run().
         * File existence is validated by main() before construction so a
         * missing file here is a logic error.
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

        /**
         * @brief Show the per-round timing breakdown for the most recent turn on demand.
         *
         * A turn is one or more rounds; a tool call ends a round and starts another with its
         * own re-prefill. Each round is reported separately (prefill / decode throughput /
         * tokens, plus the tool it invoked) because a single lumped figure across rounds mixes
         * decode with re-prefill and is meaningless. A totals line closes multi-round turns.
         */
        void showTurnStats() const
        {
            if ( last_turn_rounds_.empty() )
            {
                renderer_.printInfo( "No generation yet." );
                return;
            }

            std::vector<std::string> lines;
            const bool multi_round = last_turn_rounds_.size() > 1;

            lines.push_back( std::format( "  Last turn: {} round{}",
                last_turn_rounds_.size(), multi_round ? "s" : "" ) );

            int total_tokens = 0;
            float total_prefill_ms = 0.0f;

            for ( std::size_t i = 0; i < last_turn_rounds_.size(); ++i )
            {
                const RoundStats& round = last_turn_rounds_[ i ];
                total_tokens += round.tokens_generated;
                total_prefill_ms += round.prefill_time_ms;

                std::string line = std::format(
                    "  Round {}  prefill {:.0f} ms  \xe2\x94\x82  ",
                    i + 1, round.prefill_time_ms );

                if ( round.decode_tokens_per_second > 0.0f )
                    line += std::format( "{:.1f} tok/s  \xe2\x94\x82  {} tokens",
                        round.decode_tokens_per_second, round.tokens_generated );
                else
                    line += std::format( "{} token{}",
                        round.tokens_generated, round.tokens_generated == 1 ? "" : "s" );

                if ( round.ended_in_tool_call )
                    line += std::format( "  \xe2\x94\x82  \xe2\x86\x92 {}", round.tool_name );
                else if ( round.finish_status != GenerateStatus::Success )
                    line += std::format( "  \xe2\x94\x82  ended: {}", to_string( round.finish_status ) );

                lines.push_back( std::move( line ) );

                // Inter-token gap census (see RoundStats): outlier gaps localize a
                // mid-response stall inside generate(); uniform gaps exonerate the loop.
                if ( round.tokens_generated > 1 )
                {
                    std::string gap_line = std::format(
                        "           gaps med {:.1f} ms  \xe2\x94\x82  p99 {:.1f} ms  \xe2\x94\x82  max {:.1f} ms @ token {}",
                        round.median_gap_ms, round.p99_gap_ms,
                        round.max_gap_ms, round.max_gap_token_index );

                    if ( round.stall_count > 0 )
                        gap_line += std::format( "  \xe2\x94\x82  {} stall{} ({:.0f} ms)",
                            round.stall_count, round.stall_count == 1 ? "" : "s",
                            round.stall_total_ms );

                    lines.push_back( std::move( gap_line ) );
                }
            }

            if ( multi_round )
                lines.push_back( std::format(
                    "  Total     {} tokens  \xe2\x94\x82  {:.0f} ms prefill",
                    total_tokens, total_prefill_ms ) );

            renderer_.printStatsDetail( lines );
        }

        const std::string& modelName() const { return config_.model_name; }

        void printBanner() const
        {
            renderer_.printWelcomeBox( std::format( "Mila Chat v{}", getAPIVersion().toString() ) );
        }

        /**
         * @brief What the session actually ended up with, printed after the load rather than before.
         *
         * The config names the model that *will* be loaded as readily as the one that is, so a
         * status line printed ahead of the load asserts as fact what the spinner on the next line
         * then shows to be still in progress -- and spends the model's name twice to do it.
         * Printed here it is a statement about weights that are on the device.
         */
        void printSessionStatus() const
        {
            if ( modelIsResident() )
            {
                // "(auto)" says the number was not one they chose, which changes how they read it.
                // How it was derived does not, and is not shown anywhere.
                const std::string context_display = config_.context_is_automatic
                    ? std::format( "{} (auto)", config_.context_length )
                    : std::format( "{}", config_.context_length );

                // The thinking clause is omitted entirely for a model without the channel, rather
                // than shown as "off" -- off reads as a setting the user could turn on.
                if ( config_.thinking_capable )
                {
                    const std::string thinking_display = !config_.show_thinking
                        ? "off"
                        : templateHasEffortLevels() ? std::string( thinkingEffort( config_.thinking_effort ).name ) : "on";

                    renderer_.printInfo( std::format( "  Model: {}  ·  Context window: {}  ·  Thinking: {}",
                        modelName(), context_display, thinking_display ) );
                }
                else
                {
                    renderer_.printInfo( std::format( "  Model: {}  ·  Context window: {}",
                        modelName(), context_display ) );
                }
            }

            std::cout << "  Type /help for commands, /exit to quit.\n\n";
        }

        void printHelp() const
        {
            std::cout << R"(
Available commands:
  /help                              Show this help message
  /clear                             Clear conversation history
  /model                             Show the loaded model
  /model <name>                      Show a model, installed or published
  /model list [--online]             List installed models, or what can be installed
  /model load <name> [quant]         Load a model (clears history). quant quantizes
                                     unquantized weights on load: none, fp8, fp4, q4_0.
  /model install <name>              Install a published model
  /model remove <name>               Remove an installed model and reclaim its blobs
  /context                           Show the context window and the largest that fits
  /context <n>|auto                  Set the context window (rebuilds, keeps the conversation)
  /thinking [on|off]                 Show or set the reasoning channel
  /set                               Show the sampling the next reply uses
  /set <key> <value>|model           Set temperature, top_k, top_p or presence_penalty,
                                     or return it to the model's own
  /effort [1-5]                      Show or set the thinking token-budget level
  /verbose [off|thoughts|all]        Show or set display detail (reasoning, raw + logs)
  /stats                             Show per-round timing for the last turn
  /seed <n>                          Reseed the sampler for reproducible generation
  /exit                              Exit the application

Models:         a bare /model <name> reports; loading is spelled out, because it takes
                seconds and clears the conversation.
Quantization:   a name ending -fp4/-fp8/-q4_0 is already quantized. For an unquantized model,
                /model load <name> fp4 quantizes it on load -- same weights, less
                VRAM, but the full file is still read.
Thinking:       Gemma's <|think|> mode. Toggling it does not reload weights. Effort
                (length) is set with /effort 1-5.
Sampling:       each model opens at its publisher's recommended setting, which can differ
                with thinking on and off. What you /set stays until you set it to model.
Context:        the primary VRAM lever, and it sizes buffers at build time -- so /context
                rebuilds the model, though the conversation survives. auto measures the
                largest that fits this card; a number you give is used exactly.
Detail:         tool calls always show as an agentic trace. /verbose thoughts adds the
                reasoning channel, all adds raw output + logging. The model reasons even
                when detail is off.

Examples:
)";

            // Named from the resident model rather than spelled out: a model name is whatever
            // the store holds, so a literal example stops resolving as soon as it holds
            // something else.
            std::cout << std::format( "  /model {}\n", modelName() );

            std::cout << R"(  /model list --online
  /thinking on
  /effort 5
  /context auto
  /set temperature 0.6
)" << "\n";
        }

        void clearHistory()
        {
            history_.clear();

            if ( system_prompt_config_.system_prompt.empty() )
                return;

            std::string system_content = system_prompt_config_.system_prompt;

            const std::vector<ToolDefinition> active_tools = activeTools();

            // A template the library renders places the tools itself, from promptOptions(). Only
            // Llama's, which Chat still renders, carries them in the system turn.
            if ( !active_tools.empty() && !protocol_ )
            {
                // Instruction text precedes the tool list per the Llama 3.2 zero-shot
                // tool-calling format the model was fine-tuned on.
                system_content +=
                    "\n\nIf you decide to invoke any of the function(s), you MUST put it in the "
                    "format of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
                    "func_name2(params)]\n"
                    "You SHOULD NOT include any other text in the response.\n\n"
                    "Here is a list of functions in JSON format that you can invoke:\n";

                system_content += serializeTools( active_tools );
            }

            history_.push_back( { MessageRole::System, std::move( system_content ) } );
        }

        ChatConfig config_;

        /// The last automatic-context scan, taken with nothing resident. What /context reports and
        /// what a fit suggestion reads: a scan run now would measure a card holding this model.
        // The plan the resident model was loaded with: what /context and the one-shot JSON report,
        // rather than a second derivation of it.
        std::optional<DeploymentPlan> deployment_plan_;
        ModelVariant model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage> history_;
        std::stop_source stop_src_;
        std::unordered_map<std::string, std::function<std::string( const std::string& )>> tool_handlers_;
        ConsoleRenderer renderer_;

        // The loaded model's chat template: it renders every prompt and reads every reply. Null for Llama, which
        // Chat still renders and parses itself (see initializeTokenizer()).
        std::unique_ptr<Mila::Dnn::Conversation::Protocol> protocol_;

        // The turn generateResponse() just ran, read through protocol_ round by round: every round's reasoning
        // and text, for the display, and the last round's reply when it made no call, which history keeps.
        // Empty for a model Chat parses itself.
        std::optional<ParsedResponse> turn_display_;
        std::optional<Mila::Dnn::Conversation::Reply> turn_answer_;

        // Streaming display state: control-token ids resolved at tokenizer load
        // (absent when the vocabulary lacks them — streaming falls back to the
        // buffered path), and the per-turn pipeline created by generateResponse()
        // and consumed by emitAssistantResponse().
        std::optional<GemmaStreamTokens> gemma_stream_tokens_;
        std::unique_ptr<StreamingResponseDisplay> stream_display_;

        // Per-round timing for the most recent turn, measured from the token callback
        // cadence (the library owns no stopwatch). A turn is one or more rounds: each tool
        // call splits generation into a new round with its own prefill (a full re-prefill of
        // prompt + accumulated turn text -- the harness has no incremental-KV continuation).
        // Lumping these into one prefill/decode number is meaningless, which is why the
        // per-turn stats line was dropped in favor of the on-demand /stats breakdown.
        struct RoundStats
        {
            float prefill_time_ms = 0.0f;          // round start -> first token of round
            float decode_tokens_per_second = 0.0f; // steady-state decode throughput this round
            int tokens_generated = 0;
            bool ended_in_tool_call = false;
            std::string tool_name;                 // populated when ended_in_tool_call

            // Inter-token gap census for GPU-utilization-dip triage: the callback cadence
            // is the per-token wall time, so a mid-response stall inside generate() shows
            // up as an outlier gap here. Uniform gaps + a dipping utilization graph means
            // the GPU is being time-sliced by another process, not idled by the loop.
            float median_gap_ms = 0.0f;
            float p99_gap_ms = 0.0f;
            float max_gap_ms = 0.0f;
            int max_gap_token_index = 0;           // token that ended the largest gap
            int stall_count = 0;                   // gaps > max(2.5 x median, 40 ms)
            float stall_total_ms = 0.0f;           // wall time inside those stall gaps

            // Why the round's generate() returned -- distinguishes a natural stop from
            // hitting the token cap or the context bound (a capped round looks like a
            // hang in the buffered UI: pegged GPU, no output, until the cap is reached).
            GenerateStatus finish_status = GenerateStatus::Success;
        };

        std::vector<RoundStats> last_turn_rounds_;

        // Set by runOnce: suppresses every display path, so standard output holds the answer
        // alone. Not a detail level -- detail says how much to show, this says show nothing.
        bool one_shot_{ false };
    };
}
