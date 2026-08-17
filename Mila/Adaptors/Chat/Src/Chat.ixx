/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Supports GptModel (FP32) and LlamaModel (FP32 or BF16) backends,
 * selected at construction via ChatConfig. The active model is stored
 * as a std::variant so each template instantiation retains its full
 * static type through the generate path. Llama instruct models use
 * structured ChatMessage history formatted via MessageFormatter.
 * Tool calling is supported for Llama instruct models via ToolCallParser and for Gemma 4
 * via its native <|tool_call>/<tool_response> protocol (see generateResponse() and
 * GemmaChatProtocol.md), both against registered handler functions. The Gemma path is a
 * probe grammar pending more empirical validation, not a hardened implementation.
 */

module;
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <future>
#include <stop_token>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <charconv>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <optional>
#include <span>

export module Mila.Chat;

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
    using namespace Mila::Data;

    using GptModelFP32Type   = GptModel<DeviceType::Cuda, TensorDataType::FP32>;
    using LlamaModelFP32Type = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>;
    using LlamaModelBF16Type = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using GemmaModelBF16Type = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

    using ModelVariant = std::variant<
        std::unique_ptr<GptModelFP32Type>,
        std::unique_ptr<LlamaModelFP32Type>,
        std::unique_ptr<LlamaModelBF16Type>,
        std::unique_ptr<GemmaModelBF16Type>
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
                             "or install one with 'mila install <name>'.\n";

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
                        uint64_t seed = 0;
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
                                "Usage: /set <temperature|top_k|top_p> <value>." );
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

            for ( const RoundStats& round : last_turn_rounds_ )
            {
                tokens += round.tokens_generated;
            }

            nlohmann::json payload;
            payload[ "content" ] = answer;
            payload[ "model" ] = config_.model_name;
            payload[ "context_length" ] = config_.context_length;

            // The scripted half of the provenance the startup line prints. A caller that asked for
            // no context and got 83968 has the same right to know why as a reader of the banner.
            payload[ "context_source" ] = config_.context_is_automatic ? "auto" : "configured";
            payload[ "tokens_generated" ] = tokens;
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
            // Gemma's tool round trip (stop at <tool_call|>, dispatch, splice
            // <|tool_response>, resume) already ran inside generateResponse() -- the Llama
            // ToolCallParser grammar below does not apply and would misparse the leftover
            // <|tool_call>/<|tool_response> braces in the accumulated text.
            if ( tool_handlers_.empty() || config_.model_type == ModelType::Gemma )
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
            tool_turn.tool_call_id = tool_call->id;
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

            // Gemma tool exchanges (if any ran) leave <|tool_call>/<|tool_response> spans
            // embedded mid-turn; strip them before channel-splitting so they don't leak into
            // the displayed answer as raw call syntax.
            const std::string without_tool_spans = stripToolExchangeSpans( raw );
            const std::string clean = stripSpecialTokens( without_tool_spans );
            const ParsedResponse parsed = ChannelParser::parse( clean );

            // One shot renders nothing: the caller gets the answer on standard output in the
            // format it asked for, and a painted block would be in the middle of it.
            if ( one_shot_ )
            {
                stream_display_.reset();
                history_.push_back( { MessageRole::Assistant, parsed.answer } );

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
            history_.push_back( { MessageRole::Assistant, parsed.answer } );
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

                size_t line = 0;
                const size_t common = std::min( streamed.size(), oracle.size() );

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
         * @brief Remove <|tool_call>...<tool_call|> and <|tool_response>...<tool_response|>
         *        spans from a Gemma response before it is channel-split and displayed.
         *
         * These are protocol-internal (the harness already consumed and dispatched them in
         * generateResponse()); showing the raw call/response syntax as answer prose is noise.
         * An unterminated span (generation stopped mid-span) truncates to end-of-string.
         */
        static std::string stripToolExchangeSpans( const std::string& text )
        {
            static constexpr std::pair<std::string_view, std::string_view> kSpans[] = {
                { "<|tool_call>", "<tool_call|>" },
                { "<|tool_response>", "<tool_response|>" },
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
         * @brief Remove Llama special tokens from a generated response before
         *        storing it in the conversation history.
         *
         * The streaming decoder may include <|eot_id|> or <|eom_id|> at the tail
         * of the generated text. Storing these verbatim causes them to be re-emitted
         * literally into the next formatted prompt, corrupting the token boundary
         * structure and confusing the model on subsequent turns.
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
                "<|image|>", "<|audio|>"
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

        void generateResponse( std::string& response )
        {
            std::vector<int32_t> input_tokens = buildInputTokens();

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
            gen_params.sampling.temperature = config_.temperature;
            gen_params.sampling.top_k = config_.top_k;
            gen_params.sampling.top_p = config_.top_p;

            // Gemma 4's <|tool_call>/<tool_call|> pair is a native protocol element, not a
            // text convention (GemmaChatProtocol.md): the model expects the harness to stop
            // generation right after <tool_call|>, execute the real tool, and splice a genuine
            // <|tool_response>...<tool_response|> back in before it continues. Left alone, it
            // free-runs past its own tool call and fabricates the rest (confirmed empirically --
            // see the discovery session this loop was built to validate).
            const bool watch_gemma_tool_call =
                config_.model_type == ModelType::Gemma
                && !tool_handlers_.empty()
                && gemma_tool_call_close_token_.has_value();

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
                bool tool_call_stop = false;

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
                            [&]( int32_t tok )
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

                                if ( watch_gemma_tool_call && tok == *gemma_tool_call_close_token_ )
                                {
                                    tool_call_stop = true;
                                    stop_src_.request_stop();
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
                    const size_t mid = ordered.size() / 2;
                    std::nth_element( ordered.begin(), ordered.begin() + mid, ordered.end() );
                    round_stats.median_gap_ms = ordered[ mid ];

                    const size_t p99_index = ( ordered.size() * 99 ) / 100;
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

                std::optional<ToolCall> call;

                if ( tool_call_stop )
                {
                    if ( auto parsed = Mila::Dnn::Gemma::parseToolCall( response ) )
                        call = ToolCall{ .name = std::move( parsed->name ),
                            .arguments = std::move( parsed->arguments ) };
                }

                if ( !call.has_value() )
                {
                    // Either a normal finish or an unparseable stop -- surface the text as-is
                    // rather than looping forever on a call we cannot dispatch.
                    last_turn_rounds_.push_back( std::move( round_stats ) );
                    break;
                }

                round_stats.ended_in_tool_call = true;
                round_stats.tool_name = call->name;
                last_turn_rounds_.push_back( std::move( round_stats ) );

                const std::string tool_result = dispatchTool( *call );

                // The trace lines print between streamed blocks; make sure the
                // display's visual line is closed first.
                if ( stream_display_ )
                    stream_display_->suspendForTrace();

                emitToolCall( *call, tool_result );

                response += Mila::Dnn::Gemma::formatToolResponse( call->name, tool_result );

                // Continue the SAME assistant turn: the model's protocol splices the tool
                // response into the turn it already opened, not a fresh history entry (unlike
                // Llama's ipython-role round trip). Re-prefill prompt + everything generated
                // so far -- the harness has no incremental-KV-cache continuation entry point.
                input_tokens = buildInputTokens();
                const auto continuation = tokenizer_->encode( response );
                input_tokens.insert( input_tokens.end(), continuation.begin(), continuation.end() );
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
        std::vector<int32_t> buildInputTokens() const
        {
            std::string prompt;

            if ( config_.model_type == ModelType::Gemma )
                prompt = formatGemmaPrompt( history_, config_.show_thinking, config_.thinking_effort );
            else
                prompt = MessageFormatter::format( history_ );

            auto token_ids = tokenizer_->encode( prompt );

            return std::vector<int32_t>( token_ids.begin(), token_ids.end() );
        }

        /**
         * @brief Render a conversation history into the Gemma instruct chat template.
         *
         * Gemma 4 wraps each turn as <|turn>{role}\n{content}<turn|>\n with roles
         * "system", "user", and "model" (the assistant), and terminates with a
         * <|turn>model\n primer to prime generation. The Gemma tokenizer encodes
         * the control tokens as atomic specials, so they are emitted as literal
         * text here. (The checkpoint uses <|turn>/<turn|>, not the Gemma 3-style
         * <start_of_turn>/<end_of_turn>, which are absent from its vocabulary.)
         *
         * When enable_thinking is set, the <|think|> trigger leads a dedicated
         * system turn per the Gemma 4 protocol; the model then emits a populated
         * <|channel>thought...<channel|> reasoning section. The trigger must sit in
         * the system turn — folding it into the user turn does not activate thinking.
         */
        /**
         * @brief A point on the thinking-effort scale: a token budget plus the
         *        instruction that communicates it to the model.
         *
         * Gemma 4's <|think|> is a boolean toggle with no native budget argument, so
         * effort is steered through the system instruction. The budget is the
         * user-facing scale value; the instruction is what the model actually reads.
         */
        struct ThinkingEffort
        {
            int              budget;
            std::string_view name;
            std::string_view instruction;
        };

        static constexpr ThinkingEffort kThinkingEfforts[ 5 ] = {
            { 128,  "minimal",    "Think briefly before answering; keep your reasoning to a sentence or two." },
            { 256,  "low",        "Think before answering, keeping your reasoning concise." },
            { 512,  "balanced",   "Think step by step before answering." },
            { 1024, "high",       "Think carefully and thoroughly before answering, and check your reasoning." },
            { 2048, "exhaustive", "Think exhaustively before answering: explore alternatives and verify each step." },
        };

        /// Resolve a 1..5 effort level (clamped) to its scale entry.
        static const ThinkingEffort& thinkingEffort( int level )
        {
            return kThinkingEfforts[ std::clamp( level, 1, 5 ) - 1 ];
        }

        static std::string formatGemmaPrompt( const std::vector<ChatMessage>& history, bool enable_thinking, int effort_level )
        {
            constexpr std::string_view kThinkToken = "<|think|>";

            std::string prompt = "<bos>";

            // Gemma collapses to a single system instruction; the last System turn wins.
            std::string system_content;

            for ( const auto& message : history )
            {
                if ( message.role == MessageRole::System )
                    system_content = message.content;
            }

            // Emit a dedicated system turn when there is a system instruction or when
            // thinking is enabled (the <|think|> trigger must lead the system turn).
            if ( enable_thinking || !system_content.empty() )
            {
                prompt += "<|turn>system\n";

                if ( enable_thinking )
                {
                    const ThinkingEffort& effort = thinkingEffort( effort_level );
                    prompt += kThinkToken;
                    prompt += effort.instruction;
                    prompt += std::format( " Limit your internal reasoning to about {} tokens.", effort.budget );

                    if ( !system_content.empty() )
                        prompt += "\n\n";
                }

                prompt += system_content;
                prompt += "<turn|>\n";
            }

            for ( const auto& message : history )
            {
                if ( message.role == MessageRole::System )
                    continue;

                const bool is_model = (message.role == MessageRole::Assistant);
                prompt += is_model ? "<|turn>model\n" : "<|turn>user\n";
                prompt += message.content;
                prompt += "<turn|>\n";
            }

            prompt += "<|turn>model\n";

            // The 12B/26B/31B Gemma 4 sizes prime an empty <|channel>thought<channel|> onto
            // the prompt itself when thinking is off -- this suppresses "ghost" thought
            // channels the model may otherwise emit even when deactivated (vendor
            // prompt-formatting doc; the smaller E2B/E4B sizes do not need this, and are
            // not offered by this harness today). When thinking is ON, the model generates
            // this section itself -- priming it here would pre-empt real reasoning.
            if ( !enable_thinking )
                prompt += "<|channel>thought\n<channel|>";

            return prompt;
        }

        static std::vector<std::string_view> splitWhitespace( std::string_view text )
        {
            std::vector<std::string_view> tokens;

            size_t i = 0;

            while ( i < text.size() )
            {
                while ( i < text.size() && text[ i ] == ' ' )
                    ++i;

                if ( i >= text.size() )
                    break;

                const size_t start = i;

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
            // one artifact can be deployed at more than one.
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
            config_.precision         = resolved.precision;
            config_.is_instruct       = resolved.instruct;
            config_.base_model        = resolved.base_model;
            config_.license           = resolved.license;
            config_.streaming_capable = resolved.streaming_capable;

            // Thinking is the model's capability, so it re-derives on every switch rather than
            // carrying a preference across a family that has no reasoning channel.
            config_.thinking_capable  = resolved.thinking_capable;
            config_.show_thinking     = resolved.thinking_capable;
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

            if ( config_.context_is_automatic )
            {
                // Auto means "whatever fits the card", and what fits depends on the model, so a
                // switch re-measures rather than carrying a number derived for the model being
                // replaced -- which is the defect configured_context_length exists to prevent,
                // in the one shape that field cannot express.
                //
                // Measured before the outgoing model is released, deliberately: auto budgets
                // against device CAPACITY rather than free memory, so what is currently resident
                // does not change the answer.
                const ResolvedContext measured = resolveAutomaticContext(
                    config_.model_path, config_.model_type, config_.precision,
                    config_.quantization_mode, traits.max_context, traits.default_context );

                config_.context_length = measured.context_length;
            }
            else if ( prev_type != config_.model_type || config_.context_length == 0 )
            {
                const std::size_t configured = config_.configured_context_length;

                config_.context_length = configured == 0
                    ? traits.default_context
                    : ( configured < traits.max_context ? configured : traits.max_context );
            }

            // Destroy the current model before allocating the replacement.
            // This returns VRAM to the CUDA pool before the new model is loaded,
            // avoiding a transient old+new peak that overflows the VRAM budget
            // and forces WDDM to spill into shared system memory.
            std::visit( []( auto& m ) { m.reset(); }, model_ );

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
                const DeviceMemoryInfo memory = queryDeviceMemory();

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
                    costed.device_name = queryDeviceName();
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
                    hub_owner(), memory.total_bytes, queryDeviceName() ) )
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
         * @brief `/model load <name> [none|fp8|fp4]`.
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
                    "Usage: /model load <name> [none|fp8|fp4]." );

                return;
            }

            // Quantizing on load is a deployment choice, not an identity: it lets a BF16 artifact
            // too large for the card run anyway. A pre-quantized artifact is a different model
            // with its own name, and refuses this.
            std::optional<QuantizationMode> requested_quantization;

            for ( std::size_t index = 1; index < args.size(); ++index )
            {
                const auto parsed = parseQuantization( args[ index ] );

                if ( !parsed )
                {
                    renderer_.printInfo( std::format(
                        "Unknown option '{}'. Use none, fp8 or fp4.", args[ index ] ) );

                    return;
                }

                requested_quantization = *parsed;
            }

            // Folded before the comparison below, so `/model load LLAMA-3.2-3B-INSTRUCT-FP4`
            // against that same model resident is recognised rather than reloaded.
            const std::string name =
                resolveStoredName( std::string( args.front() ) )
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
                resolveStoredName( std::string( args.front() ) )
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
         * wrong in both directions -- too small for a Gemma turn at high effort, too large for
         * GPT-2's 1024 addressable positions.
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

            const FamilyTraits traits = familyTraits( config_.model_type );

            const ResolvedContext measured = resolveAutomaticContext(
                config_.model_path, config_.model_type, config_.precision,
                config_.quantization_mode, traits.max_context, traits.default_context );

            if ( !measured.fallback_reason.empty() )
            {
                std::cout << std::format( "  {:<16}not measured ({})\n",
                    "Largest fit:", measured.fallback_reason );

                return;
            }

            // Why it stopped short is worth one clause here, where the reader asked about context
            // specifically -- it is the one place the prefill bound is not noise.
            std::cout << std::format( "  {:<16}{}{}\n",
                "Largest fit:", measured.context_length,
                measured.bounded_by_prefill
                    ? "  (held back to keep a full prefill chunk)" : "" );

            if ( measured.context_length != config_.context_length )
            {
                renderer_.printInfo( std::format(
                    "  /context {} reloads there, /context auto keeps it measured.",
                    measured.context_length ) );
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

            const FamilyTraits traits = familyTraits( config_.model_type );

            const ResolvedContext measured = resolveAutomaticContext(
                config_.model_path, config_.model_type, config_.precision,
                config_.quantization_mode, traits.max_context, traits.default_context );

            if ( !measured.fallback_reason.empty() )
            {
                renderer_.printError( std::format(
                    "Could not measure a context for this card: {}.",
                    measured.fallback_reason ) );

                return;
            }

            reloadAtContext( measured.context_length, true );
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

        /// The sampling knobs, which reach the sampler per call and so need no reload.
        void reportSamplingSettings() const
        {
            std::cout << std::format( "  {:<16}{}\n", "temperature:", config_.temperature );
            std::cout << std::format( "  {:<16}{}\n", "top_k:", config_.top_k );
            std::cout << std::format( "  {:<16}{}\n", "top_p:", config_.top_p );

            renderer_.printInfo(
                "  Set with /set <key> <value>. temperature 0 is greedy, top_k 0 and top_p 1 "
                "each disable that filter." );
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

            renderer_.printInfo( std::format(
                "'{}' is not a setting. Use temperature, top_k or top_p.", key ) );
        }

        /**
         * @brief Warn before the load when this model will not fit, and say nothing when it will.
         *
         * Silent on the fitting path deliberately: a load that is going to work needs no
         * commentary, and this fires on every startup and every switch. What the model costs
         * when it does fit is available on demand from /model.
         *
         * Costs nothing on the device: the graph is constructed, asked, and discarded without a
         * weight being read. See Specifications/MemoryFootprint.md.
         *
         * Warn-and-proceed, deliberately -- there is no refusal here. What not fitting means
         * depends on the driver model and neither outcome can be asserted in advance, so an
         * over-eager refusal would block configurations that would have run.
         */
        void reportFootprintBeforeLoad()
        {
            const FootprintPrediction prediction =
                predictActiveFootprint( static_cast<dim_t>( config_.context_length ) );

            const std::optional<MemoryStats>& required = prediction.required;

            if ( !required )
            {
                // Silent by contract -- a pre-flight must never be what stops a model being
                // tried. At All the user has asked to see everything, and an absence with no
                // reason is what left the predictor's own failure undiagnosed for a week.
                if ( config_.detail == DetailLevel::All )
                {
                    renderer_.printInfo( std::format(
                        "No footprint prediction for {}: {}.",
                        modelName(), prediction.unavailable_reason ) );
                }

                return;
            }

            const std::size_t available =
                availableDeviceBytes( queryDeviceMemory(), residentDeviceBytes() );

            // The same grader /model list uses, against a deliberately different budget: the
            // listing asks what this card can run at all, where this asks whether one load
            // succeeds on the machine as it stands. So a model the listing showed as fitting
            // can still warn here, when something else is holding the memory -- which is the
            // answer being asked for, not a disagreement.
            const FootprintVerdict verdict = gradeFootprint( required, available );

            if ( !isOverBudget( verdict ) )
            {
                return;
            }

            // The breakdown earns its place only here: weights against working memory is what
            // says which lever applies.
            renderer_.printInfo( std::format(
                "{} at context {}: weights {}, working memory {}, about {} in total.",
                modelName(), config_.context_length,
                formatBytes( required->device_parameter_bytes ),
                formatBytes( required->device_state_bytes ),
                formatBytes( practicalDeviceBytes( *required ) ) ) );

            // One explanation of what not fitting means, shared with the listing. What the two
            // verdicts differ on is which lever helps, not what happens.
            renderer_.printError( doesNotFitExplanation( formatBytes( available ) ) );

            if ( verdict == FootprintVerdict::WeightsExceedAvailable )
            {
                // Context is not the lever when the weights alone overflow, since they do not
                // shrink with it. Quantization is, and only while the artifact still permits it.
                if ( config_.quantization_mode == QuantizationMode::None )
                {
                    renderer_.printInfo( std::format(
                        "A shorter context will not help -- the weights alone are over. "
                        "Quantizing on load is the lever -- try /model {} fp4.", modelName() ) );
                }

                return;
            }

            // The weights fit, so trimming context can bring the working memory under the line.
            suggestFittingContext();
        }

        /**
         * @brief Device memory the loaded model is holding, which a switch would give back.
         *
         * Asked of the model rather than predicted, since it is resident: a switch destroys it
         * before allocating the replacement, so this is headroom a candidate can count on.
         */
        std::size_t residentDeviceBytes() const
        {
            return std::visit( []( const auto& model ) -> std::size_t
                {
                    return model ? model->getMemoryStats().totalDeviceBytes() : 0;
                }, model_ );
        }

        /// True once weights are actually on the device, which the session config cannot say --
        /// it names the model that will be loaded as readily as the one that is.
        bool modelIsResident() const
        {
            return !std::visit( []( const auto& model ) { return model == nullptr; }, model_ );
        }

        /**
         * @brief Largest context length that would fit, reported as a suggestion.
         *
         * Asked through the same scan /context auto runs, for two reasons that both matter. The
         * number is one the user can now GET -- this used to advise editing the chat config and
         * restarting, which made it a measurement nobody could act on from where they were
         * standing. And the scan descends the grid where this bisected: bisection assumes the
         * footprint rises with context and Gemma's does not, it drops where prefill chunking caps
         * the activation buffers, so a bisection can land the wrong side of that step and report a
         * context shorter than one it had already accepted.
         *
         * Budgeted against device capacity less a margin, which is what auto uses, rather than
         * against the bytes free at this instant. Deliberate: a suggestion the user cannot
         * reproduce by typing the command it names would be worse than saying nothing.
         */
        void suggestFittingContext()
        {
            const FamilyTraits traits = familyTraits( config_.model_type );

            const ResolvedContext measured = resolveAutomaticContext(
                config_.model_path, config_.model_type, config_.precision,
                config_.quantization_mode, traits.max_context, traits.default_context );

            if ( !measured.fallback_reason.empty()
                || measured.context_length >= config_.context_length )
            {
                return;
            }

            renderer_.printInfo( std::format(
                "Context {} would fit -- run /context {}, or /context auto to keep it measured.",
                measured.context_length, measured.context_length ) );
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
            // reader of a session file has never heard of and cannot change.
            if ( config_.context_length == 0 )
            {
                throw std::invalid_argument(
                    "context_length is zero. Set a positive 'context_length' in the session "
                    "config, or remove the key to take the model's own default." );
            }

            reportFootprintBeforeLoad();

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

            const std::string_view attribution = requiredAttributionFor( config_.license );

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
                "  Precision:      {}\n"
                "  Quantization:   {}\n"
                "  Context window: {}\n"
                "  Instruct:       {}\n",
                (config_.precision == ModelPrecision::BF16) ? "bf16" : "fp32",
                quantizationName( config_.quantization_mode ),
                context_line,
                config_.is_instruct ? "yes" : "no" );

            // Reported only by a model that has the channel. An effort level beside a model with
            // no reasoning mode describes a budget for tokens it will never emit.
            if ( config_.thinking_capable )
            {
                std::cout << std::format(
                    "  Thinking:       {}\n"
                    "  Effort:         {} ({}, ~{} tokens)\n",
                    config_.show_thinking ? "on" : "off",
                    config_.thinking_effort, effort.name, effort.budget );
            }

            std::cout << std::format( "  Detail:         {}\n",
                detailLevelName( config_.detail ) );

            // Measured off the device rather than taken from the model's own report: what is
            // actually resident includes the allocator rounding and lazily grown scratch that
            // a footprint cannot model, and that residual is the whole reason to look here.
            const DeviceMemoryInfo memory = queryDeviceMemory();

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

                switch ( config_.model_type )
                {
                    case ModelType::Gpt:
                        tokenizer_ = BpeTokenizer::loadGpt2( config_.tokenizer_path );
                        break;

                    case ModelType::Llama:
                        tokenizer_ = BpeTokenizer::loadLlama32( config_.tokenizer_path );
                        break;

                    case ModelType::Gemma:
                        tokenizer_ = BpeTokenizer::loadGemma( config_.tokenizer_path );
                        break;
                }

                Logging::Logger::info( std::format( "Tokenizer loaded. Vocab size: {}", tokenizer_->getVocabSize() ) );

                // Cache the Gemma control-token ids: <tool_call|> lets generateResponse()
                // detect the tool-call boundary by raw token id rather than text scanning,
                // and the full set drives the streaming display's channel router. Only
                // genuine single-token vocab entries (per GemmaChatProtocol.md) are trusted;
                // a multi-token encode means the checkpoint lacks the registered special
                // token. Streaming requires all four routing ids or it stays buffered.
                gemma_stream_tokens_.reset();

                if ( config_.model_type == ModelType::Gemma )
                {
                    gemma_tool_call_close_token_ = probeSingleTokenId( "<tool_call|>" );

                    const auto channel_open = probeSingleTokenId( "<|channel>" );
                    const auto channel_close = probeSingleTokenId( "<channel|>" );
                    const auto tool_call_open = probeSingleTokenId( "<|tool_call>" );

                    if ( channel_open && channel_close && tool_call_open && gemma_tool_call_close_token_ )
                    {
                        GemmaStreamTokens tokens;
                        tokens.channel_open = *channel_open;
                        tokens.channel_close = *channel_close;
                        tokens.tool_call_open = *tool_call_open;
                        tokens.tool_call_close = *gemma_tool_call_close_token_;

                        // Control tokens with no display form: the buffered pipeline strips
                        // their text post-hoc (stripSpecialTokens); the router drops them live.
                        static constexpr std::string_view kSuppressedControlTokens[] = {
                            "<|turn>", "<turn|>", "<|think|>", "<|tool>", "<tool|>",
                            "<|tool_response>", "<tool_response|>",
                            "<end_of_turn>", "<start_of_turn>", "<bos>", "<eos>", "<pad>",
                            "<|image|>", "<|audio|>"
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
                else
                {
                    gemma_tool_call_close_token_ = std::nullopt;
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
        std::optional<int32_t> probeSingleTokenId( std::string_view text ) const
        {
            const auto ids = tokenizer_->encode( std::string( text ) );

            return (ids.size() == 1)
                ? std::optional<int32_t>( static_cast<int32_t>( ids[ 0 ] ) )
                : std::nullopt;
        }

        /**
         * @brief Footprint of the session's model at a context length.
         *
         * The axes come from the session config rather than a store record, which is what
         * makes this the load path's question: a prediction made under different settings
         * than the load would use describes a different model.
         */
        FootprintPrediction predictActiveFootprint( dim_t context_length ) const
        {
            return Mila::ChatApp::predictFootprint(
                config_.model_path,
                config_.model_type,
                config_.precision,
                config_.quantization_mode,
                context_length );
        }

        void loadModel()
        {
            const DeviceId device{ DeviceType::Cuda, 0 };

            switch ( config_.model_type )
            {
                case ModelType::Gpt:
                {
                    auto gpt = GptModelFP32Type::fromPretrained(
                        config_.model_path,
                        config_.context_length,
                        device,
                        /*strict=*/true );
                    if ( config_.detail == DetailLevel::All )
                    {
                        std::cout << gpt->toString();
                        std::cout << gpt->getMemoryStats().toString() << "\n";
                    }
                    model_ = std::move( gpt );
                    break;
                }

                case ModelType::Llama:
                {
                    LlamaModelConfig llama_config = LlamaModelConfig( config_.context_length );

                    if ( config_.quantization_mode == QuantizationMode::FP8 )
                        llama_config.withFP8Quantization();
                    else if ( config_.quantization_mode == QuantizationMode::FP4 )
                        llama_config.withFP4Quantization();

                    if ( config_.precision == ModelPrecision::BF16 )
                    {
                        auto llama_bf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
                            config_.model_path, llama_config, device );
                        if ( config_.detail == DetailLevel::All )
                        {
                            std::cout << llama_bf16->toString();
                            std::cout << llama_bf16->getMemoryStats().toString() << "\n";
                        }
                        model_ = std::move( llama_bf16 );
                    }
                    else
                        model_ = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>::fromPretrained(
                            config_.model_path, llama_config, device );

                    break;
                }

                case ModelType::Gemma:
                {
                    GemmaModelConfig gemma_config = GemmaModelConfig( config_.context_length );

                    if ( config_.quantization_mode == QuantizationMode::FP8 )
                        gemma_config.withFP8Quantization();
                    else if ( config_.quantization_mode == QuantizationMode::FP4 )
                        gemma_config.withFP4Quantization();

                    auto gemma = GemmaModelBF16Type::fromPretrained(
                        config_.model_path, gemma_config, device );
                    if ( config_.detail == DetailLevel::All )
                    {
                        std::cout << gemma->toString();
                        std::cout << gemma->getMemoryStats().toString() << "\n";
                    }
                    model_ = std::move( gemma );
                    break;
                }
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

            for ( size_t i = 0; i < last_turn_rounds_.size(); ++i )
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

        static constexpr const char* kVersion = "v0.20";

        const std::string& modelName() const { return config_.model_name; }

        void printBanner() const
        {
            renderer_.printWelcomeBox( std::format( "Mila Chat {}", kVersion ) );
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
                    const std::string thinking_display = config_.show_thinking
                        ? std::string( thinkingEffort( config_.thinking_effort ).name )
                        : "off";

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
  /model load <name> [quant]         Load a model (clears history). quant quantizes an
                                     unquantized artifact on load: none, fp8, fp4.
  /model install <name>              Download and install a published model
  /model remove <name>               Remove an installed model and reclaim its blobs
  /context                           Show the context window and the largest that fits
  /context <n>|auto                  Set the context window (rebuilds, keeps the conversation)
  /thinking [on|off]                 Show or set the reasoning channel
  /set <key> <value>                 Set temperature, top_k or top_p
  /effort [1-5]                      Show or set the thinking token-budget level
  /verbose [off|thoughts|all]        Show or set display detail (reasoning, raw + logs)
  /stats                             Show per-round timing for the last turn
  /seed <n>                          Reseed the sampler for reproducible generation
  /exit                              Exit the application

Models:         a bare /model <name> reports; loading is spelled out, because it takes
                seconds and clears the conversation.
Quantization:   a name ending -fp4/-fp8 is a pre-quantized artifact. For an unquantized
                one, /model load <name> fp4 quantizes it on load -- same weights, less
                VRAM, but the full file is still read.
Thinking:       Gemma's <|think|> mode. Toggling it does not reload weights. Effort
                (length) is set with /effort 1-5.
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

            // Only advertise tools that have a registered handler.
            // Describing unhandled tools primes the model to emit tool calls
            // it will never get a result for.
            std::vector<ToolDefinition> active_tools;
            for ( const auto& tool : system_prompt_config_.tools )
            {
                if ( tool_handlers_.contains( tool.name ) )
                    active_tools.push_back( tool );
            }

            if ( !active_tools.empty() )
            {
                if ( config_.model_type == ModelType::Llama )
                {
                    // Instruction text precedes the tool list per the Llama 3.2 zero-shot
                    // tool-calling format the model was fine-tuned on.
                    system_content +=
                        "\n\nIf you decide to invoke any of the function(s), you MUST put it in the "
                        "format of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
                        "func_name2(params)]\n"
                        "You SHOULD NOT include any other text in the response.\n\n"
                        "Here is a list of functions in JSON format that you can invoke:\n";
                }
                else
                {
                    // No invented call-syntax instructions: Gemma 4 has its own trained
                    // <|tool_call>/<tool_call|> protocol (GemmaChatProtocol.md). This is a
                    // plain description, deliberately left unopinionated about call syntax,
                    // so /verbose all can capture the model's native format via printRaw.
                    system_content += "\n\nYou have access to the following tools:\n";
                }

                system_content += serializeTools( active_tools );
            }

            history_.push_back( { MessageRole::System, std::move( system_content ) } );
        }

        ChatConfig config_;
        ModelVariant model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage> history_;
        std::stop_source stop_src_;
        std::unordered_map<std::string, std::function<std::string( const std::string& )>> tool_handlers_;
        ConsoleRenderer renderer_;

        // Gemma 4 tool-call protocol probe state (see generateResponse()).
        std::optional<int32_t> gemma_tool_call_close_token_;

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
