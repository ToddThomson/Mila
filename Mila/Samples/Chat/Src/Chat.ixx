/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Supports GptModel (FP32) and LlamaModel (FP32 or BF16) backends,
 * selected at construction via ChatConfig. The active model is stored
 * as a std::variant so each template instantiation retains its full
 * static type through the generate path. Llama instruct models use
 * structured ChatMessage history formatted via MessageFormatter.
 * Tool calling is supported for Llama instruct models via ToolCallParser
 * and registered handler functions.
 */

module;
#include <iostream>
#include <string>
#include <vector>
#include <variant>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <future>
#include <stop_token>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <unordered_map>

export module Mila.Chat;

export import Chat.Config;
export import Chat.ModelCatalog;
export import Chat.Message;
export import Chat.MessageFormatter;
export import Chat.SystemPrompt;
export import Chat.ToolCallParser;
import Chat.ChannelParser;
import Chat.Json;
import Chat.Renderer;

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

        void run()
        {
            printWelcome();
            loadActiveModel();
            clearHistory();

            std::string user_input;

            while ( true )
            {
                renderer_.printUserPrompt();
                std::getline( std::cin, user_input );

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
                    if ( cmd == "verbose" || cmd.starts_with( "verbose " ) )
                    {
                        if ( cmd == "verbose" )
                        {
                            renderer_.printInfo( std::format(
                                "Detail: {}. Set with /verbose <off|thoughts|tools|all>.",
                                detailLevelName( config_.detail ) ) );
                            continue;
                        }

                        const auto level = parseDetailLevel( cmd.substr( 8 ) );

                        if ( !level )
                        {
                            renderer_.printInfo( "Usage: /verbose <off|thoughts|tools|all>." );
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

                    if ( cmd == "model" || cmd.starts_with( "model " ) )
                    {
                        if ( cmd == "model" )
                        {
                            printModelInfo();
                            continue;
                        }

                        const std::vector<std::string_view> args = splitWhitespace( cmd.substr( 6 ) );

                        if ( args.empty() )
                        {
                            printModelInfo();
                            continue;
                        }

                        const std::string_view alias = args.front();

                        const ModelEntry* entry = findModel( alias );
                        if ( entry == nullptr )
                        {
                            renderer_.printInfo( std::format(
                                "Unknown model alias '{}'. Type /help for available aliases.", alias ) );
                            continue;
                        }

                        // Remaining tokens are an optional quantization and an optional
                        // "thinking" flag, order-independent. An omitted quantization
                        // falls back to the alias default (FP4 for Gemma, whose BF16
                        // weights do not fit the dev card).
                        QuantizationMode quant = entry->default_quantization;
                        bool thinking = false;
                        bool bad_token = false;

                        for ( size_t i = 1; i < args.size(); ++i )
                        {
                            if ( args[ i ] == "thinking" )
                            {
                                thinking = true;
                                continue;
                            }

                            const auto parsed = parseQuantization( args[ i ] );
                            if ( !parsed )
                            {
                                renderer_.printInfo( std::format(
                                    "Unknown option '{}'. Use none, fp8, fp4, or thinking.", args[ i ] ) );
                                bad_token = true;
                                break;
                            }
                            quant = *parsed;
                        }

                        if ( bad_token )
                            continue;

                        config_.show_thinking = thinking;

                        // Toggling only the thinking flag on the already-loaded model
                        // must not trigger a multi-GB weight reload.
                        if ( isCurrentModel( *entry, quant ) )
                        {
                            renderer_.printInfo( thinking
                                ? "Thinking display enabled."
                                : "Thinking display disabled." );
                            continue;
                        }

                        switchModel( *entry, quant );
                        continue;
                    }

                    renderer_.printInfo( std::format(
                        "Unknown command: {}. Type /help for available commands.", user_input ) );
                    continue;
                }

                history_.push_back( { MessageRole::User, user_input } );

                std::string response;
                response.reserve( 4096 );

                generateResponse( response );

                handleResponse( response );

                printGenerationStatistics();
            }
        }

    private:

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
            if ( tool_handlers_.empty() )
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

            if ( config_.detail >= DetailLevel::Tools )
                renderer_.printInfo( std::format(
                    "[tool call] {}({})", tool_call->name, tool_call->arguments ) );

            const std::string tool_result = dispatchTool( *tool_call );

            if ( config_.detail >= DetailLevel::Tools )
                renderer_.printInfo( std::format( "[tool result] {}", tool_result ) );

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

            // The reasoning channel only exists when thinking mode is active.
            if ( level >= DetailLevel::Thoughts && !config_.show_thinking )
                message += " (Thinking mode is off — enable it with /model gemma-12b thinking to see reasoning.)";

            renderer_.printInfo( message );
        }

        /**
         * @brief Strip tokens, split channels, render, and store an assistant turn.
         *
         * The Gemma reasoning channel is rendered only at detail level Thoughts or
         * above, and the verbatim raw output at All; the final answer is always
         * rendered and is the only text pushed to history so the reasoning never
         * re-enters the next formatted prompt. For models that emit no reasoning
         * channel the parse is a pass-through and the whole response is the answer.
         */
        void emitAssistantResponse( const std::string& raw )
        {
            if ( config_.detail == DetailLevel::All )
                renderer_.printRaw( raw );

            const std::string clean = stripSpecialTokens( raw );
            const ParsedResponse parsed = ChannelParser::parse( clean );

            if ( config_.detail >= DetailLevel::Thoughts && !parsed.thinking.empty() )
                renderer_.printThinking( parsed.thinking );

            renderer_.printMilaResponse( parsed.answer );
            history_.push_back( { MessageRole::Assistant, parsed.answer } );
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
            renderer_.startSpinner();

            std::vector<int32_t> input_tokens = buildInputTokens();

            stop_src_ = std::stop_source{};

            GenerateConfig gen_config;
            gen_config.max_new_tokens = config_.max_new_tokens;
            gen_config.temperature = config_.temperature;
            gen_config.top_k = config_.top_k;

            std::visit(
                [&]( auto& m )
                {
                    auto result = m->generateStreaming(
                        input_tokens,
                        [&]( int32_t tok )
                        {
                            response += tokenizer_->decode(
                                std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                        },
                        gen_config,
                        stop_src_.get_token() );

                    last_statistics_ = result.statistics;
                },
                model_ );

            renderer_.stopSpinner();
        }

        /**
         * @brief Build the token sequence for the current generation step.
         *
         * Llama instruct models format the full structured history via
         * MessageFormatter. GPT and Llama base models encode only the last
         * user message content.
         *
         * @return Token ids ready to pass to generateAsync().
         */
        std::vector<int32_t> buildInputTokens() const
        {
            std::string prompt;

            if ( config_.model_type == ModelType::Llama && config_.is_instruct )
                prompt = MessageFormatter::format( history_ );
            else if ( config_.model_type == ModelType::Gemma && config_.is_instruct )
                prompt = formatGemmaPrompt( history_, config_.show_thinking, config_.thinking_effort );
            else
                prompt = history_.back().content;

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
         * @brief True when the requested model entry and quantization match the model
         *        already resident in memory, so a switch would be a no-op reload.
         */
        bool isCurrentModel( const ModelEntry& entry, QuantizationMode quant ) const
        {
            const bool loaded = !std::visit( []( const auto& m ) { return m == nullptr; }, model_ );

            return loaded
                && config_.model_type        == entry.family
                && config_.model_size        == entry.size
                && config_.precision         == entry.precision
                && config_.quantization_mode == quant;
        }

        void switchModel( const ModelEntry& entry, QuantizationMode quant )
        {
            const ModelType prev_type = config_.model_type;

            config_.model_type        = entry.family;
            config_.model_size        = entry.size;
            config_.precision         = entry.precision;
            config_.is_instruct       = entry.is_instruct;
            config_.quantization_mode = quant;
            config_.model_path        = config_.models_dir / entry.weights_file;
            config_.tokenizer_path    = config_.models_dir / entry.tokenizer_file;

            // Preserve context_length across same-architecture switches; reset to the
            // new model's default on an architecture change.
            if ( prev_type != config_.model_type )
                config_.context_length = entry.default_context;

            // Destroy the current model before allocating the replacement.
            // This returns VRAM to the CUDA pool before the new model is loaded,
            // avoiding a transient old+new peak that overflows the VRAM budget
            // and forces WDDM to spill into shared system memory.
            std::visit( []( auto& m ) { m.reset(); }, model_ );

            loadActiveModel();
            clearHistory();

            renderer_.printInfo( "Model switched. Conversation history cleared." );
        }

        /**
         * @brief Load the active tokenizer and model, under the progress spinner.
         *
         * At detail level All the spinner is skipped so the INFO logging and the
         * model / memory dumps remain readable; otherwise the multi-second weight
         * load runs under the same braille spinner used for turns.
         */
        void loadActiveModel()
        {
            if ( config_.detail == DetailLevel::All )
            {
                initializeTokenizer();
                loadModel();
                return;
            }

            renderer_.startSpinner( std::format( "Loading {} ({})",
                modelAlias(), quantizationName( config_.quantization_mode ) ) );

            initializeTokenizer();
            loadModel();

            renderer_.stopSpinner();
        }

        void printModelInfo() const
        {
            const ThinkingEffort& effort = thinkingEffort( config_.thinking_effort );

            std::cout << std::format(
                "  Model:        {}\n"
                "  Precision:    {}\n"
                "  Quantization: {}\n"
                "  Instruct:     {}\n"
                "  Thinking:     {}\n"
                "  Effort:       {} ({}, ~{} tokens)\n"
                "  Detail:       {}\n",
                modelAlias(),
                (config_.precision == ModelPrecision::BF16) ? "bf16" : "fp32",
                quantizationName( config_.quantization_mode ),
                config_.is_instruct ? "yes" : "no",
                config_.show_thinking ? "on" : "off",
                config_.thinking_effort, effort.name, effort.budget,
                detailLevelName( config_.detail ) );
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
            }
            catch ( const std::exception& e )
            {
                Logging::Logger::error( std::format( "Failed to load tokenizer: {}", e.what() ) );

                throw;
            }
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
         * @brief Print generation statistics from the most recent response.
         *
         * Displays time to first token (TTFT) and autoregressive decode throughput
         * (tokens per second) after each completed generation run. Only printed
         * when the statistics are valid (i.e. at least one generation has run).
         */
        void printGenerationStatistics() const
        {
            if ( !last_statistics_.valid() )
                return;

            renderer_.printStats(
                last_statistics_.prefill_time_ms,
                last_statistics_.decode_tokens_per_second,
                static_cast<int>( last_statistics_.tokens_generated ) );
        }

        static constexpr const char* kVersion = "v0.20";

        std::string modelAlias() const
        {
            // Reverse the catalog: family + size + precision uniquely identify an alias.
            for ( const auto& entry : kModelCatalog )
            {
                if ( entry.family == config_.model_type
                    && entry.size == config_.model_size
                    && entry.precision == config_.precision )
                    return std::string( entry.alias );
            }

            return "unknown";
        }

        void printWelcome() const
        {
            // REVIEW: When thinking is on we should display the current effort level here too, since it is a user-facing setting.

            renderer_.printWelcomeBox( std::format( "Mila Chat {}", kVersion ) );
            renderer_.printInfo( std::format( "  Model: {}  ·  Thinking: {}  ·  Detail: {}",
                modelAlias(), config_.show_thinking ? "on" : "off", detailLevelName( config_.detail ) ) );
            std::cout << "  Type /help for commands, /exit to quit.\n\n";
        }

        void printHelp() const
        {
            std::cout << R"(
Available commands:
  /help                              Show this help message
  /clear                             Clear conversation history
  /model                             Show current model and quantization
  /model <alias> [quant] [thinking]  Switch model (clears history)
  /effort [1-5]                      Show or set the thinking token-budget level
  /verbose [off|thoughts|tools|all]  Show or set display detail (reasoning, tool calls, raw + logs)
  /exit                              Exit the application

Model aliases:  )";

            bool first = true;
            for ( const auto& entry : kModelCatalog )
            {
                std::cout << (first ? "" : ", ") << entry.alias;

                if ( entry.alias == kDefaultModelAlias )
                    std::cout << " (default)";

                first = false;
            }

            std::cout << R"(
Quantization:   none, fp8, fp4  (each model has its own default)
Thinking:       add 'thinking' to make Gemma reason (its <|think|> mode); toggling it
                does not reload weights. Effort (length) is set with /effort 1-5.
Detail:         /verbose thoughts shows the reasoning, tools the tool calls, all adds
                raw output + logging. The model reasons even when detail is off.

Examples:
  /model gemma-12b fp4 thinking
  /verbose thoughts
  /effort 5
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
        ModelVariant model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage> history_;
        std::stop_source stop_src_;
        std::unordered_map<std::string, std::function<std::string( const std::string& )>> tool_handlers_;
        ConsoleRenderer renderer_;

        // Statistics from the most recent generateResponse() call, for printGenerationStatistics().
        GenerationStatistics last_statistics_{};
    };
}
