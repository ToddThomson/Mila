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
#include <functional>
#include <unordered_map>

export module Mila.Chat;

export import Chat.Config;
export import Chat.Message;
export import Chat.MessageFormatter;
export import Chat.SystemPrompt;
export import Chat.ToolCallParser;
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
         * Loads the tokenizer, model, and optional system prompt in order.
         * Throws on any failure — no partially-constructed session is observable.
         *
         * @param config Session configuration.
         * @throws std::runtime_error on tokenizer, model, or system prompt load failure.
         */
        explicit Chat( ChatConfig config )
            : config_( std::move( config ) )
        {
            initializeTokenizer();
            loadModel();
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

                    if ( cmd == "model" || cmd.starts_with( "model " ) )
                    {
                        if ( cmd == "model" )
                        {
                            printModelInfo();
                            continue;
                        }

                        std::string_view args = cmd.substr( 6 );

                        const auto space = args.find( ' ' );
                        const std::string_view alias    = (space == std::string_view::npos) ? args : args.substr( 0, space );
                        const std::string_view quant_sv = (space == std::string_view::npos) ? std::string_view{} : args.substr( space + 1 );

                        const auto desc = resolveAlias( alias );
                        if ( !desc )
                        {
                            renderer_.printInfo( std::format(
                                "Unknown model alias '{}'. Type /help for available aliases.", alias ) );
                            continue;
                        }

                        // An omitted quantization argument falls back to the alias default
                        // (FP4 for Gemma, whose BF16 weights do not fit the dev card).
                        QuantizationMode quant;
                        if ( quant_sv.empty() )
                        {
                            quant = desc->default_quantization;
                        }
                        else
                        {
                            const auto parsed = parseQuantization( quant_sv );
                            if ( !parsed )
                            {
                                renderer_.printInfo( std::format(
                                    "Unknown quantization '{}'. Use none, fp8, or fp4.", quant_sv ) );
                                continue;
                            }
                            quant = *parsed;
                        }

                        switchModel( *desc, quant );
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
                const std::string clean = stripSpecialTokens( response );
                renderer_.printMilaResponse( clean );
                history_.push_back( { MessageRole::Assistant, clean } );
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
                const std::string clean_err = stripSpecialTokens( response );
                renderer_.printMilaResponse( clean_err );
                history_.push_back( { MessageRole::Assistant, clean_err } );
                return;
            }

            if ( !tool_call.has_value() )
            {
                const std::string clean = stripSpecialTokens( response );
                renderer_.printMilaResponse( clean );
                history_.push_back( { MessageRole::Assistant, clean } );
                return;
            }

            ChatMessage assistant_turn;
            assistant_turn.role = MessageRole::Assistant;
            assistant_turn.tool_calls.push_back( *tool_call );
            history_.push_back( std::move( assistant_turn ) );

            const std::string tool_result = dispatchTool( *tool_call );

            ChatMessage tool_turn;
            tool_turn.role = MessageRole::Tool;
            tool_turn.content = tool_result;
            tool_turn.tool_call_id = tool_call->id;
            history_.push_back( std::move( tool_turn ) );

            std::string final_response;
            final_response.reserve( 512 );

            generateResponse( final_response );

            const std::string clean_final = stripSpecialTokens( final_response );
            renderer_.printMilaResponse( clean_final );
            history_.push_back( { MessageRole::Assistant, clean_final } );
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
                "<end_of_turn>", "<start_of_turn>", "<bos>", "<eos>", "<pad>"
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
            renderer_.beginThinking();

            std::vector<int32_t> input_tokens = buildInputTokens();

            stop_src_ = std::stop_source{};

            std::visit(
                [&]( auto& m )
                {
                    m->generateStreaming(
                        input_tokens,
                        [&]( int32_t tok )
                        {
                            response += tokenizer_->decode(
                                std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                        },
                        config_.max_new_tokens,
                        config_.temperature,
                        config_.top_k,
                        stop_src_.get_token() );
                },
                model_ );

            renderer_.endThinking();
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
                prompt = formatGemmaPrompt( history_ );
            else
                prompt = history_.back().content;

            auto token_ids = tokenizer_->encode( prompt );

            return std::vector<int32_t>( token_ids.begin(), token_ids.end() );
        }

        /**
         * @brief Render a conversation history into the Gemma instruct chat template.
         *
         * Gemma 4 wraps each turn as <start_of_turn>{role}\n{content}<end_of_turn>\n
         * with roles "user" and "model" (the assistant), and terminates with a
         * <start_of_turn>model\n primer to prime generation. Gemma has no dedicated
         * system role, so a System turn is folded into the start of the next user
         * turn. The Gemma tokenizer encodes <bos>/<start_of_turn>/<end_of_turn> as
         * atomic special tokens, so they are emitted as literal text here.
         */
        static std::string formatGemmaPrompt( const std::vector<ChatMessage>& history )
        {
            std::string prompt = "<bos>";
            std::string pending_system;

            for ( const auto& message : history )
            {
                if ( message.role == MessageRole::System )
                {
                    pending_system = message.content;
                    continue;
                }

                const bool is_model = (message.role == MessageRole::Assistant);
                prompt += is_model ? "<start_of_turn>model\n" : "<start_of_turn>user\n";

                if ( !is_model && !pending_system.empty() )
                {
                    prompt += pending_system;
                    prompt += "\n\n";
                    pending_system.clear();
                }

                prompt += message.content;
                prompt += "<end_of_turn>\n";
            }

            prompt += "<start_of_turn>model\n";

            return prompt;
        }

        struct ModelDescriptor
        {
            ModelType        type;
            ModelSize        size;
            ModelPrecision   precision;
            bool             is_instruct;
            QuantizationMode default_quantization;  ///< Used when /model omits the quant argument.
        };

        static std::optional<ModelDescriptor> resolveAlias( std::string_view alias )
        {
            if ( alias == "gpt2" )          return ModelDescriptor{ ModelType::Gpt,   ModelSize::B3,  ModelPrecision::FP32, false, QuantizationMode::None };
            if ( alias == "llama-1b" )      return ModelDescriptor{ ModelType::Llama, ModelSize::B1,  ModelPrecision::BF16, true,  QuantizationMode::None };
            if ( alias == "llama-3b" )      return ModelDescriptor{ ModelType::Llama, ModelSize::B3,  ModelPrecision::BF16, true,  QuantizationMode::None };
            if ( alias == "llama-8b" )      return ModelDescriptor{ ModelType::Llama, ModelSize::B8,  ModelPrecision::BF16, true,  QuantizationMode::None };
            if ( alias == "llama-1b-fp32" ) return ModelDescriptor{ ModelType::Llama, ModelSize::B1,  ModelPrecision::FP32, true,  QuantizationMode::None };
            if ( alias == "llama-3b-fp32" ) return ModelDescriptor{ ModelType::Llama, ModelSize::B3,  ModelPrecision::FP32, true,  QuantizationMode::None };
            if ( alias == "llama-8b-fp32" ) return ModelDescriptor{ ModelType::Llama, ModelSize::B8,  ModelPrecision::FP32, true,  QuantizationMode::None };
            // Gemma 4 12B: BF16 weights (~24 GB) do not fit the dev card, so FP4 is the default.
            if ( alias == "gemma-12b" )     return ModelDescriptor{ ModelType::Gemma, ModelSize::B12, ModelPrecision::BF16, true,  QuantizationMode::FP4 };
            return std::nullopt;
        }

        static std::optional<QuantizationMode> parseQuantization( std::string_view s )
        {
            if ( s.empty() || s == "none" ) return QuantizationMode::None;
            if ( s == "fp8" )               return QuantizationMode::FP8;
            if ( s == "fp4" )               return QuantizationMode::FP4;
            return std::nullopt;
        }

        void switchModel( const ModelDescriptor& desc, QuantizationMode quant )
        {
            const ModelType prev_type = config_.model_type;

            config_.model_type        = desc.type;
            config_.model_size        = desc.size;
            config_.precision         = desc.precision;
            config_.is_instruct       = desc.is_instruct;
            config_.quantization_mode = quant;

            // Preserve context_length across same-architecture switches; reset on arch change.
            if ( prev_type != config_.model_type )
                config_.context_length = defaultContextLength( config_.model_type );

            if ( config_.model_type == ModelType::Gpt )
            {
                config_.model_path     = config_.models_dir / "gpt2" / "gpt2_small_fp32.bin";
                config_.tokenizer_path = config_.models_dir / "gpt2" / "gpt2_tokenizer.bin";
            }
            else if ( config_.model_type == ModelType::Gemma )
            {
                config_.model_path     = config_.models_dir / "gemma" / "gemma4_12b_it_bf16.bin";
                config_.tokenizer_path = config_.models_dir / "gemma" / "gemma_tokenizer.bin";
            }
            else
            {
                const char* family_str = (config_.model_size == ModelSize::B8) ? "llama31" : "llama32";
                const char* size_str   = (config_.model_size == ModelSize::B1) ? "1b"
                                       : (config_.model_size == ModelSize::B8) ? "8b" : "3b";
                const char* prec_str   = (config_.precision  == ModelPrecision::BF16) ? "bf16" : "fp32";
                config_.model_path     = config_.models_dir / "llama" /
                    std::format( "{}_{}_instruct_{}.bin", family_str, size_str, prec_str );
                config_.tokenizer_path = config_.models_dir / "llama" / "llama32_tokenizer.bin";
            }

            renderer_.printInfo( std::format( "Loading: {}", config_.model_path.filename().string() ) );

            // Destroy the current model before allocating the replacement.
            // This returns VRAM to the CUDA pool before the new model is loaded,
            // avoiding a transient old+new peak that overflows the VRAM budget
            // and forces WDDM to spill into shared system memory.
            std::visit( []( auto& m ) { m.reset(); }, model_ );

            initializeTokenizer();
            loadModel();
            clearHistory();

            renderer_.printInfo( "Model switched. Conversation history cleared." );
        }

        void printModelInfo() const
        {
            const char* quant_str;
            switch ( config_.quantization_mode )
            {
                case QuantizationMode::FP8: quant_str = "fp8"; break;
                case QuantizationMode::FP4: quant_str = "fp4"; break;
                default:                    quant_str = "none"; break;
            }

            const std::string alias = modelAlias();

            std::cout << std::format(
                "  Model:        {}\n"
                "  Precision:    {}\n"
                "  Quantization: {}\n"
                "  Instruct:     {}\n",
                alias,
                (config_.precision == ModelPrecision::BF16) ? "bf16" : "fp32",
                quant_str,
                config_.is_instruct ? "yes" : "no" );
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
                    std::cout << gpt->toString();
                    std::cout << gpt->getMemoryStats().toString() << "\n";
                    std::cout << "Model loaded successfully\n";
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
                        std::cout << llama_bf16->toString();
                        std::cout << llama_bf16->getMemoryStats().toString() << "\n";
                        model_ = std::move( llama_bf16 );
                    }
                    else
                        model_ = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>::fromPretrained(
                            config_.model_path, llama_config, device );

                    std::cout << "Model loaded successfully\n";
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
                    std::cout << gemma->toString();
                    std::cout << gemma->getMemoryStats().toString() << "\n";
                    std::cout << "Model loaded successfully\n";
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
                std::cout << "Loading system prompt from: "
                    << *config_.system_prompt_path << "\n";

                system_prompt_config_ = SystemPromptLoader::load(
                    *config_.system_prompt_path );

                std::cout << "System prompt loaded";

                if ( !system_prompt_config_.tools.empty() )
                {
                    std::cout << std::format(
                        " with {} tool definition{}",
                        system_prompt_config_.tools.size(),
                        system_prompt_config_.tools.size() == 1 ? "" : "s" );
                }

                std::cout << ".\n";
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
            std::visit(
                [this]( const auto& model )
                {
                    const auto& stats = model->getLastGenerationStatistics();

                    if ( !stats.valid() )
                        return;

                    renderer_.printStats(
                        stats.prefill_time_ms,
                        stats.decode_tokens_per_second,
                        static_cast<int>( stats.tokens_generated ) );
                },
                model_ );
        }

        static constexpr const char* kVersion = "v0.1";

        std::string modelAlias() const
        {
            if ( config_.model_type == ModelType::Gpt )
                return "gpt2";

            if ( config_.model_type == ModelType::Gemma )
                return "gemma-12b";

            const char* size_str = (config_.model_size == ModelSize::B1) ? "1b"
                                  : (config_.model_size == ModelSize::B8) ? "8b" : "3b";
            return (config_.precision == ModelPrecision::FP32)
                ? std::format( "llama-{}-fp32", size_str )
                : std::format( "llama-{}", size_str );
        }

        void printWelcome() const
        {
            renderer_.printWelcomeBox( std::format( "Mila Chat {}", kVersion ) );
            renderer_.printInfo( std::format( "  Model: {}", modelAlias() ) );
            std::cout << "  Type /help for commands, /exit to quit.\n\n";
        }

        void printHelp() const
        {
            std::cout << R"(
Available commands:
  /help                      Show this help message
  /clear                     Clear conversation history
  /model                     Show current model and quantization
  /model <alias> [quant]     Switch model (clears history)
  /exit                      Exit the application

Model aliases:  llama-3b (default), llama-1b, llama-8b, llama-3b-fp32, llama-1b-fp32, llama-8b-fp32, gemma-12b, gpt2
Quantization:   none (default), fp8, fp4  (gemma-12b defaults to fp4)

Examples:
  /model llama-3b
  /model llama-3b fp8
  /model llama-8b fp4
  /model gemma-12b
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
    };
}