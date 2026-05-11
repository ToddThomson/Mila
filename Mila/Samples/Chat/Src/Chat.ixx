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
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;

    using GptModelFP32Type   = GptModel<DeviceType::Cuda, TensorDataType::FP32>;
    using LlamaModelFP32Type = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>;
    using LlamaModelBF16Type = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

    using ModelVariant = std::variant<
        std::unique_ptr<GptModelFP32Type>,
        std::unique_ptr<LlamaModelFP32Type>,
        std::unique_ptr<LlamaModelBF16Type>
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

            if ( !system_prompt_config_.system_prompt.empty() )
            {
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
                    system_content += "\n\nYou have access to the following tools:\n";
                    system_content += serializeTools( active_tools );

                    system_content += "\n\nWhen you need to call a tool, respond using ONLY this format and nothing else:\n"
                        "[tool_name(param_name=param_value)]\n"
                        "Do not describe what you are doing. Do not include any other text.";

                    //system_content += "\n\nWhen calling a tool, you MUST respond using exactly this format and nothing else:\n"
                    //    "<|python_tag|>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}<|eom_id|>\n"
                    //    "Do not describe what you are doing. Do not include any other text. "
                    //    "Emit only the tool call token sequence and stop.";
                }

                history_.push_back( { MessageRole::System, std::move( system_content ) } );
            }

            std::string user_input;

            while ( true )
            {
                std::cout << "\nYou: ";
                std::getline( std::cin, user_input );

                if ( user_input.empty() )
                    continue;

                if ( user_input == "exit" || user_input == "quit" )
                {
                    std::cout << "Goodbye!\n";
                    break;
                }

                if ( user_input == "clear" )
                {
                    history_.clear();

                    if ( !system_prompt_config_.system_prompt.empty() )
                    {
                        history_.push_back( { MessageRole::System, system_prompt_config_.system_prompt } );
                    }

                    std::cout << "Conversation history cleared.\n";
                    continue;
                }

                if ( user_input == "help" )
                {
                    printHelp();
                    continue;
                }

                history_.push_back( { MessageRole::User, user_input } );

                std::string response;
                response.reserve( 4096 );

                generateResponse( response, /*stream=*/false );

                handleResponse( response );
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
                std::cout << "\nMila: " << response << '\n';
                history_.push_back( { MessageRole::Assistant, response } );
                
                return;
            }

            std::optional<ToolCall> tool_call;

            try
            {
                tool_call = ToolCallParser::parse( response );
            }
            catch ( const std::runtime_error& e )
            {
                std::cerr << "\n[tool call parse error: " << e.what() << "]\n";
                std::cout << "\nMila: " << response << '\n';
                history_.push_back( { MessageRole::Assistant, response } );
                return;
            }

            if ( !tool_call.has_value() )
            {
                std::cout << "\nMila: " << response << '\n';
                history_.push_back( { MessageRole::Assistant, response } );
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

            std::cout << "\nMila: ";

            for ( const auto& msg : history_ )
            {
                std::cerr << "[DEBUG] role: " << static_cast<int>(msg.role)
                    << " content: [" << msg.content << "]\n";
            }

            generateResponse( final_response, /*stream=*/true );

            std::cout << '\n';

            history_.push_back( { MessageRole::Assistant, final_response } );
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

        /**
         * @brief Format history, tokenize, generate, and optionally stream the response.
         *
         * When stream is true each decoded token is printed to stdout as it is produced.
         * When stream is false generation runs silently; the caller inspects the response
         * string to decide whether to display it (plain text) or discard it (tool call).
         *
         * @param response String to accumulate the generated response into.
         * @param stream   When true, decoded tokens are printed to stdout as generated.
         */
        void generateResponse( std::string& response, bool stream )
        {
            std::vector<int32_t> input_tokens = buildInputTokens();

            stop_src_ = std::stop_source{};

            std::visit(
                [&]( auto& m )
                {
                    m->generateStreaming(
                        input_tokens,
                        [&]( int32_t tok )
                        {
                            auto text = tokenizer_->decode(
                                std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                            response += text;

                            if ( stream )
                                std::cout << text << std::flush;
                        },
                        config_.max_new_tokens,
                        config_.temperature,
                        config_.top_k,
                        stop_src_.get_token() );
                },
                model_ );
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

            if ( config_.model_type == ModelType::Llama && isInstructModel() )
            {
                prompt = MessageFormatter::format( history_ );
            }
            else
            {
                prompt = history_.back().content;
            }

            // DEBUG
            std::cerr << "[PROMPT DEBUG]\n" << prompt << "\n[END PROMPT]\n";

            auto token_ids = tokenizer_->encode( prompt );

            // DEBUG: dump token IDs for HF comparison
            //std::cout << "[TOKEN DEBUG] Prompt token count: " << token_ids.size() << "\n";
            //std::cout << "[TOKEN DEBUG] Token IDs: ";
            //for ( const auto& id : token_ids )
            //{
            //    std::cout << id << " ";
            //}
            //std::cout << "\n";
            // END DEBUG

            return std::vector<int32_t>( token_ids.begin(), token_ids.end() );
        }

        /**
         * @brief Returns true when the loaded model is an instruct variant.
         *
         * Inferred from the model path filename — instruct models contain
         * "instruct" (case-insensitive). Consistent with the naming convention
         * used by the Llama weight converter.
         */
        bool isInstructModel() const
        {
            std::string lower = config_.model_path.string();
            std::ranges::transform( lower, lower.begin(),
                []( unsigned char c ) { return static_cast<char>(std::tolower( c )); } );

            return lower.find( "instruct" ) != std::string::npos;
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
            //Logging::Logger::info( std::format( "Loading model from: {}", config_.model_path ) );

            const DeviceId device{ DeviceType::Cuda, 0 };

            switch ( config_.model_type )
            {
                case ModelType::Gpt:
                    model_ = GptModelFP32Type::fromPretrained(
                        config_.model_path,
                        config_.context_length,
                        device,
                        /*strict=*/true );
                    break;

                case ModelType::Llama:
                {
                    LlamaModelConfig llama_config = LlamaModelConfig( config_.context_length );

                    if ( config_.precision == ModelPrecision::BF16 )
                    {
                        model_ = LlamaModelBF16Type::fromPretrained(
                            config_.model_path,
                            llama_config,
                            device );
                    }
                    else
                    {
                        model_ = LlamaModelFP32Type::fromPretrained(
                            config_.model_path,
                            llama_config,
                            device );
                    }
                    
                    break;
                }
            }

            std::visit(
                []( auto& m )
                {
                    std::cout << m->toString();
                    std::cout << m->getMemoryStats().toString() << "\n";
                    std::cout << "Model loaded successfully\n";
                },
                model_ );
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

        void printWelcome() const
        {
            const char* backend   = (config_.model_type == ModelType::Gpt) ? "GPT" : "LLaMA";
            const char* precision = (config_.precision == ModelPrecision::BF16) ? "BF16" : "FP32";
            const char* mode      = isInstructModel() ? " Instruct" : "";

            std::cout << R"(
+--------------------------------------+
|         Mila Chat CLI v1.0           |
|      Powered by Mila DNN Library     |
+--------------------------------------+

Type 'help' for commands, 'exit' to quit.
)" << "\n";

            std::cout << "Backend: " << backend << mode << " (" << precision << ")\n";

            if ( !system_prompt_config_.system_prompt.empty() )
            {
                std::cout << "System prompt: active";

                if ( !system_prompt_config_.tools.empty() )
                {
                    std::cout << std::format(
                        " ({} tool{})",
                        system_prompt_config_.tools.size(),
                        system_prompt_config_.tools.size() == 1 ? "" : "s" );
                }

                std::cout << "\n";
            }
        }

        void printHelp() const
        {
            std::cout << R"(
Available Commands:
  help   - Show this help message
  clear  - Clear conversation history
  exit   - Exit the application
  quit   - Exit the application

Just type your message to chat with Mila AI.
)" << "\n";
        }

        ChatConfig config_;
        ModelVariant model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage> history_;
        std::stop_source stop_src_;
        std::unordered_map<std::string, std::function<std::string( const std::string& )>> tool_handlers_;
    };
}