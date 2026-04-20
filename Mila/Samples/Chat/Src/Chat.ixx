/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Supports GptModel (FP32) and LlamaModel (FP32 or BF16) backends,
 * selected at construction via ChatConfig. The active model is stored
 * as a std::variant so each template instantiation retains its full
 * static type through the generate path. Llama instruct models use
 * structured ChatMessage history formatted via MessageFormatter.
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

export module Mila.Chat;
export import Chat.Config;
export import Chat.Message;
export import Chat.MessageFormatter;
export import Chat.SystemPrompt;
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

        void run()
        {
            printWelcome();

            if ( !system_prompt_config_.system_prompt.empty() )
            {
                history_.push_back( { MessageRole::System, system_prompt_config_.system_prompt } );
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
                response.reserve( 512 );

                std::cout << "\nMila: ";

                generateResponse( response );

                std::cout << '\n';

                history_.push_back( { MessageRole::Assistant, response } );
            }
        }

    private:

        /**
         * @brief Format history, tokenize, generate, and stream the response.
         *
         * Llama instruct models render the full structured history via
         * MessageFormatter before tokenization. GPT and Llama base models
         * encode only the last user message content directly.
         *
         * @param response String to accumulate the generated response into.
         */
        void generateResponse( std::string& response )
        {
            std::vector<int32_t> input_tokens = buildInputTokens();

            stop_src_ = std::stop_source{};

            auto fut = std::visit(
                [&]( auto& m )
                {
                    return m->generateAsync(
                        input_tokens,
                        [&]( int32_t tok )
                        {
                            auto text = tokenizer_->decode(
                                std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                            response += text;
                            std::cout << text << std::flush;
                        },
                        config_.max_new_tokens,
                        config_.temperature,
                        config_.top_k,
                        stop_src_.get_token() );
                },
                model_ );

            fut.wait();
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

            auto token_ids = tokenizer_->encode( prompt );

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
                std::cout << "Loading tokenizer from: " << config_.tokenizer_path << "\n";

                switch ( config_.model_type )
                {
                    case ModelType::Gpt:
                        tokenizer_ = BpeTokenizer::loadGpt2( config_.tokenizer_path );
                        break;

                    case ModelType::Llama:
                        tokenizer_ = BpeTokenizer::loadLlama32( config_.tokenizer_path );
                        break;
                }

                std::cout << "Tokenizer loaded. Vocab size: "
                    << tokenizer_->getVocabSize() << "\n";
            }
            catch ( const std::exception& e )
            {
                std::cerr << "Error loading tokenizer: " << e.what() << "\n";
                throw;
            }
        }

        void loadModel()
        {
            std::cout << "Loading model from: " << config_.model_path << "\n";

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
                    if ( config_.precision == ModelPrecision::BF16 )
                    {
                        model_ = LlamaModelBF16Type::fromPretrained(
                            config_.model_path,
                            config_.context_length,
                            device,
                            /*strict=*/true );
                    }
                    else
                    {
                        model_ = LlamaModelFP32Type::fromPretrained(
                            config_.model_path,
                            config_.context_length,
                            device,
                            /*strict=*/true );
                    }
                    break;
            }

            std::visit(
                []( auto& m )
                {
                    std::cout << m->toString();
                    std::cout << m->getMemoryStats().toString() << "\n";
                    std::cout << "Model loaded successfully!\n";
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

        ChatConfig         config_;
        ModelVariant       model_;
        SystemPromptConfig system_prompt_config_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::vector<ChatMessage>      history_;
        std::stop_source              stop_src_;
    };
}