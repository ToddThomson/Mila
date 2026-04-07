/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Supports GptModel and LlamaModel backends, selected at construction via ChatConfig.
 */

module;
#include <iostream>
#include <string>
#include <vector>
#include <variant>
#include <sstream>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <future>
#include <stop_token>

export module Mila.Chat;
export import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;

    using LanguageModelType = LanguageModel<DeviceType::Cuda, TensorDataType::FP32>;

    export class Chat
    {
    public:

        /**
         * @brief Construct a Chat session from a fully-populated ChatConfig.
         *
         * Loads the tokenizer and model on construction; throws on any failure.
         *
         * @param config Session configuration (model type, paths, generation params).
         * @throws std::runtime_error on tokenizer or model load failure.
         */
        explicit Chat( ChatConfig config )
            : config_( std::move( config ) )
        {
            initializeTokenizer();
            loadModel();
        }

        void run()
        {
            printWelcome();

            std::string user_input;
            std::vector<std::string> conversation_history;

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
                    conversation_history.clear();
                    std::cout << "Conversation history cleared.\n";
                    continue;
                }

                if ( user_input == "help" )
                {
                    printHelp();
                    continue;
                }

                conversation_history.push_back( "User: " + user_input );

                const std::string& prompt = conversation_history.back().substr( 6 );
                std::vector<TokenId> prompt_tokens = tokenizer_->encode( prompt );
                std::vector<int32_t> input_tokens( prompt_tokens.begin(), prompt_tokens.end() );

                std::string response;
                response.reserve( 512 );

                std::cout << "\nMila: ";

                stop_src_ = std::stop_source{};

                auto fut = model_->generateAsync(
                    input_tokens,
                    [&]( int32_t tok )
                    {
                        auto text = tokenizer_->decode( std::vector<TokenId>{ static_cast<TokenId>(tok) } );
                        response += text;
                        std::cout << text << std::flush;
                    },
                    config_.max_new_tokens,
                    config_.temperature,
                    config_.top_k,
                    stop_src_.get_token() );

                fut.wait();
                std::cout << '\n';

                conversation_history.push_back( "Mila: " + trimResponse( response ) );
            }
        }

    private:

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

            switch ( config_.model_type )
            {
                case ModelType::Gpt:
                    model_ = GptModel<DeviceType::Cuda, TensorDataType::FP32>::fromPretrained(
                        config_.model_path,
                        config_.context_length,
                        DeviceId{ DeviceType::Cuda, 0 },
                        /*strict=*/true );
                    break;

                case ModelType::Llama:
                    model_ = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>::fromPretrained(
                        config_.model_path,
                        config_.context_length,
                        DeviceId{ DeviceType::Cuda, 0 },
                        /*strict=*/true );
                    break;
            }

            std::cout << model_->toString();

            auto stats = model_->getMemoryStats();
            std::cout << stats.toString() << "\n";

            std::cout << "Model loaded successfully!\n";
        }

        /**
         * @brief Strip leading whitespace and truncate at the first paragraph break.
         *
         * Applied to the accumulated streaming response before storing in history.
         * The live printed output is unaffected.
         */
        std::string trimResponse( const std::string& raw ) const
        {
            auto start = raw.find_first_not_of( " \t\n\r" );

            if ( start == std::string::npos )
                return {};

            std::string result = raw.substr( start );

            auto end = result.find( "\n\n" );

            if ( end != std::string::npos )
                result.resize( end );

            return result;
        }

        void printWelcome() const
        {
            const char* backend = (config_.model_type == ModelType::Gpt) ? "GPT" : "LLaMA";

            std::cout << R"(
+--------------------------------------+
|         Mila Chat CLI v1.0           |
|      Powered by Mila DNN Library     |
+--------------------------------------+

Type 'help' for commands, 'exit' to quit.
)" << "\n";

            std::cout << "Backend: " << backend << "\n";
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
        std::unique_ptr<LanguageModelType> model_;
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
        std::stop_source stop_src_;
    };
}