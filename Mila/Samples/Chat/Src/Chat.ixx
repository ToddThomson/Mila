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

export module Mila.Chat;
export import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;
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

                std::string response = generateResponse( conversation_history );

                conversation_history.push_back( "Mila: " + response );

                std::cout << "\nMila: " << response << "\n";
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
            //try
            //{
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
            //}
            //catch ( const std::exception& e )
            //{
            //    std::cerr << "Error loading model: " << e.what() << "\n";
            //    throw;
            //}
        }

        std::string generateResponse( const std::vector<std::string>& history )
        {
            try
            {
                if ( !tokenizer_ )
                    return "Tokenizer not loaded.";

                // Both GPT-2 and LLaMA base models are completion models; pass the raw
                // user text without a chat template to avoid instruction-format mismatch.
                const std::string& prompt = history.back().substr( 6 );  // strip "User: "

                std::vector<TokenId> prompt_tokens = tokenizer_->encode( prompt );

                std::vector<int32_t> input_tokens( prompt_tokens.begin(), prompt_tokens.end() );

                std::vector<int32_t> generated = model_->generate(
                    std::vector<int32_t>( input_tokens ),
                    config_.max_new_tokens,
                    config_.temperature,
                    config_.top_k );

                std::string full_text = tokenizer_->decode( std::vector<TokenId>( generated.begin(), generated.end() ) );

                return extractResponse( full_text, prompt );
            }
            catch ( const std::exception& e )
            {
                return "Error: " + std::string( e.what() );
            }
        }

        std::string extractResponse(
            const std::string& full_output,
            const std::string& prompt ) const
        {
            if ( full_output.size() <= prompt.size() )
                return full_output;

            std::string response = full_output.substr( prompt.size() );

            auto start = response.find_first_not_of( " \t\n\r" );
            if ( start != std::string::npos )
                response = response.substr( start );

            auto end = response.find( "\n\n" );
            if ( end != std::string::npos )
                response = response.substr( 0, end );

            return response;
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
    };
}