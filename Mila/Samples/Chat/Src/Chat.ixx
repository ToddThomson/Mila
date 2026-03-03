/**
 * @file Chat.ixx
 * @brief Mila chat application.
 *
 * Uses GptModel — the inference-only wrapper around a loaded GptTransformer —
 * rather than GptTransformer directly. GptModel owns fromPretrained() and
 * generate(), keeping inference concerns out of the Network layer.
 */

module;
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>

export module Mila.Chat;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;
    using namespace Mila::Data;

    // GptModel — inference-only, owns fromPretrained() and generate()
    using GptModelType = GptModel<DeviceType::Cuda, TensorDataType::FP32>;

    static std::filesystem::path gpt2_tokenizer_path()
    {
        std::filesystem::path models_path = MODELS_DIR;
        return models_path / "gpt2" / "gpt2_tokenizer.bin";
    }

    export class Chat
    {
    public:

        explicit Chat( const std::filesystem::path& model_path )
            : model_path_( model_path )
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
                std::cout << "Loading tokenizer from: " << gpt2_tokenizer_path() << "\n";
                tokenizer_ = BpeTokenizer::loadGpt2( gpt2_tokenizer_path() );
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
            try
            {
                std::cout << "Loading model from: " << model_path_ << "\n";

                model_ = GptModelType::fromPretrained(
                    model_path_,
                    DeviceId{ DeviceType::Cuda, 0 },
                    /*strict=*/true );

                std::cout << model_->toString();
                std::cout << "Model loaded successfully!\n";
            }
            catch ( const std::exception& e )
            {
                std::cerr << "Error loading model: " << e.what() << "\n";
                throw;
            }
        }

        std::string generateResponse( const std::vector<std::string>& history )
        {
            try
            {
                if ( !model_ || !tokenizer_ )
                    return "Model not loaded.";

                // For validation: use raw user input directly, no chat template
                // GPT-2 is a completion model — preparePrompt() adds instruction-following
                // format that base GPT-2 was never trained on
                const std::string& prompt = history.back().substr( 6 );  // strip "User: " prefix

                std::vector<TokenId> prompt_tokens = tokenizer_->encode( prompt );

                std::vector<int32_t> generated = model_->generate(
                    std::vector<int32_t>( prompt_tokens.begin(), prompt_tokens.end() ),
                    /*max_new_tokens=*/512,
                    /*temperature=*/0.8f,
                    /*top_k=*/ 40 );

                std::string full_text = tokenizer_->decode(
                    std::vector<TokenId>( generated.begin(), generated.end() ) );

                return extractResponse( full_text, prompt );
            }
            catch ( const std::exception& e )
            {
                return "Error: " + std::string( e.what() );
            }
        }

        std::string preparePrompt( const std::vector<std::string>& history ) const
        {
            std::ostringstream ss;
            ss << "You are a helpful AI assistant.\n\n";

            // Keep last 10 exchanges to manage context length
            size_t start = history.size() > 10 ? history.size() - 10 : 0;
            for ( size_t i = start; i < history.size(); ++i )
                ss << history[ i ] << "\n";

            ss << "Assistant:";
            return ss.str();
        }

        std::string extractResponse(
            const std::string& full_output,
            const std::string& prompt ) const
        {
            if ( full_output.size() <= prompt.size() )
                return full_output;

            std::string response = full_output.substr( prompt.size() );

            // Trim leading whitespace
            auto start = response.find_first_not_of( " \t\n\r" );
            if ( start != std::string::npos )
                response = response.substr( start );

            // Stop at double newline
            auto end = response.find( "\n\n" );
            if ( end != std::string::npos )
                response = response.substr( 0, end );

            return response;
        }

        void printWelcome() const
        {
            std::cout << R"(
+--------------------------------------+
|         Mila Chat CLI v1.0           |
|      Powered by Mila DNN Library     |
+--------------------------------------+

Type 'help' for commands, 'exit' to quit.
)" << "\n";
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

        std::filesystem::path model_path_;
        std::unique_ptr<GptModelType> model_{ nullptr };
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
    };
}