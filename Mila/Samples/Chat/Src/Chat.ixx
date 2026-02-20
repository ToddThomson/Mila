module;
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <random>

export module Chat;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;
    using namespace Mila::Data;

    using GptType = GptTransformer<DeviceType::Cuda, TensorDataType::FP32>;

    // Helper: path to GPT-2 tokenizer under TEST_DATA_DIR (set via CMake target_compile_definitions)
    static std::filesystem::path gpt2_tokenizer_path()
    {
        std::filesystem::path models_path = MODELS_DIR;
        return models_path / "gpt2" / "gpt2_tokenizer.bin";
    }

    export class Chat 
    {
    public:
        
        Chat( const std::filesystem::path model_path )
            : model_path_( model_path )
        {
            initializeTokenizer();
            loadModel();
        }

        void run() {
            printWelcome();

            std::string userInput;
            std::vector<std::string> conversationHistory;

            while ( true ) {
                std::cout << "\nYou: ";
                getline( std::cin, userInput );

                if ( userInput.empty() ) {
                    continue;
                }

                if ( userInput == "exit" || userInput == "quit" ) {
                    std::cout << "Goodbye!\n";
                    break;
                }

                if ( userInput == "clear" ) {
                    conversationHistory.clear();
                    std::cout << "Conversation history cleared.\n";
                    continue;
                }

                if ( userInput == "help" ) {
                    printHelp();
                    continue;
                }

                // Add user input to history
                conversationHistory.push_back( "User: " + userInput );

                std::string response = generateResponse( conversationHistory );

                conversationHistory.push_back( "Mila: " + response );

                std::cout << "\nMila: " << response << "\n";
            }
        }

    private:
        
        void initializeTokenizer() {
            try {
                std::cout << "Loading tokenizer from: " << gpt2_tokenizer_path() << "\n";
                tokenizer_ = BpeTokenizer::loadGpt2( gpt2_tokenizer_path() );

                std::cout << "Tokenizer loaded successfully!\n";

                // DEBUG: Verify vocab loaded correctly
                std::cout << "Vocab size: " << tokenizer_->getVocabSize() << "\n";
            }
            catch ( const std::exception& e ) {
                std::cerr << "Error loading tokenizer: " << e.what() << "\n";
                throw;
            }
        }

        void loadModel() {
            try {
                std::cout << "Loading model from: " << model_path_ << "\n";

                // Create transformer from pretrained archive.
                // Use batch_size=1 and device CUDA:0 by default.
                transformer_ = GptType::fromPretrained(
                    model_path_,
                    /*batch_size=*/1,
                    /*seq_length=*/1024,
                    DeviceId{ DeviceType::Cuda, 0 },
                    /*strict=*/true
                );

                // Ensure network is in inference mode (no gradients).
                if ( transformer_ ) {
                    transformer_->setTraining( false );

                    // Optional: print parameter count if available
                    try {
                        std::cout << "Parameters: " << transformer_->parameterCount() << "\n";
                    }
                    catch ( const std::exception& ) {
                        // Ignore if parameterCount is not available or throws
                    }
                }
                else {
                    std::cerr << "Warning: transformer creation returned null.\n";
                }

                std::cout << "Model loaded successfully!\n";
            }
            catch ( const std::exception& e ) {
                std::cerr << "Error loading model: " << e.what() << "\n";
                throw;
            }
        }

        std::string generateResponse( const std::vector<std::string>& history )
        {
            try
            {
                std::string prompt = "Once upon a time";

                if ( !transformer_ || !tokenizer_ )
                {
                    return "Model not loaded.";
                }

                // Encode prompt
                std::vector<TokenId> prompt_tokens = tokenizer_->encode( prompt );

                Utils::Logger::info( std::format( "Generating from {} tokens...", prompt_tokens.size() ) );

                // Simple generation call with dynamic sequence lengths!
                std::vector<int32_t> generated = transformer_->generate(
                    std::vector<int32_t>( prompt_tokens.begin(), prompt_tokens.end() ),
                    64,    // max_new_tokens
                    0.8f,  // temperature
                    40     // top_k
                );

                // Decode
                std::vector<TokenId> token_ids( generated.begin(), generated.end() );
                std::string full_text = tokenizer_->decode( token_ids );

                return full_text;
            }
            catch ( const std::exception& e )
            {
                return "Error: " + std::string( e.what() );
            }
        }

        std::string preparePrompt( const std::vector<std::string>& history ) const {
            std::stringstream ss;

            // System prompt
            ss << "You are a helpful AI assistant.\n\n";

            // Add recent history (keep last 10 exchanges to manage context length)
            size_t start = history.size() > 10 ? history.size() - 10 : 0;
            for ( size_t i = start; i < history.size(); ++i ) {
                ss << history[ i ] << "\n";
            }

            ss << "Assistant:";

            return ss.str();
        }

        std::string extractResponse( const std::string& fullOutput, const std::string& prompt ) const {
            // Remove the prompt from the generated output
            if ( fullOutput.size() > prompt.size() ) {
                std::string response = fullOutput.substr( prompt.size() );

                // Trim whitespace
                auto start = response.find_first_not_of( " \t\n\r" );
                if ( start != std::string::npos ) {
                    response = response.substr( start );
                }

                // Stop at double newline or special tokens
                auto end = response.find( "\n\n" );
                if ( end != std::string::npos ) {
                    response = response.substr( 0, end );
                }

                return response;
            }

            return fullOutput;
        }

        void printWelcome() const {
            std::cout << R"(
                ╔══════════════════════════════════════╗
                ║         Mila Chat CLI v1.0          ║
                ║   Powered by Mila DNN Library       ║
                ╚══════════════════════════════════════╝

                Type 'help' for commands, 'exit' to quit.
                )" << "\n";
        }

        void printHelp() const {
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
        std::shared_ptr<GptType> transformer_{ nullptr };
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
    };
}