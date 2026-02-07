module;
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <sstream>

export module Chat;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;

    export class Chat 
    {
    public:
        
        Chat( const std::string& modelPath )
            : modelPath_( modelPath ) {
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

                // Generate response
                std::string response = generateResponse( conversationHistory );

                // Add assistant response to history
                conversationHistory.push_back( "Mila: " + response );

                std::cout << "\nMila: " << response << "\n";
            }
        }

    private:

        void loadModel() {
            try {
                std::cout << "Loading model from: " << modelPath_ << "\n";

                // Use the GPT-2 Small preset (preferred over manual config)
                GptConfig config = GPT2_Small();

                // Construct a device/precision specific transformer instance
                transformer_ = std::make_unique<GptTransformer<DeviceType::Cuda, TensorDataType::FP32>>(
                    "gpt2_small",
                    config,
                    DeviceId{ DeviceType::Cuda, 0 }
                );

                // Build transformer with batch size 1 and preset max sequence length
                // FIXME: transformer_->build( shape_t{ 1, static_cast<size_t>(config.getMaxSequenceLength()) } );

                // Load pre-trained weights
                // FIXME: transformer_->load( modelPath_ );

                // Set to inference mode
                // FIXME: transformer_->eval();

                std::cout << "Model loaded successfully!\n";
                // FIXME:: std::cout << "Parameters: " << transformer_->num_parameters() << "\n";
            }
            catch ( const std::exception& e ) {
                std::cerr << "Error loading model: " << e.what() << "\n";
                throw;
            }
        }

        std::string generateResponse( const std::vector<std::string>& history ) {
            try {
                // Prepare the prompt from conversation history
                std::string prompt = preparePrompt( history );

                // Tokenize the input
                //auto tokens = tokenizer_.encode( prompt );

                // Generate response tokens
                /*Mila::GenerationConfig genConfig{
                    .max_new_tokens = 256,
                    .temperature = 0.8f,
                    .top_k = 40,
                    .top_p = 0.95f,
                    .do_sample = true
                };*/

                //auto outputTokens = transformer_->generate( tokens, genConfig );

                // Decode tokens to text
                //std::string response = tokenizer_.decode( outputTokens );

                // Extract just the assistant's response (remove the prompt)
                // response = extractResponse( response, prompt );

                return "this is all there is!";// response;
            }
            catch ( const std::exception& e ) {
                return "Error generating response: " + std::string( e.what() );
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

        std::string modelPath_;
        std::unique_ptr<GptTransformer<DeviceType::Cuda, TensorDataType::FP32>> transformer_;
        //GptTokenizer tokenizer_;
    };
}