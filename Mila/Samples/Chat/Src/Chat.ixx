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

        std::string generateResponse_old( const std::vector<std::string>& history )
        {
            try
            {
                // Prepare the prompt from conversation history
                std::string prompt = preparePrompt( history );

                // DEBUG Simplify for testing
                //prompt = "You are a helpful AI Assistant. Your name is Mila";
                prompt = "Once upon a time";

                // If transformer is not loaded, return fallback text
                if ( !transformer_ )
                {
                    return "Model not loaded.";
                }

                if ( !tokenizer_ )
                {
                    return "Tokenizer not loaded.";
                }

                // Encode prompt
                std::vector<TokenId> prompt_tokens = tokenizer_->encode( prompt );

                // DEBUG: Print token IDs
                Utils::Logger::info( std::format( "Input prompt: '{}'", prompt ) );
                Utils::Logger::info( std::format( "Token count: {}", prompt_tokens.size() ) );
                Utils::Logger::info( "Token IDs:" );
                for ( size_t i = 0; i < prompt_tokens.size(); ++i )
                {
                    Utils::Logger::info( std::format( "  [{}]: {}", i, prompt_tokens[ i ] ) );
                }

                // Check if any tokens are out of range
                size_t vocab_size = tokenizer_->getVocabSize();
                Utils::Logger::info( std::format( "Vocab size: {}", vocab_size ) );
                for ( auto id : prompt_tokens )
                {
                    if ( id >= vocab_size )
                    {
                        Utils::Logger::error( std::format( "ERROR: Token ID {} is out of range (vocab_size={})",
                            id, vocab_size ) );
                    }
                }

                // Generation params (adjustable)
                const int64_t model_batch_size = 1;
                const int64_t model_seq_length = 1024; // safe default window (must be <= model max seq length)
                const size_t max_new_tokens = 64;
                const float temperature = 0.8f;

                // RNG for sampling
                std::mt19937 rng( static_cast<unsigned int>( std::chrono::high_resolution_clock::now().time_since_epoch().count() ) );

                // Vocab size from tokenizer
                //size_t vocab_size = tokenizer_->getVocabSize();

                // Working buffers (device + cpu)
                using DeviceMR = typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource;

                shape_t model_shape = { model_batch_size, model_seq_length };
                shape_t logits_shape = { model_batch_size, model_seq_length, static_cast<int64_t>(vocab_size) };

                Tensor<TensorDataType::INT32, DeviceMR> context_device( DeviceId{ DeviceType::Cuda, 0 }, model_shape );
                Tensor<TensorDataType::FP32, DeviceMR> logits_device( DeviceId{ DeviceType::Cuda, 0 }, logits_shape );
                Tensor<TensorDataType::FP32, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
                Tensor<TensorDataType::INT32, CpuMemoryResource> context_cpu( Device::Cpu(), model_shape );

                int32_t pad_value = 50256;  // GPT-2's <|endoftext|> token
                auto eos_opt = tokenizer_->getEosTokenId();
                if ( eos_opt )
                    pad_value = static_cast<int32_t>(*eos_opt);

                // Start generation with prompt tokens
                std::vector<TokenId> generated_tokens = prompt_tokens;

                // Helper: sample from logits with temperature
                auto sampleFromLogits = [&]( const float* logits_ptr, size_t vsize, float temp ) -> int32_t {
                    // Numerically stable softmax sampling
                    float max_logit = -std::numeric_limits<float>::infinity();
                    for ( size_t v = 0; v < vsize; ++v )
                        max_logit = std::max( max_logit, logits_ptr[v] );

                    std::vector<float> probs(vsize);
                    double sum = 0.0;
                    for ( size_t v = 0; v < vsize; ++v ) {
                        float scaled = (logits_ptr[v] - max_logit) / temp;
                        float e = std::exp( scaled );
                        probs[v] = e;
                        sum += e;
                    }

                    if ( sum <= 0.0 ) {
                        // fallback to argmax
                        size_t best = 0;
                        for ( size_t v = 1; v < vsize; ++v )
                            if ( logits_ptr[v] > logits_ptr[best] ) best = v;
                        return static_cast<int32_t>( best );
                    }

                    for ( size_t v = 0; v < vsize; ++v ) probs[v] = static_cast<float>( probs[v] / sum );

                    std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
                    float r = dist( rng );
                    float cumsum = 0.0f;
                    for ( size_t v = 0; v < vsize; ++v ) {
                        cumsum += probs[v];
                        if ( r < cumsum ) return static_cast<int32_t>( v );
                    }

                    return static_cast<int32_t>( vsize - 1 );
                };

                // Temporarily ensure model is in inference mode
                bool was_training = transformer_->isTraining();
                transformer_->setTraining( false );

                for ( size_t step = 0; step < max_new_tokens; ++step )
                {
                    int64_t context_start = std::max<int64_t>( 0, static_cast<int64_t>( generated_tokens.size() ) - model_seq_length );
                    int64_t actual_context_len = static_cast<int64_t>( generated_tokens.size() ) - context_start;

                    // Fill CPU context with pad
                    std::fill_n( context_cpu.data(), static_cast<size_t>( model_batch_size * model_seq_length ), pad_value );

                    // Copy last tokens into context_cpu (batch 0)
                    for ( int64_t j = 0; j < actual_context_len; ++j )
                    {
                        context_cpu.data()[ static_cast<size_t>( j ) ] = static_cast<int32_t>( generated_tokens[ static_cast<size_t>( context_start + j ) ] );
                    }

                    // Transfer to device
                    copy( context_cpu, context_device );

                    // Forward pass
                    auto& logits_ref = transformer_->forward( context_device );
                    transformer_->synchronize();

                    // Copy logits back to CPU
                    copy( logits_ref, logits_cpu );

                    // Determine last real position in sequence
                    int64_t last_real_pos = ( actual_context_len > 0 ) ? ( actual_context_len - 1 ) : ( model_seq_length - 1 );
                    if ( last_real_pos < 0 ) last_real_pos = 0;

                    size_t batch_0_last_token_offset = static_cast<size_t>( last_real_pos ) * vocab_size;

                    // Sample next token
                    const float* logits_ptr = logits_cpu.data() + batch_0_last_token_offset;
                    int32_t next_token = sampleFromLogits( logits_ptr, vocab_size, temperature );

                    generated_tokens.push_back( static_cast<TokenId>( next_token ) );

                    // Stop if EOS token encountered (if tokenizer exposes it)
                    auto eos_opt = tokenizer_->getEosTokenId();
                    if ( eos_opt && static_cast<int32_t>( next_token ) == static_cast<int32_t>( *eos_opt ) )
                    {
                        break;
                    }
                }

                // Restore training state
                transformer_->setTraining( was_training );

                // Decode generated token ids back to text using tokenizer
                std::string full_text = tokenizer_->decode( std::span<const TokenId>( generated_tokens.data(), generated_tokens.size() ) );

                // Extract assistant response portion (remove prompt)
                std::string response = extractResponse( full_text, prompt );

                return response;
            }
            catch ( const std::exception& e )
            {
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

        std::filesystem::path model_path_;
        std::shared_ptr<GptType> transformer_{ nullptr };
        std::shared_ptr<BpeTokenizer> tokenizer_{ nullptr };
    };
}