#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <cctype>

import Mila;
import Mila.Chat;

using namespace Mila::ChatApp;

constexpr ModelType kDefaultModelType = ModelType::Gpt;

static std::filesystem::path gpt2_weights_path()
{
    return std::filesystem::path( MODELS_DIR ) / "gpt2" / "gpt2_small_fp32.bin";
}

static std::filesystem::path llama_weights_path()
{
    return std::filesystem::path( MODELS_DIR ) / "llama" / "llama32_1b_fp32.bin";
}

static void printUsage( const char* prog_name )
{
    std::cerr << "Usage: " << prog_name
        << " [--model-type gpt|llama] [--tokenizer <path>] [--context-length <n>] [model_path]\n"
        << "  --model-type      Model architecture: gpt or llama.\n"
        << "                    Inferred from model_path if not specified.\n"
        << "  --tokenizer       Path to the tokenizer file.\n"
        << "  --context-length  Maximum sequence length for inference.\n"
        << "                    Defaults to 1024 for GPT-2, 4096 for Llama.\n"
        << "                    Reduce to lower GPU memory usage.\n"
        << "                    Cannot exceed the model architectural maximum.\n"
        << "  model_path        Path to the pretrained weights file.\n";
}

static ChatConfig parseArgs( int argc, char* argv[] )
{
    std::filesystem::path models_dir = MODELS_DIR;
    ModelType             model_type = kDefaultModelType;
    std::optional<std::filesystem::path> model_path;
    std::optional<std::filesystem::path> tokenizer_path;
    std::optional<std::size_t>           context_length;
    bool explicit_type = false;

    for ( int i = 1; i < argc; ++i )
    {
        std::string_view arg = argv[ i ];

        if ( arg == "--model-type" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--model-type requires a value" );
            std::string_view type = argv[ ++i ];
            if ( type == "gpt" )
                model_type = ModelType::Gpt;
            else if ( type == "llama" )
                model_type = ModelType::Llama;
            else
                throw std::invalid_argument(
                    std::format( "Unknown --model-type: '{}'. Expected gpt or llama.", type ) );
            explicit_type = true;
        }
        else if ( arg == "--tokenizer" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--tokenizer requires a value" );
            tokenizer_path = argv[ ++i ];
        }
        else if ( arg == "--context-length" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--context-length requires a value" );
            std::string_view val = argv[ ++i ];
            std::size_t n = 0;
            auto result = std::from_chars( val.data(), val.data() + val.size(), n );
            if ( result.ec != std::errc{} || n == 0 )
                throw std::invalid_argument( std::format(
                    "--context-length must be a positive integer, got '{}'", val ) );
            context_length = n;
        }
        else if ( !arg.starts_with( "--" ) )
        {
            model_path = arg;
        }
        else
        {
            throw std::invalid_argument( std::format( "Unknown option: '{}'", arg ) );
        }
    }

    // Infer model type from path if not explicitly set.
    if ( !explicit_type && model_path )
    {
        std::string lower = model_path->string();
        std::ranges::transform( lower, lower.begin(),
            []( unsigned char c ) { return static_cast<char>(std::tolower( c )); } );
        model_type = lower.find( "llama" ) != std::string::npos
            ? ModelType::Llama
            : ModelType::Gpt;
    }

    if ( !model_path )
    {
        model_path = (model_type == ModelType::Gpt)
            ? gpt2_weights_path()
            : llama_weights_path();
    }

    if ( !tokenizer_path )
    {
        tokenizer_path = (model_type == ModelType::Gpt)
            ? models_dir / "gpt2" / "gpt2_tokenizer.bin"
            : models_dir / "llama" / "llama32_tokenizer.bin";
    }

    ChatConfig config;
    config.model_type = model_type;
    config.model_path = std::move( *model_path );
    config.tokenizer_path = std::move( *tokenizer_path );

    // Resolve context_length to a model-type-aware default if not
    // explicitly provided by the user.
    if ( context_length.has_value() )
    {
        config.context_length = *context_length;
    }
    else
    {
        config.context_length = (model_type == ModelType::Gpt)
            ? 1024   // GPT-2 architectural maximum
            : 4096;  // Llama consumer GPU safe default
    }

    return config;
}

int main( int argc, char* argv[] )
{
    Mila::initialize();

    //try
    //{
        ChatConfig config = parseArgs( argc, argv );

        if ( !std::filesystem::exists( config.model_path ) )
        {
            std::cerr << "Error: Model file not found: " << config.model_path << "\n";
            printUsage( argv[ 0 ] );
            return 1;
        }

        if ( !std::filesystem::exists( config.tokenizer_path ) )
        {
            std::cerr << "Error: Tokenizer file not found: " << config.tokenizer_path << "\n";
            printUsage( argv[ 0 ] );
            return 1;
        }

        Chat chat( std::move( config ) );
        chat.run();

        return 0;
    //}
    //catch ( const std::exception& e )
    //{
    //    std::cerr << "Fatal error: " << e.what() << "\n";
    //    return 1;
    //}
}