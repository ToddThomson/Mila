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

// Helper: path to GPT-2 weights under MODELS_DIR (set via CMake target_compile_definitions)
static std::filesystem::path gpt2_weights_path()
{
    std::filesystem::path models_dir = MODELS_DIR;
    return models_dir / "gpt2" / "gpt2_small_fp32.bin";
}

static void printUsage( const char* prog_name )
{
    std::cerr << "Usage: " << prog_name
        << " [--model-type gpt|llama] [--tokenizer <path>] [model_path]\n"
        << "  --model-type  Model architecture: gpt or llama.\n"
        << "                Inferred from model_path if not specified.\n"
        << "  --tokenizer   Path to the tokenizer file.\n"
        << "  model_path    Path to the pretrained weights file.\n";
}

static ChatConfig parseArgs( int argc, char* argv[] )
{
    std::filesystem::path models_dir = MODELS_DIR;

    std::filesystem::path                model_path     = models_dir / "gpt2" / "gpt2_small_fp32.bin";
    std::optional<std::filesystem::path> tokenizer_path;
    std::optional<ModelType>             explicit_type;

    for ( int i = 1; i < argc; ++i )
    {
        std::string_view arg = argv[ i ];

        if ( arg == "--model-type" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--model-type requires a value" );

            std::string_view type = argv[ ++i ];

            if ( type == "gpt" )        explicit_type = ModelType::Gpt;
            else if ( type == "llama" ) explicit_type = ModelType::Llama;
            else throw std::invalid_argument(
                std::format( "Unknown --model-type: '{}'. Expected gpt or llama.", type ) );
        }
        else if ( arg == "--tokenizer" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--tokenizer requires a value" );

            tokenizer_path = argv[ ++i ];
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

    // Infer backend from the model path when --model-type is not given
    if ( !explicit_type )
    {
        std::string lower = model_path.string();
        std::ranges::transform( lower, lower.begin(),
            []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

        explicit_type = lower.find( "llama" ) != std::string::npos
            ? ModelType::Llama
            : ModelType::Gpt;
    }

    // Default tokenizer path keyed to the resolved backend
    if ( !tokenizer_path )
    {
        tokenizer_path = ( *explicit_type == ModelType::Gpt )
            ? models_dir / "gpt2"  / "gpt2_tokenizer.bin"
            : models_dir / "llama" / "llama_tokenizer.bin";
    }

    ChatConfig config;
    config.model_type     = *explicit_type;
    config.model_path     = std::move( model_path );
    config.tokenizer_path = std::move( *tokenizer_path );

    return config;
}

int main( int argc, char* argv[] )
{
    Mila::initialize();

    try
    {
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
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}