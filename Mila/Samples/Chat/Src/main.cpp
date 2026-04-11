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

// Defaults: Llama 3.2 3B BF16 WIP Bf16 support
// Llama 3.2 1B FP32 is supported and validated
constexpr ModelType      kDefaultModelType = ModelType::Llama;
constexpr ModelSize      kDefaultModelSize = ModelSize::B1;
constexpr ModelPrecision kDefaultPrecision = ModelPrecision::BF16;

static std::filesystem::path gpt2_weights_path()
{
    return std::filesystem::path( MODELS_DIR ) / "gpt2" / "gpt2_small_fp32.bin";
}

static std::filesystem::path llama_weights_path( ModelSize size, ModelPrecision precision )
{
    const char* size_str = (size == ModelSize::B1) ? "1b" : "3b";
    const char* prec_str = (precision == ModelPrecision::BF16) ? "bf16" : "fp32";
    return std::filesystem::path( MODELS_DIR ) / "llama"
        / std::format( "llama32_{}_{}.bin", size_str, prec_str );
}

static void printUsage( const char* prog_name )
{
    std::cerr << "Usage: " << prog_name
        << " [--model-type gpt|llama] [--model-size 1b|3b] [--precision fp32|bf16]"
           " [--tokenizer <path>] [--context-length <n>] [model_path]\n"
        << "  --model-type      Model architecture: gpt or llama. Default: llama.\n"
        << "                    Inferred from model_path if not specified.\n"
        << "  --model-size      Llama parameter count: 1b or 3b. Default: 3b.\n"
        << "                    Inferred from model_path if not specified.\n"
        << "  --precision       Weight dtype: fp32 or bf16. Default: bf16.\n"
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
    ModelType      model_type = kDefaultModelType;
    ModelSize      model_size = kDefaultModelSize;
    ModelPrecision precision  = kDefaultPrecision;
    std::optional<std::filesystem::path> model_path;
    std::optional<std::filesystem::path> tokenizer_path;
    std::optional<std::size_t>           context_length;
    bool explicit_type      = false;
    bool explicit_size      = false;
    bool explicit_precision = false;

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
        else if ( arg == "--model-size" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--model-size requires a value" );
            std::string_view size = argv[ ++i ];

            if ( size == "1b" )
                model_size = ModelSize::B1;
            else if ( size == "3b" )
                model_size = ModelSize::B3;
            else
                throw std::invalid_argument(
                    std::format( "Unknown --model-size: '{}'. Expected 1b or 3b.", size ) );

            explicit_size = true;
        }
        else if ( arg == "--precision" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--precision requires a value" );
            std::string_view prec = argv[ ++i ];

            if ( prec == "fp32" )
                precision = ModelPrecision::FP32;
            else if ( prec == "bf16" )
                precision = ModelPrecision::BF16;
            else
                throw std::invalid_argument(
                    std::format( "Unknown --precision: '{}'. Expected fp32 or bf16.", prec ) );

            explicit_precision = true;
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

    // When a path is provided, infer any unset attributes from it before
    // building defaults — this is the correct ordering so the default path
    // construction below reflects the inferred values.
    if ( model_path )
    {
        std::string lower = model_path->string();
        std::ranges::transform( lower, lower.begin(),
            []( unsigned char c ) { return static_cast<char>(std::tolower( c )); } );

        if ( !explicit_type )
            model_type = lower.find( "llama" ) != std::string::npos
                ? ModelType::Llama : ModelType::Gpt;

        if ( !explicit_size )
            model_size = lower.find( "_1b_" ) != std::string::npos
                ? ModelSize::B1 : ModelSize::B3;

        if ( !explicit_precision )
            precision = lower.find( "bf16" ) != std::string::npos
                ? ModelPrecision::BF16 : ModelPrecision::FP32;
    }

    // Resolve the default path from the (now fully resolved) attributes.
    if ( !model_path )
    {
        model_path = (model_type == ModelType::Gpt)
            ? gpt2_weights_path()
            : llama_weights_path( model_size, precision );
    }

    if ( !tokenizer_path )
    {
        tokenizer_path = (model_type == ModelType::Gpt)
            ? models_dir / "gpt2" / "gpt2_tokenizer.bin"
            : models_dir / "llama" / "llama32_tokenizer.bin";
    }

    ChatConfig config;
    config.model_type = model_type;
    config.model_size = model_size;
    config.precision  = precision;
    config.model_path = std::move( *model_path );
    config.tokenizer_path = std::move( *tokenizer_path );

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