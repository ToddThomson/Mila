#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <charconv>

#ifdef _WIN32
#include <windows.h>
#endif

import Mila;
import Mila.Chat;
import nlohmann.json;

using namespace Mila::ChatApp;

constexpr ModelType        kDefaultModelType        = ModelType::Llama;
constexpr ModelSize        kDefaultModelSize        = ModelSize::B8;
constexpr ModelPrecision   kDefaultPrecision        = ModelPrecision::BF16;
constexpr QuantizationMode kDefaultQuantizationMode = QuantizationMode::FP4;

static std::filesystem::path gpt2_weights_path()
{
    return std::filesystem::path( MODELS_DIR ) / "gpt2" / "gpt2_small_fp32.bin";
}

static std::filesystem::path llama_weights_path( ModelSize size, ModelPrecision precision )
{
    const char* family_str = (size == ModelSize::B8) ? "llama31" : "llama32";
    const char* size_str   = (size == ModelSize::B1) ? "1b"
                           : (size == ModelSize::B8) ? "8b" : "3b";
    const char* prec_str   = (precision == ModelPrecision::BF16) ? "bf16" : "fp32";

    return std::filesystem::path( MODELS_DIR ) / "llama"
        / std::format( "{}_{}_instruct_{}.bin", family_str, size_str, prec_str );
}

static std::filesystem::path gemma_weights_path()
{
    return std::filesystem::path( MODELS_DIR ) / "gemma" / "gemma4_12b_it_bf16.bin";
}

static void printUsage( const char* prog_name )
{
    std::cerr
        << "Usage: " << prog_name
        << " [--config <path>] [--model <alias>] [--quantization none|fp8|fp4]\n"
        << "       [--tokenizer <path>] [--context-length <n>]\n"
        << "       [--system-prompt <path>] [model_path]\n"
        << "\n"
        << "  --config          JSON session config file. CLI args override file values.\n"
        << "  --model           Model alias (recommended). Available aliases:\n"
        << "                      gpt2          GPT-2 small, FP32\n"
        << "                      llama-1b      Llama 3.2 1B Instruct, BF16\n"
        << "                      llama-3b      Llama 3.2 3B Instruct, BF16\n"
        << "                      llama-8b      Llama 3.1 8B Instruct, BF16  (default)\n"
        << "                      llama-1b-fp32 Llama 3.2 1B Instruct, FP32\n"
        << "                      llama-3b-fp32 Llama 3.2 3B Instruct, FP32\n"
        << "                      llama-8b-fp32 Llama 3.1 8B Instruct, FP32\n"
        << "                      gemma-12b     Gemma 4 12B Instruct, BF16 weights, FP4 by default\n"
        << "  --quantization    Weight quantization: none, fp8, or fp4. Default: fp4.\n"
        << "                    fp8 enables FP8 weights and FP8 KV cache compression.\n"
        << "                    fp4 enables INT4 weights (W4A16) and FP8 KV cache compression.\n"
        << "  --tokenizer       Path to the tokenizer file.\n"
        << "  --context-length  Maximum sequence length for inference.\n"
        << "                    Defaults to 1024 for GPT-2/Gemma, 4096 for Llama.\n"
        << "                    Reduce to lower GPU memory usage (Gemma 12B on a 12 GB\n"
        << "                    card may need 512; watch the model-load memory stats).\n"
        << "  --system-prompt   JSON file with system_prompt string and optional tools array.\n"
        << "  model_path        Path to the pretrained weights file (overrides --model path).\n"
        << "\n"
        << "Advanced (use when model_path is provided without --model):\n"
        << "  --model-type      gpt or llama. Inferred from model_path if not specified.\n"
        << "  --model-size      1b or 3b.     Inferred from model_path if not specified.\n"
        << "  --precision       fp32 or bf16. Inferred from model_path if not specified.\n";
}

/**
 * @brief Resolve a short model alias to type/size/precision/is_instruct fields.
 *
 * Sets the four output fields and marks all four explicit flags true on success.
 * Returns false (and leaves all outputs unchanged) for an unrecognised alias.
 */
static bool resolveModelAlias(
    const std::string& alias,
    ModelType& type,      bool& explicit_type,
    ModelSize& size,      bool& explicit_size,
    ModelPrecision& prec, bool& explicit_prec,
    bool& is_instruct,    bool& explicit_instruct )
{
    if ( alias == "gpt2" )
    {
        type = ModelType::Gpt;   size = ModelSize::B3; prec = ModelPrecision::FP32; is_instruct = false;
    }
    else if ( alias == "llama-1b" )
    {
        type = ModelType::Llama; size = ModelSize::B1; prec = ModelPrecision::BF16; is_instruct = true;
    }
    else if ( alias == "llama-3b" )
    {
        type = ModelType::Llama; size = ModelSize::B3; prec = ModelPrecision::BF16; is_instruct = true;
    }
    else if ( alias == "llama-8b" )
    {
        type = ModelType::Llama; size = ModelSize::B8; prec = ModelPrecision::BF16; is_instruct = true;
    }
    else if ( alias == "llama-1b-fp32" )
    {
        type = ModelType::Llama; size = ModelSize::B1; prec = ModelPrecision::FP32; is_instruct = true;
    }
    else if ( alias == "llama-3b-fp32" )
    {
        type = ModelType::Llama; size = ModelSize::B3; prec = ModelPrecision::FP32; is_instruct = true;
    }
    else if ( alias == "llama-8b-fp32" )
    {
        type = ModelType::Llama; size = ModelSize::B8; prec = ModelPrecision::FP32; is_instruct = true;
    }
    else if ( alias == "gemma-12b" )
    {
        type = ModelType::Gemma; size = ModelSize::B12; prec = ModelPrecision::BF16; is_instruct = true;
    }
    else
    {
        return false;
    }

    explicit_type = explicit_size = explicit_prec = explicit_instruct = true;
    return true;
}

/**
 * @brief Apply values from a JSON config file as a baseline for ChatConfig.
 *
 * Explicit CLI arguments applied after this call override any file values.
 * Unknown keys are silently ignored so future fields do not break old files.
 */
static void applyConfigFile(
    const std::filesystem::path& path,
    ModelType& model_type, bool& explicit_type,
    ModelSize& model_size, bool& explicit_size,
    ModelPrecision& precision, bool& explicit_precision,
    QuantizationMode& quantization_mode, bool& explicit_quantization,
    bool& is_instruct, bool& explicit_instruct,
    std::optional<std::filesystem::path>& model_path,
    std::optional<std::filesystem::path>& tokenizer_path,
    std::optional<std::size_t>& context_length,
    std::optional<std::filesystem::path>& system_prompt_path,
    std::optional<std::size_t>& max_new_tokens,
    std::optional<float>& temperature,
    std::optional<int>& top_k )
{
    std::ifstream file( path );

    if ( !file )
    {
        throw std::runtime_error(
            std::format( "--config: cannot open '{}'", path.string() ) );
    }

    nlohmann::json j;

    try
    {
        file >> j;
    }
    catch ( const nlohmann::json::parse_error& e )
    {
        throw std::runtime_error(
            std::format( "--config: JSON parse error in '{}': {}", path.string(), e.what() ) );
    }

    if ( !explicit_type && j.contains( "model_type" ) && j[ "model_type" ].is_string() )
    {
        auto v = j[ "model_type" ].get<std::string>();

        if ( v == "gpt" )
            model_type = ModelType::Gpt;
        else if ( v == "llama" )
            model_type = ModelType::Llama;
        else if ( v == "gemma" )
            model_type = ModelType::Gemma;
    }

    if ( !explicit_size && j.contains( "model_size" ) && j[ "model_size" ].is_string() )
    {
        auto v = j[ "model_size" ].get<std::string>();

        if ( v == "1b" )
            model_size = ModelSize::B1;
        else if ( v == "3b" )
            model_size = ModelSize::B3;
        else if ( v == "8b" )
            model_size = ModelSize::B8;
        else if ( v == "12b" )
            model_size = ModelSize::B12;
    }

    if ( !explicit_precision && j.contains( "precision" ) && j[ "precision" ].is_string() )
    {
        auto v = j[ "precision" ].get<std::string>();

        if ( v == "fp32" )
            precision = ModelPrecision::FP32;
        else if ( v == "bf16" )
            precision = ModelPrecision::BF16;
    }

    if ( !explicit_quantization && j.contains( "quantization" ) && j[ "quantization" ].is_string() )
    {
        auto v = j[ "quantization" ].get<std::string>();

        if ( v == "none" )
            quantization_mode = QuantizationMode::None;
        else if ( v == "fp8" )
            quantization_mode = QuantizationMode::FP8;
        else if ( v == "fp4" )
            quantization_mode = QuantizationMode::FP4;
    }

    if ( !model_path && j.contains( "model_path" ) && j[ "model_path" ].is_string() )
        model_path = j[ "model_path" ].get<std::string>();

    if ( !tokenizer_path && j.contains( "tokenizer_path" ) && j[ "tokenizer_path" ].is_string() )
        tokenizer_path = j[ "tokenizer_path" ].get<std::string>();

    if ( !context_length && j.contains( "context_length" ) && j[ "context_length" ].is_number_unsigned() )
        context_length = j[ "context_length" ].get<std::size_t>();

    if ( !system_prompt_path && j.contains( "system_prompt_path" ) && j[ "system_prompt_path" ].is_string() )
        system_prompt_path = std::filesystem::path( j[ "system_prompt_path" ].get<std::string>() );

    if ( !max_new_tokens && j.contains( "max_new_tokens" ) && j[ "max_new_tokens" ].is_number_unsigned() )
        max_new_tokens = j[ "max_new_tokens" ].get<std::size_t>();

    if ( !temperature && j.contains( "temperature" ) && j[ "temperature" ].is_number() )
        temperature = j[ "temperature" ].get<float>();

    if ( !top_k && j.contains( "top_k" ) && j[ "top_k" ].is_number_integer() )
        top_k = j[ "top_k" ].get<int>();

    if ( !explicit_instruct && j.contains( "is_instruct" ) && j[ "is_instruct" ].is_boolean() )
    {
        is_instruct = j[ "is_instruct" ].get<bool>();
        explicit_instruct = true;
    }
}

static ChatConfig parseArgs( int argc, char* argv[] )
{
    std::filesystem::path models_dir = MODELS_DIR;
    ModelType      model_type = kDefaultModelType;
    ModelSize      model_size = kDefaultModelSize;
    ModelPrecision precision  = kDefaultPrecision;
    std::optional<std::filesystem::path> model_path;
    std::optional<std::filesystem::path> tokenizer_path;
    std::optional<std::filesystem::path> system_prompt_path;
    std::optional<std::size_t> context_length;
    std::optional<std::size_t> max_new_tokens;
    std::optional<float> temperature;
    std::optional<int> top_k;
    QuantizationMode quantization_mode = kDefaultQuantizationMode;
    bool is_instruct           = false;
    bool explicit_type         = false;
    bool explicit_size         = false;
    bool explicit_precision    = false;
    bool explicit_quantization = false;
    bool explicit_instruct     = false;

    // Default config — overridden by --config on the command line.
    std::filesystem::path config_path = "Data/session.json";

    for ( int i = 1; i < argc; ++i )
    {
        std::string_view arg = argv[ i ];

        if ( arg == "--config" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--config requires a value" );
            config_path = argv[ ++i ];
        }
        else if ( arg == "--model" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--model requires a value" );
            std::string alias = argv[ ++i ];
            if ( !resolveModelAlias( alias,
                     model_type, explicit_type,
                     model_size, explicit_size,
                     precision,  explicit_precision,
                     is_instruct, explicit_instruct ) )
                throw std::invalid_argument( std::format(
                    "Unknown --model alias: '{}'. Run with --help to see available aliases.", alias ) );
        }
        else if ( arg == "--model-type" )
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
            else if ( size == "8b" )
                model_size = ModelSize::B8;
            else
                throw std::invalid_argument(
                    std::format( "Unknown --model-size: '{}'. Expected 1b, 3b, or 8b.", size ) );

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
        else if ( arg == "--quantization" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--quantization requires a value" );
            std::string_view qmode = argv[ ++i ];

            if ( qmode == "none" )
                quantization_mode = QuantizationMode::None;
            else if ( qmode == "fp8" )
                quantization_mode = QuantizationMode::FP8;
            else if ( qmode == "fp4" )
                quantization_mode = QuantizationMode::FP4;
            else
                throw std::invalid_argument(
                    std::format( "Unknown --quantization: '{}'. Expected none, fp8, or fp4.", qmode ) );

            explicit_quantization = true;
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
        else if ( arg == "--system-prompt" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--system-prompt requires a value" );
            system_prompt_path = argv[ ++i ];
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

    // Apply config file as baseline before path inference — CLI args already
    // parsed above take precedence via the explicit_* flags.
    if ( std::filesystem::exists( config_path ) )
    {
        applyConfigFile(
            config_path,
            model_type, explicit_type,
            model_size, explicit_size,
            precision, explicit_precision,
            quantization_mode, explicit_quantization,
            is_instruct, explicit_instruct,
            model_path, tokenizer_path,
            context_length, system_prompt_path,
            max_new_tokens, temperature, top_k );
    }

    if ( model_path )
    {
        std::string lower = model_path->string();
        std::ranges::transform( lower, lower.begin(),
            []( unsigned char c ) { return static_cast<char>(std::tolower( c )); } );

        if ( !explicit_type )
        {
            if ( lower.find( "gemma" ) != std::string::npos )
                model_type = ModelType::Gemma;
            else if ( lower.find( "llama" ) != std::string::npos )
                model_type = ModelType::Llama;
            else
                model_type = ModelType::Gpt;
        }

        if ( !explicit_size )
            model_size = lower.find( "_1b_" ) != std::string::npos
                ? ModelSize::B1 : ModelSize::B3;

        if ( !explicit_precision )
            precision = lower.find( "bf16" ) != std::string::npos
                ? ModelPrecision::BF16 : ModelPrecision::FP32;
    }

    if ( !model_path )
    {
        switch ( model_type )
        {
            case ModelType::Gpt:   model_path = gpt2_weights_path(); break;
            case ModelType::Gemma: model_path = gemma_weights_path(); break;
            case ModelType::Llama: model_path = llama_weights_path( model_size, precision ); break;
        }
    }

    if ( !explicit_instruct )
    {
        std::string lower = model_path->string();
        std::ranges::transform( lower, lower.begin(),
            []( unsigned char c ) { return static_cast<char>(std::tolower( c )); } );
        // Gemma marks instruct-tuned checkpoints with "_it_" rather than "instruct".
        is_instruct = lower.find( "instruct" ) != std::string::npos
                   || lower.find( "_it_" ) != std::string::npos;
    }

    if ( !tokenizer_path )
    {
        switch ( model_type )
        {
            case ModelType::Gpt:   tokenizer_path = models_dir / "gpt2" / "gpt2_tokenizer.bin"; break;
            case ModelType::Gemma: tokenizer_path = models_dir / "gemma" / "gemma_tokenizer.bin"; break;
            case ModelType::Llama: tokenizer_path = models_dir / "llama" / "llama32_tokenizer.bin"; break;
        }
    }

    ChatConfig config;
    config.model_type         = model_type;
    config.model_size         = model_size;
    config.precision          = precision;
    config.quantization_mode  = quantization_mode;
    config.is_instruct        = is_instruct;
    config.models_dir         = models_dir;
    config.model_path         = std::move( *model_path );
    config.tokenizer_path     = std::move( *tokenizer_path );
    config.config_path        = config_path;
    config.system_prompt_path = system_prompt_path;

    if ( context_length.has_value() )
    {
        config.context_length = *context_length;
    }
    else
    {
        config.context_length = defaultContextLength( model_type );
    }

    if ( max_new_tokens.has_value() )
        config.max_new_tokens = *max_new_tokens;

    if ( temperature.has_value() )
        config.temperature = *temperature;

    if ( top_k.has_value() )
        config.top_k = *top_k;

    return config;
}

int main( int argc, char* argv[] )
{

#ifdef _WIN32
    SetConsoleOutputCP( CP_UTF8 );
    SetConsoleCP( CP_UTF8 );
    HANDLE hOut = GetStdHandle( STD_OUTPUT_HANDLE );
    if ( hOut != INVALID_HANDLE_VALUE )
    {
        DWORD mode = 0;
        if ( GetConsoleMode( hOut, &mode ) )
            SetConsoleMode( hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING );
    }
#endif

    auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Info );
    Mila::initialize( 0, std::move( sink ) );

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

        if ( config.system_prompt_path.has_value() &&
             !std::filesystem::exists( *config.system_prompt_path ) )
        {
            std::cerr << "Error: System prompt file not found: "
                      << *config.system_prompt_path << "\n";
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