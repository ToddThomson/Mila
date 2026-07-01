/**
 * @file ProfileModel.ixx
 * @brief General-purpose, non-interactive model profiling harness for Nsight.
 *
 * Loads a Llama model and exercises one phase of the inference path under a
 * cudaProfilerApi capture region and an NVTX range, so Nsight Systems can be
 * run with --capture-range=cudaProfilerApi to profile only the measured region
 * (excluding model load and warmup). Greedy decode keeps runs repeatable.
 *
 * Phases:
 *   prefill   profilePrefill() only (prompt forward pass + sync).
 *   decode    full generate(); decode loop dominates a long generation.
 *   generate  full generate(); alias of decode with a separate label.
 *
 * The public API exposes generate() (prefill + decode together) and
 * profilePrefill(), but not a decode-only entry point, so the decode and
 * generate phases both run the full generate path. generate() streams tokens
 * through a callback and returns only a finish reason; the profiler measures
 * prefill (call -> first token) and decode (first -> last token) timing from
 * the callback cadence.
 *
 * All Mila template instantiation (model loading via fromPretrained) is confined
 * to this module interface unit. See [[feedback-build-in-vs]]: the latest VS2026
 * MSVC raises C2079 (basic_istream::sentry undefined) when a plain .cpp consumer
 * instantiates the readTensorBlob/seekg template, so model loading must live in
 * an .ixx. main.cpp only imports this module.
 */

module;

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <filesystem>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <format>
#include <cuda_profiler_api.h>

export module Profiling.ProfileModel;

import Mila;
import Profiling.NvtxRange;

namespace Mila::Profiling
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;

    enum class Phase { Prefill, Decode, Generate };
    enum class Quantization { None, FP8, FP4 };
    enum class Precision { BF16, FP32 };

    struct Options
    {
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        Phase phase{ Phase::Decode };
        Quantization quantization{ Quantization::FP4 };
        Precision precision{ Precision::BF16 };
        std::string prompt{
            "Write a detailed essay about the history of computing and the people who shaped it." };
        std::size_t max_new_tokens{ 256 };
        std::size_t prefill_seq_len{ 0 };  // 0 => use the encoded prompt length
        int warmup_runs{ 1 };
        std::size_t context_length{ 4096 };
    };

    const char* phaseName( Phase phase )
    {
        switch ( phase )
        {
            case Phase::Prefill:  return "prefill";
            case Phase::Decode:   return "decode";
            default:              return "generate";
        }
    }

    void printUsage( const char* program )
    {
        std::cerr
            << "Usage: " << program << " [options]\n"
            << "  --phase           prefill | decode | generate.  Default: decode.\n"
            << "  --quantization    none | fp8 | fp4 (bf16 only). Default: fp4.\n"
            << "  --precision       bf16 | fp32.                  Default: bf16.\n"
            << "  --model-path      Weights file. Default: llama31_8b_instruct_bf16.bin.\n"
            << "  --tokenizer       Tokenizer file. Default: llama32_tokenizer.bin.\n"
            << "  --prompt          Prompt text (decode/generate, and prefill unless --seq-len).\n"
            << "  --tokens          Max new tokens for decode/generate. Default: 256.\n"
            << "  --seq-len         Prefill with this many dummy tokens instead of the prompt.\n"
            << "  --warmup          Unmeasured priming runs before the measured run. Default: 1.\n"
            << "  --context-length  Max sequence length allocated at load. Default: 4096.\n";
    }

    [[noreturn]] void argError( const std::string& message )
    {
        throw std::invalid_argument( message );
    }

    std::size_t parseSize( std::string_view value, const char* flag )
    {
        std::size_t result = 0;
        const auto status = std::from_chars( value.data(), value.data() + value.size(), result );

        if ( status.ec != std::errc{} )
            argError( std::format( "{} expects a non-negative integer, got '{}'", flag, value ) );

        return result;
    }

    Options parseArgs( int argc, char** argv )
    {
        Options options;
        options.model_path     = std::filesystem::path( MODELS_DIR ) / "llama" / "llama31_8b_instruct_bf16.bin";
        options.tokenizer_path = std::filesystem::path( MODELS_DIR ) / "llama" / "llama32_tokenizer.bin";

        for ( int i = 1; i < argc; ++i )
        {
            std::string_view arg = argv[ i ];

            auto nextValue = [&]( const char* flag ) -> std::string_view
            {
                if ( i + 1 >= argc )
                    argError( std::format( "{} requires a value", flag ) );

                return argv[ ++i ];
            };

            if ( arg == "--phase" )
            {
                std::string_view value = nextValue( "--phase" );

                if ( value == "prefill" )
                    options.phase = Phase::Prefill;
                else if ( value == "decode" )
                    options.phase = Phase::Decode;
                else if ( value == "generate" )
                    options.phase = Phase::Generate;
                else
                    argError( std::format( "Unknown --phase '{}'. Expected prefill, decode, or generate.", value ) );
            }
            else if ( arg == "--quantization" )
            {
                std::string_view value = nextValue( "--quantization" );

                if ( value == "none" )
                    options.quantization = Quantization::None;
                else if ( value == "fp8" )
                    options.quantization = Quantization::FP8;
                else if ( value == "fp4" )
                    options.quantization = Quantization::FP4;
                else
                    argError( std::format( "Unknown --quantization '{}'. Expected none, fp8, or fp4.", value ) );
            }
            else if ( arg == "--precision" )
            {
                std::string_view value = nextValue( "--precision" );

                if ( value == "bf16" )
                    options.precision = Precision::BF16;
                else if ( value == "fp32" )
                    options.precision = Precision::FP32;
                else
                    argError( std::format( "Unknown --precision '{}'. Expected bf16 or fp32.", value ) );
            }
            else if ( arg == "--model-path" )
            {
                options.model_path = nextValue( "--model-path" );
            }
            else if ( arg == "--tokenizer" )
            {
                options.tokenizer_path = nextValue( "--tokenizer" );
            }
            else if ( arg == "--prompt" )
            {
                options.prompt = nextValue( "--prompt" );
            }
            else if ( arg == "--tokens" )
            {
                options.max_new_tokens = parseSize( nextValue( "--tokens" ), "--tokens" );

                if ( options.max_new_tokens == 0 )
                    argError( "--tokens must be greater than zero" );
            }
            else if ( arg == "--seq-len" )
            {
                options.prefill_seq_len = parseSize( nextValue( "--seq-len" ), "--seq-len" );
            }
            else if ( arg == "--warmup" )
            {
                options.warmup_runs = static_cast<int>( parseSize( nextValue( "--warmup" ), "--warmup" ) );
            }
            else if ( arg == "--context-length" )
            {
                options.context_length = parseSize( nextValue( "--context-length" ), "--context-length" );

                if ( options.context_length == 0 )
                    argError( "--context-length must be greater than zero" );
            }
            else if ( arg == "--help" || arg == "-h" )
            {
                printUsage( argv[ 0 ] );
                std::exit( EXIT_SUCCESS );
            }
            else
            {
                argError( std::format( "Unknown option '{}'", arg ) );
            }
        }

        return options;
    }

    template<TensorDataType TPrecision>
    void runProfile( const Options& options )
    {
        using Model = LlamaModel<DeviceType::Cuda, TPrecision>;

        LlamaModelConfig model_config( options.context_length );

        if ( options.quantization == Quantization::FP8 )
            model_config.withFP8Quantization();
        else if ( options.quantization == Quantization::FP4 )
            model_config.withFP4Quantization();

        const DeviceId device{ DeviceType::Cuda, 0 };

        std::cout << "Loading model: " << options.model_path << "\n";

        auto model = Model::fromPretrained( options.model_path, model_config, device );

        std::cout << "Model loaded.\n";

        auto tokenizer = BpeTokenizer::loadLlama32( options.tokenizer_path );
        const auto encoded = tokenizer->encode( options.prompt );
        const std::vector<int32_t> prompt_tokens( encoded.begin(), encoded.end() );

        if ( options.phase == Phase::Prefill )
        {
            // --seq-len overrides the prompt with that many dummy tokens so prefill
            // cost can be measured at a fixed sequence length independent of the
            // tokenizer output.
            std::vector<int32_t> prefill_tokens = prompt_tokens;

            if ( options.prefill_seq_len > 0 )
                prefill_tokens.assign( options.prefill_seq_len, 0 );

            auto runPrefill = [&]( bool profiled )
            {
                if ( profiled )
                    cudaProfilerStart();

                {
                    Mila::Profiling::NvtxRange range( "prefill" );
                    model->profilePrefill( prefill_tokens );
                }

                if ( profiled )
                    cudaProfilerStop();
            };

            std::cout << std::format(
                "[prefill] seq_len={} warmup_runs={}\n",
                prefill_tokens.size(), options.warmup_runs );

            for ( int run = 0; run < options.warmup_runs; ++run )
                runPrefill( false );

            runPrefill( true );

            return;
        }

        // Greedy decode (temperature 0, top_k disabled) makes the generated token
        // sequence deterministic so successive runs and captures are repeatable.
        auto runGeneration = [&]( const char* label, bool profiled )
        {
            std::size_t produced = 0;

            if ( profiled )
                cudaProfilerStart();

            Mila::Dnn::GenerateParams gen_params;
            gen_params.max_new_tokens = static_cast<int>( options.max_new_tokens );
            gen_params.sampling.temperature = 0.0f;
            gen_params.sampling.top_k = 0;

            // The library streams tokens and returns only a finish reason; the profiler
            // measures timing from the callback cadence (prefill = call -> first token,
            // decode = first -> last token).
            const auto call_start = std::chrono::high_resolution_clock::now();
            auto first_token_time = call_start;
            auto last_token_time = call_start;

            {
                Mila::Profiling::NvtxRange range( label );
                [[maybe_unused]] const auto status = model->generate(
                    prompt_tokens,
                    [&]( int32_t )
                    {
                        const auto now = std::chrono::high_resolution_clock::now();
                        if ( produced == 0 )
                            first_token_time = now;
                        last_token_time = now;
                        ++produced;
                    },
                    gen_params,
                    {} );
            }

            if ( profiled )
                cudaProfilerStop();

            const std::size_t decode_tokens = produced > 0 ? produced - 1 : 0;
            const float prefill_ms =
                std::chrono::duration<float, std::milli>( first_token_time - call_start ).count();
            const float decode_ms =
                std::chrono::duration<float, std::milli>( last_token_time - first_token_time ).count();
            const float decode_tok_per_s =
                ( decode_ms > 0.0f && decode_tokens > 0 )
                ? static_cast<float>( decode_tokens ) / ( decode_ms / 1000.0f )
                : 0.0f;

            std::cout << std::format(
                "[{}] prompt_tokens={} tokens_generated={} prefill_ms={:.2f} "
                "decode_ms={:.2f} decode_tok_per_s={:.2f}\n",
                label,
                prompt_tokens.size(),
                produced,
                prefill_ms,
                decode_ms,
                decode_tok_per_s );
        };

        std::cout << std::format(
            "[{}] prompt_tokens={} max_new_tokens={} warmup_runs={} (greedy decode)\n",
            phaseName( options.phase ), prompt_tokens.size(), options.max_new_tokens, options.warmup_runs );

        for ( int run = 0; run < options.warmup_runs; ++run )
            runGeneration( "warmup", false );

        runGeneration(
            options.phase == Phase::Decode ? "decode_measured" : "generate_measured",
            true );
    }

    export int profileMain( int argc, char** argv )
    {
        try
        {
            Mila::initialize();

            const Options options = parseArgs( argc, argv );

            if ( !std::filesystem::exists( options.model_path ) )
            {
                std::cerr << "Error: model file not found: " << options.model_path << "\n";
                return EXIT_FAILURE;
            }

            if ( !std::filesystem::exists( options.tokenizer_path ) )
            {
                std::cerr << "Error: tokenizer file not found: " << options.tokenizer_path << "\n";
                return EXIT_FAILURE;
            }

            if ( options.precision == Precision::FP32 )
            {
                if ( options.quantization != Quantization::None )
                {
                    std::cerr << "Error: --quantization fp8/fp4 requires --precision bf16.\n";
                    return EXIT_FAILURE;
                }

                runProfile<TensorDataType::FP32>( options );
            }
            else
            {
                runProfile<TensorDataType::BF16>( options );
            }

            return EXIT_SUCCESS;
        }
        catch ( const std::exception& e )
        {
            std::cerr << "Fatal error: " << e.what() << "\n";
            return EXIT_FAILURE;
        }
    }
}
