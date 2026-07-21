/**
 * @file ProfileModel.ixx
 * @brief General-purpose, non-interactive model profiling harness for Nsight.
 *
 * Loads a Llama or Gemma model and exercises one phase of the inference path
 * under a cudaProfilerApi capture region and an NVTX range, so Nsight Systems
 * can be run with --capture-range=cudaProfilerApi to profile only the measured
 * region (excluding model load and warmup). Greedy decode keeps runs
 * repeatable; --temperature > 0 switches to the stochastic sampler so its
 * kernel appears in the capture.
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
#include <thread>
#include <atomic>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

export module Profiling.ProfileModel;

import Mila;
import Profiling.NvtxRange;

namespace Mila::Profiling
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;

    enum class ModelFamily { Llama, Gemma };
    enum class Phase { Prefill, Decode, Generate };
    enum class Quantization { None, FP8, FP4 };
    enum class Precision { BF16, FP32 };

    struct Options
    {
        ModelFamily model_family{ ModelFamily::Llama };
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        Phase phase{ Phase::Decode };
        Quantization quantization{ Quantization::FP4 };
        Precision precision{ Precision::BF16 };
        std::string prompt{
            "Write a detailed essay about the history of computing and the people who shaped it." };
        std::size_t max_new_tokens{ 256 };
        std::size_t prefill_seq_len{ 0 };  // 0 => use the encoded prompt length
        float temperature{ 0.0f };  // 0 => greedy (repeatable); > 0 => stochastic sampler
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

    // Device VRAM poll. nvidia-smi cannot see committed VRAM under Windows WDDM
    // (it reports the idle baseline while a multi-GB model is resident), so the
    // reliable footprint reading is cudaMemGetInfo from inside the process. Printed
    // at baseline / after-load / after-run to bracket the chunk-scaled activation
    // budget the prefill heuristic (Gemma resolvePrefillChunkSize) works against.
    void printGpuMemory( const char* label )
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo( &free_bytes, &total_bytes );

        if ( status != cudaSuccess )
        {
            std::cout << std::format( "[mem] {}: cudaMemGetInfo failed ({})\n",
                label, cudaGetErrorString( status ) );
            return;
        }

        constexpr double mib = 1024.0 * 1024.0;
        const size_t used_bytes = total_bytes - free_bytes;

        std::cout << std::format(
            "[mem] {}: used={:.0f} MiB  free={:.0f} MiB  total={:.0f} MiB\n",
            label, used_bytes / mib, free_bytes / mib, total_bytes / mib );
    }

    // Model-load timing: wall time plus the effective end-to-end throughput
    // (checkpoint bytes on disk / wall). The checkpoint is BF16 (~4x the resident
    // FP4 weights), so this throughput is the efficiency of the whole read -> H2D ->
    // on-device-quantize pipeline. A number well under the disk's sequential-read
    // ceiling points at the read/transfer stage, not the (device) quantize -- which
    // is what tells us whether a load-time optimization has anything to bite on.
    void reportLoad( const std::filesystem::path& model_path, double load_ms )
    {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;

        std::error_code ec;
        const auto bytes = std::filesystem::file_size( model_path, ec );

        if ( ec )
        {
            std::cout << std::format( "[load] wall_ms={:.1f}  (checkpoint size unavailable: {})\n",
                load_ms, ec.message() );
            return;
        }

        const double gib_bytes = static_cast<double>( bytes ) / gib;
        const double gib_per_s = gib_bytes / ( load_ms / 1000.0 );

        std::cout << std::format(
            "[load] wall_ms={:.1f}  checkpoint={:.2f} GiB  throughput={:.2f} GiB/s\n",
            load_ms, gib_bytes, gib_per_s );
    }

    // Background high-water sampler for a single scoped window (e.g. model load).
    // The three point-polls (baseline/after-load/after-run) miss any transient that
    // allocates and frees inside the window -- the quantize-on-load BF16 staging and
    // the pinned/mmap load buffers do exactly that. This thread polls cudaMemGetInfo
    // every few ms and reports the peak device usage (min free) it observed, which is
    // the only in-process way to catch a load-time VRAM spike (nvidia-smi is blind
    // under WDDM).
    class VramHighWaterSampler
    {
    public:
        void start()
        {
            stop_.store( false, std::memory_order_relaxed );
            worker_ = std::thread( [this]
            {
                cudaSetDevice( 0 );

                while ( !stop_.load( std::memory_order_relaxed ) )
                {
                    size_t free_bytes = 0;
                    size_t total_bytes = 0;

                    if ( cudaMemGetInfo( &free_bytes, &total_bytes ) == cudaSuccess )
                    {
                        total_bytes_ = total_bytes;

                        if ( free_bytes < min_free_bytes_ )
                            min_free_bytes_ = free_bytes;

                        // Largest transient freed: track the running peak usage and the
                        // deepest drop below it. A buffer allocated then freed EARLY in load
                        // (before the resident set is fully built) never shows as the global
                        // max-used, but it does show as a drawdown -- running_peak climbs over
                        // it, then usage drops when it is freed. max_drawdown == the biggest
                        // such alloc/free (the "~2 GB returned" the user sees in Task Manager).
                        const size_t used = total_bytes - free_bytes;

                        if ( used > running_peak_used_ )
                            running_peak_used_ = used;

                        const size_t drawdown = running_peak_used_ - used;

                        if ( drawdown > max_drawdown_bytes_ )
                            max_drawdown_bytes_ = drawdown;
                    }

                    std::this_thread::sleep_for( std::chrono::milliseconds( 3 ) );
                }
            } );
        }

        void stopAndReport( const char* label )
        {
            stop_.store( true, std::memory_order_relaxed );

            if ( worker_.joinable() )
                worker_.join();

            constexpr double mib = 1024.0 * 1024.0;
            const size_t peak_used = ( total_bytes_ > min_free_bytes_ )
                ? total_bytes_ - min_free_bytes_ : 0;

            std::cout << std::format(
                "[mem-peak] {}: peak used={:.0f} MiB  min free={:.0f} MiB  "
                "largest transient freed={:.0f} MiB  (sampled @3ms)\n",
                label, peak_used / mib, min_free_bytes_ / mib, max_drawdown_bytes_ / mib );
        }

    private:
        std::thread worker_;
        std::atomic<bool> stop_{ false };
        // Written only by the worker thread; read after join() -- no data race.
        size_t min_free_bytes_{ std::numeric_limits<size_t>::max() };
        size_t total_bytes_{ 0 };
        size_t running_peak_used_{ 0 };
        size_t max_drawdown_bytes_{ 0 };
    };

    void printUsage( const char* program )
    {
        std::cerr
            << "Usage: " << program << " [options]\n"
            << "  --model           llama | gemma.                Default: llama.\n"
            << "  --phase           prefill | decode | generate.  Default: decode.\n"
            << "  --quantization    none | fp8 | fp4 (bf16 only). Default: fp4.\n"
            << "  --precision       bf16 | fp32 (llama only).     Default: bf16.\n"
            << "  --model-path      Weights file. Default: per --model family.\n"
            << "  --tokenizer       Tokenizer file. Default: per --model family.\n"
            << "  --prompt          Prompt text (decode/generate, and prefill unless --seq-len).\n"
            << "  --tokens          Max new tokens for decode/generate. Default: 256.\n"
            << "  --seq-len         Prefill with this many dummy tokens instead of the prompt.\n"
            << "  --temperature     0 = greedy (repeatable); > 0 profiles the stochastic sampler. Default: 0.\n"
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

    float parseFloat( std::string_view value, const char* flag )
    {
        float result = 0.0f;
        const auto status = std::from_chars( value.data(), value.data() + value.size(), result );

        if ( status.ec != std::errc{} )
            argError( std::format( "{} expects a number, got '{}'", flag, value ) );

        return result;
    }

    Options parseArgs( int argc, char** argv )
    {
        Options options;

        for ( int i = 1; i < argc; ++i )
        {
            std::string_view arg = argv[ i ];

            auto nextValue = [&]( const char* flag ) -> std::string_view
            {
                if ( i + 1 >= argc )
                    argError( std::format( "{} requires a value", flag ) );

                return argv[ ++i ];
            };

            if ( arg == "--model" )
            {
                std::string_view value = nextValue( "--model" );

                if ( value == "llama" )
                    options.model_family = ModelFamily::Llama;
                else if ( value == "gemma" )
                    options.model_family = ModelFamily::Gemma;
                else
                    argError( std::format( "Unknown --model '{}'. Expected llama or gemma.", value ) );
            }
            else if ( arg == "--phase" )
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
            else if ( arg == "--temperature" )
            {
                options.temperature = parseFloat( nextValue( "--temperature" ), "--temperature" );

                if ( options.temperature < 0.0f )
                    argError( "--temperature must be non-negative" );
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

        if ( options.model_path.empty() )
        {
            options.model_path = ( options.model_family == ModelFamily::Gemma )
                ? std::filesystem::path( MODELS_DIR ) / "Gemma" / "gemma4_12b_it_bf16.bin"
                : std::filesystem::path( MODELS_DIR ) / "llama" / "llama31_8b_instruct_bf16.bin";
        }

        if ( options.tokenizer_path.empty() )
        {
            options.tokenizer_path = ( options.model_family == ModelFamily::Gemma )
                ? std::filesystem::path( MODELS_DIR ) / "Gemma" / "gemma_tokenizer.bin"
                : std::filesystem::path( MODELS_DIR ) / "llama" / "llama32_tokenizer.bin";
        }

        return options;
    }

    // Shared phase driver. Model families differ only in construction; the
    // measured surface (generate(), profilePrefill()) has the same shape on
    // LlamaModel and GemmaModel, so the phases are family-agnostic.
    template<typename TModel>
    void runPhases( TModel& model, const Options& options, const std::vector<int32_t>& prompt_tokens )
    {
        if ( options.phase == Phase::Prefill )
        {
            // --seq-len overrides the prompt with that many dummy tokens so prefill
            // cost can be measured at a fixed sequence length independent of the
            // tokenizer output.
            std::vector<int32_t> prefill_tokens = prompt_tokens;

            if ( options.prefill_seq_len > 0 )
                prefill_tokens.assign( options.prefill_seq_len, 0 );

            // profilePrefill() runs the prompt forward pass and synchronizes, so a
            // wall-clock bracket around it is a valid device-inclusive prefill time.
            auto timedPrefill = [&]() -> float
            {
                const auto start = std::chrono::high_resolution_clock::now();

                {
                    Mila::Profiling::NvtxRange range( "prefill" );
                    model.profilePrefill( prefill_tokens );
                }

                const auto stop = std::chrono::high_resolution_clock::now();

                return std::chrono::duration<float, std::milli>( stop - start ).count();
            };

            std::cout << std::format(
                "[prefill] seq_len={} context_length={} warmup_runs={}\n",
                prefill_tokens.size(), options.context_length, options.warmup_runs );

            for ( int run = 0; run < options.warmup_runs; ++run )
                timedPrefill();

            // A few measured runs; report min (least noise) and mean. The tax-gone
            // sweep (GqaFlashAttention.md 10, item 2) reads min_ms across context
            // lengths -- a flat curve means the over-allocation tax is gone.
            constexpr int kMeasuredRuns = 5;
            float min_ms = std::numeric_limits<float>::max();
            float sum_ms = 0.0f;

            for ( int run = 0; run < kMeasuredRuns; ++run )
            {
                const float ms = timedPrefill();
                min_ms = std::min( min_ms, ms );
                sum_ms += ms;
            }

            std::cout << std::format(
                "[prefill] runs={} min_ms={:.2f} mean_ms={:.2f}\n",
                kMeasuredRuns, min_ms, sum_ms / kMeasuredRuns );

            // Final capture run for Nsight (--capture-range=cudaProfilerApi): the
            // attribution split (attention vs linear GEMM vs launch gaps) is read here.
            cudaProfilerStart();
            timedPrefill();
            cudaProfilerStop();

            return;
        }

        // Greedy decode (temperature 0, top_k disabled) makes the generated token
        // sequence deterministic so successive runs and captures are repeatable.
        // --temperature > 0 keeps the library sampling defaults active so the
        // stochastic sampler kernel shows up in the capture instead of argmax.
        auto runGeneration = [&]( const char* label, bool profiled )
        {
            std::size_t produced = 0;

            if ( profiled )
                cudaProfilerStart();

            Mila::Dnn::GenerateParams gen_params;
            gen_params.max_new_tokens = static_cast<int>( options.max_new_tokens );
            gen_params.sampling.temperature = options.temperature;

            if ( options.temperature == 0.0f )
                gen_params.sampling.top_k = 0;

            // The library streams tokens and returns only a finish reason; the profiler
            // measures timing from the callback cadence (prefill = call -> first token,
            // decode = first -> last token).
            const auto call_start = std::chrono::high_resolution_clock::now();
            auto first_token_time = call_start;
            auto last_token_time = call_start;

            {
                Mila::Profiling::NvtxRange range( label );
                [[maybe_unused]] const auto status = model.generate(
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
            "[{}] prompt_tokens={} max_new_tokens={} temperature={} warmup_runs={}\n",
            phaseName( options.phase ), prompt_tokens.size(), options.max_new_tokens,
            options.temperature, options.warmup_runs );

        for ( int run = 0; run < options.warmup_runs; ++run )
            runGeneration( "warmup", false );

        runGeneration(
            options.phase == Phase::Decode ? "decode_measured" : "generate_measured",
            true );
    }

    template<TensorDataType TPrecision>
    void runProfileLlama( const Options& options )
    {
        using Model = LlamaModel<DeviceType::Cuda, TPrecision>;

        LlamaModelConfig model_config( options.context_length );

        if ( options.quantization == Quantization::FP8 )
            model_config.withFP8Quantization();
        else if ( options.quantization == Quantization::FP4 )
            model_config.withFP4Quantization();

        const DeviceId device{ DeviceType::Cuda, 0 };

        std::cout << "Loading model: " << options.model_path << "\n";

        VramHighWaterSampler load_sampler;
        load_sampler.start();
        const auto load_start = std::chrono::high_resolution_clock::now();

        // IIFE so the NVTX "model_load" range covers exactly the load (visible under
        // nsys for the H2D memcpy breakdown) while keeping model in the outer scope.
        auto model = [&]
        {
            NvtxRange range( "model_load" );
            return Model::fromPretrained( options.model_path, model_config, device );
        }();

        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - load_start ).count();

        std::cout << "Model loaded.\n";
        reportLoad( options.model_path, load_ms );
        load_sampler.stopAndReport( "during load window (transient high-water)" );
        std::cout << model->toString();
        printGpuMemory( "after model load (weights + KV cache + prefill workspace)" );

        auto tokenizer = BpeTokenizer::loadLlama32( options.tokenizer_path );
        const auto encoded = tokenizer->encode( options.prompt );
        const std::vector<int32_t> prompt_tokens( encoded.begin(), encoded.end() );

        runPhases( *model, options, prompt_tokens );

        printGpuMemory( "after run (includes cuBLASLt workspace growth)" );
    }

    template<TensorDataType TPrecision>
    void runProfileGemma( const Options& options )
    {
        using Model = GemmaModel<DeviceType::Cuda, TPrecision>;

        GemmaModelConfig model_config( options.context_length );

        if ( options.quantization == Quantization::FP8 )
            model_config.withFP8Quantization();
        else if ( options.quantization == Quantization::FP4 )
            model_config.withFP4Quantization();

        const DeviceId device{ DeviceType::Cuda, 0 };

        std::cout << "Loading model: " << options.model_path << "\n";

        VramHighWaterSampler load_sampler;
        load_sampler.start();
        const auto load_start = std::chrono::high_resolution_clock::now();

        // IIFE so the NVTX "model_load" range covers exactly the load (visible under
        // nsys for the H2D memcpy breakdown) while keeping model in the outer scope.
        auto model = [&]
        {
            NvtxRange range( "model_load" );
            return Model::fromPretrained( options.model_path, model_config, device );
        }();

        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - load_start ).count();

        std::cout << "Model loaded.\n";
        reportLoad( options.model_path, load_ms );
        load_sampler.stopAndReport( "during load window (transient high-water)" );
        std::cout << model->toString();
        printGpuMemory( "after model load (weights + KV cache + prefill workspace)" );

        auto tokenizer = BpeTokenizer::loadGemma( options.tokenizer_path );
        const auto encoded = tokenizer->encode( options.prompt );
        const std::vector<int32_t> prompt_tokens( encoded.begin(), encoded.end() );

        runPhases( *model, options, prompt_tokens );

        printGpuMemory( "after run (includes cuBLASLt workspace growth)" );
    }

    export int profileMain( int argc, char** argv )
    {
        try
        {
            Mila::initialize();

            printGpuMemory( "baseline (after init, before model load)" );

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

                // GemmaModel is only instantiated at BF16 anywhere in the tree (Chat is
                // BF16-only); keep FP32 Llama-only rather than grow a new instantiation.
                if ( options.model_family == ModelFamily::Gemma )
                {
                    std::cerr << "Error: --model gemma supports --precision bf16 only.\n";
                    return EXIT_FAILURE;
                }

                runProfileLlama<TensorDataType::FP32>( options );
            }
            else if ( options.model_family == ModelFamily::Gemma )
            {
                runProfileGemma<TensorDataType::BF16>( options );
            }
            else
            {
                runProfileLlama<TensorDataType::BF16>( options );
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
