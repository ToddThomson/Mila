/**
 * @file ModelTiming.cpp
 * @brief Llama on Metal through `LlamaModel::load` and `generate`, timed at llama-bench's shapes in FP16 and BF16, with memory.
 *
 * Phase 4's measures; the method and results are in Specifications/Notebooks/MetalFirstModel.md section 4.
 */

#include <mach/mach.h>
#include <sys/sysctl.h>

import std;
import Mila;

namespace
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    using Clock = std::chrono::steady_clock;

    constexpr std::uint32_t kPromptSeed = 20261105;

    // Ordinary vocabulary, clear of Llama 3's special tokens from 128000.
    constexpr std::int32_t kFirstOrdinaryToken = 1000;
    constexpr std::int32_t kLastOrdinaryToken = 100000;

    struct Options
    {
        std::filesystem::path weights;
        int rounds{ 3 };
        int runs{ 5 };
        dim_t context{ 8192 };
        dim_t prefill_tokens{ 512 };
        int decode_tokens{ 128 };
        dim_t depth{ 4096 };
        bool planner{ true };
        bool rates{ true };
        bool fp16{ true };
        bool bf16{ true };
    };

    struct ProcessMemory
    {
        std::uint64_t footprint{ 0 };
        std::uint64_t peak{ 0 };
    };

    ProcessMemory processMemory()
    {
        task_vm_info_data_t info{};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;

        if ( task_info( mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>( &info ), &count ) != KERN_SUCCESS )
        {
            throw std::runtime_error( "task_info( TASK_VM_INFO ) failed" );
        }

        return { info.phys_footprint, static_cast<std::uint64_t>( info.ledger_phys_footprint_peak ) };
    }

    struct SwapState
    {
        std::uint64_t swap_used{ 0 };
        std::uint64_t swapouts{ 0 };
        std::uint64_t pageouts{ 0 };
    };

    SwapState swapState()
    {
        xsw_usage usage{};
        std::size_t size = sizeof( usage );

        if ( sysctlbyname( "vm.swapusage", &usage, &size, nullptr, 0 ) != 0 )
        {
            throw std::runtime_error( "sysctl vm.swapusage failed" );
        }

        vm_statistics64_data_t statistics{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

        if ( host_statistics64( mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>( &statistics ), &count ) != KERN_SUCCESS )
        {
            throw std::runtime_error( "host_statistics64 failed" );
        }

        return { usage.xsu_used, statistics.swapouts, statistics.pageouts };
    }

    double gib( std::uint64_t bytes )
    {
        return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 );
    }

    double seconds( Clock::time_point from, Clock::time_point to )
    {
        return std::chrono::duration<double>( to - from ).count();
    }

    std::vector<std::int32_t> randomTokens( dim_t count, std::uint32_t seed )
    {
        std::mt19937 engine( seed );
        std::uniform_int_distribution<std::int32_t> ids( kFirstOrdinaryToken, kLastOrdinaryToken );

        std::vector<std::int32_t> tokens( static_cast<std::size_t>( count ) );
        std::ranges::generate( tokens, [&]() { return ids( engine ); } );

        return tokens;
    }

    std::string_view precisionName( TensorDataType precision )
    {
        return precision == TensorDataType::FP16 ? "FP16" : "BF16";
    }

    std::string_view contextLimitName( DeploymentPlan::ContextLimit limit )
    {
        switch ( limit )
        {
            case DeploymentPlan::ContextLimit::FixedByCaller:
                return "fixed by the caller";
            case DeploymentPlan::ContextLimit::TrainedMaximum:
                return "the trained maximum";
            case DeploymentPlan::ContextLimit::FullPrefillChunk:
                return "a longer context would shrink the prefill chunk";
            case DeploymentPlan::ContextLimit::DeviceMemory:
                return "device memory";
        }

        return "unknown";
    }

    /// Tokens per second over all loads and runs of one shape in one precision.
    struct Samples
    {
        std::vector<double> prefill;
        std::vector<double> decode;
        std::vector<double> decode_at_depth;
        std::vector<double> load_seconds;
    };

    struct Spread
    {
        double median{ 0.0 };
        double lowest{ 0.0 };
        double highest{ 0.0 };
    };

    Spread spread( std::vector<double> values )
    {
        if ( values.empty() )
        {
            return {};
        }

        std::ranges::sort( values );
        const std::size_t middle = values.size() / 2;
        const double median = values.size() % 2 == 1 ? values[ middle ] : 0.5 * ( values[ middle - 1 ] + values[ middle ] );

        return { median, values.front(), values.back() };
    }

    GenerateParams greedy( int max_new_tokens )
    {
        GenerateParams params;
        params.max_new_tokens = max_new_tokens;
        params.sampling.temperature = 0.0f;
        params.sampling.top_k = 0;

        // Every run prefills its whole prompt, and decodes its full count past an end-of-sequence token.
        params.reuse_kv_prefix = false;
        params.stop_tokens = { -1 };

        return params;
    }

    template<typename TModel>
    double prefillRate( TModel& model, std::span<const std::int32_t> prompt )
    {
        Clock::time_point first_token{};
        const Clock::time_point start = Clock::now();

        (void)model.generate( prompt, [&]( std::int32_t ) { first_token = Clock::now(); }, greedy( 1 ) );

        return static_cast<double>( prompt.size() ) / seconds( start, first_token );
    }

    template<typename TModel>
    double decodeRate( TModel& model, std::span<const std::int32_t> prompt, int tokens )
    {
        std::vector<Clock::time_point> stamps;
        stamps.reserve( static_cast<std::size_t>( tokens ) );

        (void)model.generate( prompt, [&]( std::int32_t ) { stamps.push_back( Clock::now() ); }, greedy( tokens ) );

        if ( stamps.size() != static_cast<std::size_t>( tokens ) )
        {
            throw std::runtime_error( std::format( "asked for {} tokens, generated {}", tokens, stamps.size() ) );
        }

        return static_cast<double>( tokens - 1 ) / seconds( stamps.front(), stamps.back() );
    }

    void printMemory( std::string_view label, const DeploymentPlan& plan )
    {
        const DeviceReading reading = DeviceReading::take( plan.device() );
        const ProcessMemory process = processMemory();
        const std::uint64_t allocated = reading.total_bytes - reading.free_bytes;

        std::cout << std::format( "[ memory ] {}: plan {:.3f} GiB, device allocated {:.3f} GiB, process {:.3f} GiB (peak {:.3f}), "
            "working set {:.3f} GiB, allocated {:.1f}% of it\n",
            label, gib( plan.footprint().totalDeviceBytes() ), gib( allocated ), gib( process.footprint ), gib( process.peak ),
            gib( reading.total_bytes ), 100.0 * static_cast<double>( allocated ) / static_cast<double>( reading.total_bytes ) );
    }

    template<TensorDataType TPrecision>
    void measureLoad( const Options& options, Samples& samples, std::string_view label )
    {
        using Model = LlamaModel<DeviceType::Metal, TPrecision>;

        const Clock::time_point start = Clock::now();
        auto model = Model::load( options.weights, LlamaModelConfig( options.context ), Device::Metal( 0 ) );
        const double load = seconds( start, Clock::now() );

        samples.load_seconds.push_back( load );
        std::cout << std::format( "[ load ] {} {}: {:.2f} s, context {}, prefill chunk {}\n", label, precisionName( TPrecision ), load,
            model->getDeploymentPlan().contextLength(), model->getDeploymentPlan().prefillChunkRows() );
        printMemory( std::format( "{} {} after load", label, precisionName( TPrecision ) ), model->getDeploymentPlan() );

        const std::vector<std::int32_t> prompt = randomTokens( options.prefill_tokens, kPromptSeed );
        const std::vector<std::int32_t> one_token = randomTokens( 1, kPromptSeed + 1 );
        const std::vector<std::int32_t> deep = randomTokens( options.depth, kPromptSeed + 2 );

        (void)prefillRate( *model, prompt );
        (void)decodeRate( *model, one_token, options.decode_tokens );
        (void)decodeRate( *model, deep, options.decode_tokens );

        for ( int run = 0; run < options.runs; ++run )
        {
            const double prefill = prefillRate( *model, prompt );
            const double decode = decodeRate( *model, one_token, options.decode_tokens );
            const double decode_at_depth = decodeRate( *model, deep, options.decode_tokens );

            samples.prefill.push_back( prefill );
            samples.decode.push_back( decode );
            samples.decode_at_depth.push_back( decode_at_depth );

            std::cout << std::format( "[ run ] {} {} run {}: prefill {:.1f} tokens/s, decode {:.2f} tokens/s, decode after {} {:.2f} tokens/s\n",
                label, precisionName( TPrecision ), run, prefill, decode, options.depth, decode_at_depth );
        }

        printMemory( std::format( "{} {} after the runs", label, precisionName( TPrecision ) ), model->getDeploymentPlan() );
    }

    template<TensorDataType TPrecision>
    void measurePlannerContext( const Options& options )
    {
        using Model = LlamaModel<DeviceType::Metal, TPrecision>;

        const DeploymentRequest request = DeploymentRequest().withAutomaticContextLength().withAllowedDevices( Device::Metal( 0 ) );
        const auto planned = Model::planDeployment( options.weights, request );

        if ( !planned )
        {
            std::cout << std::format( "[ planner ] {}: refused, {}\n", precisionName( TPrecision ), planned.error().toString() );
            return;
        }

        const DeploymentPlan& plan = planned->best();

        std::cout << std::format( "[ planner ] {}: context {}, limited by {}, prefill chunk {}, plan {:.3f} GiB of a working set of {:.3f} GiB "
            "({:.1f}%), free at the reading {:.3f} GiB\n",
            precisionName( TPrecision ), plan.contextLength(), contextLimitName( plan.contextLimitedBy() ), plan.prefillChunkRows(),
            gib( plan.footprint().totalDeviceBytes() ), gib( plan.reading().total_bytes ),
            100.0 * static_cast<double>( plan.footprint().totalDeviceBytes() ) / static_cast<double>( plan.reading().total_bytes ),
            gib( plan.reading().free_bytes ) );

        const SwapState before = swapState();
        const Clock::time_point start = Clock::now();
        auto model = Model::load( options.weights, plan );

        std::cout << std::format( "[ planner ] {}: loaded in {:.2f} s\n", precisionName( TPrecision ), seconds( start, Clock::now() ) );
        printMemory( std::format( "planner {} after load", precisionName( TPrecision ) ), model->getDeploymentPlan() );

        const std::vector<std::int32_t> prompt = randomTokens( options.prefill_tokens, kPromptSeed );
        const std::vector<std::int32_t> one_token = randomTokens( 1, kPromptSeed + 1 );

        (void)prefillRate( *model, prompt );
        (void)decodeRate( *model, one_token, options.decode_tokens );

        for ( int run = 0; run < options.runs; ++run )
        {
            const double prefill = prefillRate( *model, prompt );
            const double decode = decodeRate( *model, one_token, options.decode_tokens );

            std::cout << std::format( "[ planner ] {} run {}: prefill {:.1f} tokens/s, decode {:.2f} tokens/s\n",
                precisionName( TPrecision ), run, prefill, decode );
        }

        const SwapState after = swapState();

        printMemory( std::format( "planner {} after the run", precisionName( TPrecision ) ), model->getDeploymentPlan() );
        std::cout << std::format( "[ planner ] {}: swap used {:.3f} -> {:.3f} GiB, swap-outs {} -> {}, page-outs {} -> {}\n",
            precisionName( TPrecision ), gib( before.swap_used ), gib( after.swap_used ), before.swapouts, after.swapouts,
            before.pageouts, after.pageouts );
    }

    void printSummary( TensorDataType precision, const Samples& samples, const Options& options )
    {
        auto line = [&]( std::string_view shape, const std::vector<double>& values, std::string_view unit )
        {
            const Spread s = spread( values );
            std::cout << std::format( "[ summary ] {} {}: median {:.2f} {} (lowest {:.2f}, highest {:.2f}, {} samples)\n",
                precisionName( precision ), shape, s.median, unit, s.lowest, s.highest, values.size() );
        };

        line( std::format( "prefill {}", options.prefill_tokens ), samples.prefill, "tokens/s" );
        line( std::format( "decode {}", options.decode_tokens ), samples.decode, "tokens/s" );
        line( std::format( "decode {} after {}", options.decode_tokens, options.depth ), samples.decode_at_depth, "tokens/s" );
        line( "load", samples.load_seconds, "s" );
    }

    Options parse( int argc, char** argv )
    {
        Options options;

        for ( int i = 1; i < argc; ++i )
        {
            const std::string_view arg = argv[ i ];

            auto next = [&]() -> std::string_view
            {
                if ( i + 1 >= argc )
                {
                    throw std::invalid_argument( std::format( "{} needs a value", arg ) );
                }

                return argv[ ++i ];
            };

            if ( arg == "--weights" )
            {
                options.weights = next();
            } else if ( arg == "--rounds" )
            {
                options.rounds = std::stoi( std::string( next() ) );
            } else if ( arg == "--runs" )
            {
                options.runs = std::stoi( std::string( next() ) );
            } else if ( arg == "--depth" )
            {
                options.depth = std::stoll( std::string( next() ) );
            } else if ( arg == "--no-planner" )
            {
                options.planner = false;
            } else if ( arg == "--planner-only" )
            {
                options.rates = false;
            } else if ( arg == "--precision" )
            {
                const std::string_view precision = next();
                options.fp16 = precision == "fp16";
                options.bf16 = precision == "bf16";

                if ( !options.fp16 && !options.bf16 )
                {
                    throw std::invalid_argument( "--precision is fp16 or bf16" );
                }
            } else
            {
                throw std::invalid_argument( std::format( "unknown argument {}", arg ) );
            }
        }

        if ( options.weights.empty() )
        {
            throw std::invalid_argument( "usage: mila-metal-model-timing --weights <llama bf16 weights> [--rounds 3] [--runs 5] [--depth 4096] "
                "[--precision fp16|bf16] [--no-planner | --planner-only]" );
        }

        return options;
    }
}

int main( int argc, char** argv )
{
    try
    {
        const Options options = parse( argc, argv );

        Mila::initialize( 0, std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning ) );

        Samples fp16;
        Samples bf16;

        for ( int round = 0; options.rates && round < options.rounds; ++round )
        {
            const std::string label = std::format( "round {}", round );

            auto measureFp16 = [&]()
            {
                if ( options.fp16 )
                {
                    measureLoad<TensorDataType::FP16>( options, fp16, label );
                }
            };

            auto measureBf16 = [&]()
            {
                if ( options.bf16 )
                {
                    measureLoad<TensorDataType::BF16>( options, bf16, label );
                }
            };

            if ( round % 2 == 0 )
            {
                measureFp16();
                measureBf16();
            } else
            {
                measureBf16();
                measureFp16();
            }
        }

        if ( options.rates && options.fp16 )
        {
            printSummary( TensorDataType::FP16, fp16, options );
        }

        if ( options.rates && options.bf16 )
        {
            printSummary( TensorDataType::BF16, bf16, options );
        }

        if ( options.planner && options.fp16 )
        {
            measurePlannerContext<TensorDataType::FP16>( options );
        }

        if ( options.planner && options.bf16 )
        {
            measurePlannerContext<TensorDataType::BF16>( options );
        }

        return 0;
    } catch ( const std::exception& error )
    {
        std::cerr << "mila-metal-model-timing: " << error.what() << "\n";

        return 1;
    }
}
