/**
 * @file ModelTiming.cpp
 * @brief Llama or Gemma on Metal through the family's `load` and `generate`, timed at llama-bench's shapes in FP16 and BF16, with memory;
 * `--prefill-depths` times a prompt added after a held one, as `-p 512 -d N`.
 *
 * Phase 4's measures, in the weights' own stored format; the method is in Specifications/Notebooks/MetalFirstModel.md section 4.
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

    // Ordinary vocabulary in both families: clear of Llama 3's special tokens from 128000 and of Gemma's below 1000.
    constexpr std::int32_t kFirstOrdinaryToken = 1000;
    constexpr std::int32_t kLastOrdinaryToken = 100000;

    enum class Family
    {
        Llama,
        Gemma
    };

    template<Family TFamily, TensorDataType TPrecision>
    struct FamilyModel;

    template<TensorDataType TPrecision>
    struct FamilyModel<Family::Llama, TPrecision>
    {
        using type = LlamaModel<DeviceType::Metal, TPrecision>;
    };

    template<TensorDataType TPrecision>
    struct FamilyModel<Family::Gemma, TPrecision>
    {
        using type = GemmaModel<DeviceType::Metal, TPrecision>;
    };

    struct Options
    {
        std::filesystem::path weights;
        Family family{ Family::Llama };
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

        // Greedy unless set: an application samples, which a comparison of engines does not.
        float temperature{ 0.0f };
        int top_k{ 0 };
        float top_p{ 1.0f };

        // Each dispatch timed by the GPU and printed by kernel; the rates it gives are not the model's.
        bool profile_dispatches{ false };

        // With profile_dispatches, the decode is profiled after a held prompt of each of these lengths, 0 the one-token prompt;
        // unset, after the one-token prompt alone.
        std::vector<dim_t> profile_depths;

        // When set, each load times a prompt of `prefill_tokens` added after a held prompt of each depth, llama-bench's
        // `-p 512 -d N`, in place of the decode shapes.
        std::vector<dim_t> prefill_depths;

        // Shows the library's information lines, among them each prompt's prefix reuse.
        bool log_info{ false };
    };

    // Set once from the command line; read by every generate call.
    Options sampling_options;

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

        // By held depth: the held prompt's own prefill, and the prompt added after it.
        std::map<dim_t, std::vector<double>> fill;
        std::map<dim_t, std::vector<double>> extend;
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

    GenerateParams generateParams( int max_new_tokens, bool reuse_kv_prefix = false )
    {
        GenerateParams params;
        params.max_new_tokens = max_new_tokens;
        params.sampling.temperature = sampling_options.temperature;
        params.sampling.top_k = sampling_options.top_k;
        params.sampling.top_p = sampling_options.top_p;

        // Every run prefills its whole prompt unless it extends a held one, and decodes its full count past an
        // end-of-sequence token.
        params.reuse_kv_prefix = reuse_kv_prefix;
        params.stop_tokens = { -1 };

        return params;
    }

    template<typename TModel>
    double prefillRate( TModel& model, std::span<const std::int32_t> prompt )
    {
        Clock::time_point first_token{};
        const Clock::time_point start = Clock::now();

        (void)model.generate( prompt, [&]( std::int32_t ) { first_token = Clock::now(); }, generateParams( 1 ) );

        return static_cast<double>( prompt.size() ) / seconds( start, first_token );
    }

    template<typename TModel>
    double decodeRate( TModel& model, std::span<const std::int32_t> prompt, int tokens )
    {
        std::vector<Clock::time_point> stamps;
        stamps.reserve( static_cast<std::size_t>( tokens ) );

        (void)model.generate( prompt, [&]( std::int32_t ) { stamps.push_back( Clock::now() ); }, generateParams( tokens ) );

        if ( stamps.size() != static_cast<std::size_t>( tokens ) )
        {
            throw std::runtime_error( std::format( "asked for {} tokens, generated {}", tokens, stamps.size() ) );
        }

        return static_cast<double>( tokens - 1 ) / seconds( stamps.front(), stamps.back() );
    }

    // Each kernel's GPU time divided by `per`: generated tokens for a decode, one for a prefill.
    void printProfile( const std::vector<DispatchTime>& profile, int per, std::string_view unit, std::string_view heading )
    {
        double total = 0.0;

        for ( const DispatchTime& time : profile )
        {
            total += time.gpu_seconds;
        }

        std::cout << std::format( "[ profile ] {}: {} kernels, {:.2f} ms of GPU time {}\n", heading, profile.size(), 1000.0 * total / per, unit );

        for ( const DispatchTime& time : profile )
        {
            std::cout << std::format( "[ profile ] {:9.3f} ms {} {:5.1f}%  {:7.1f} dispatches {}  {}\n",
                1000.0 * time.gpu_seconds / per, unit, 100.0 * time.gpu_seconds / total,
                static_cast<double>( time.dispatches ) / per, unit, time.kernel );
        }
    }

    // One decode of `tokens` after `prompt`, every dispatch timed by the GPU from the first generated token on, so the prompt's
    // prefill is not in it, shown per decode step.
    template<typename TModel>
    void printDecodeProfile( TModel& model, std::span<const std::int32_t> prompt, int tokens, std::string_view label )
    {
        int generated = 0;

        (void)model.generate( prompt, [&]( std::int32_t )
            {
                if ( ++generated == 1 )
                {
                    model.setDispatchProfiling( true );
                }
            }, generateParams( tokens ) );

        model.setDispatchProfiling( false );

        printProfile( model.dispatchProfile(), tokens - 1, "a token",
            std::format( "{}: decode over {} tokens after {}", label, tokens - 1, prompt.size() ) );
    }

    struct DepthRates
    {
        double fill{ 0.0 };
        double extend{ 0.0 };
    };

    // Prefills `held` whole, then `held` followed by `added` with the held prefix reused, timing each to its first token.
    // The extension's rate is llama-bench's `-p added -d held`; the fill's is what a first message of that length costs.
    // With `profile`, only the extension is timed dispatch by dispatch.
    template<typename TModel>
    DepthRates depthRates( TModel& model, std::span<const std::int32_t> held, std::span<const std::int32_t> added, bool profile = false )
    {
        DepthRates rates;

        if ( !held.empty() )
        {
            rates.fill = prefillRate( model, held );
        }

        std::vector<std::int32_t> prompt( held.begin(), held.end() );
        prompt.insert( prompt.end(), added.begin(), added.end() );

        if ( profile )
        {
            model.setDispatchProfiling( true );
        }

        Clock::time_point first_token{};
        const Clock::time_point start = Clock::now();

        (void)model.generate( prompt, [&]( std::int32_t ) { first_token = Clock::now(); }, generateParams( 1, !held.empty() ) );

        if ( profile )
        {
            model.setDispatchProfiling( false );
        }

        rates.extend = static_cast<double>( added.size() ) / seconds( start, first_token );

        return rates;
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

    // Each depth warmed once, then timed `runs` times; with --profile-dispatches, one profiled extension per depth instead.
    template<typename TModel>
    void measureDepths( TModel& model, const Options& options, Samples& samples, std::string_view label, std::span<const std::int32_t> added )
    {
        for ( const dim_t depth : options.prefill_depths )
        {
            const std::vector<std::int32_t> held = randomTokens( depth, kPromptSeed + 3 );

            (void)depthRates( model, held, added );

            if ( options.profile_dispatches )
            {
                const DepthRates rates = depthRates( model, held, added, true );

                printProfile( model.dispatchProfile(), 1, "a prompt",
                    std::format( "{}: prefill {} after {} (profiled at {:.1f} tokens/s)", label, added.size(), depth, rates.extend ) );

                continue;
            }

            for ( int run = 0; run < options.runs; ++run )
            {
                const DepthRates rates = depthRates( model, held, added );

                samples.extend[ depth ].push_back( rates.extend );

                if ( depth > 0 )
                {
                    samples.fill[ depth ].push_back( rates.fill );
                }

                std::cout << std::format( "[ run ] {} run {}: prefill {} after {} {:.1f} tokens/s, the {} held {:.1f} tokens/s\n",
                    label, run, added.size(), depth, rates.extend, depth, rates.fill );
            }
        }
    }

    template<Family TFamily, TensorDataType TPrecision>
    void measureLoad( const Options& options, Samples& samples, std::string_view label )
    {
        using Model = typename FamilyModel<TFamily, TPrecision>::type;

        // Naming no weight format loads the one the weights are stored in.
        const DeploymentRequest request = DeploymentRequest().withContextLength( options.context ).withAllowedDevices( Device::Metal( 0 ) );

        const Clock::time_point start = Clock::now();
        auto model = Model::load( options.weights, request );
        const double load = seconds( start, Clock::now() );

        samples.load_seconds.push_back( load );
        std::cout << std::format( "[ load ] {} {}: {:.2f} s, context {}, prefill chunk {}, weights {}\n", label, precisionName( TPrecision ), load,
            model->getDeploymentPlan().contextLength(), model->getDeploymentPlan().prefillChunkRows(),
            weightQuantizationName( model->getDeploymentPlan().weightQuantization() ) );
        printMemory( std::format( "{} {} after load", label, precisionName( TPrecision ) ), model->getDeploymentPlan() );

        const std::vector<std::int32_t> prompt = randomTokens( options.prefill_tokens, kPromptSeed );

        if ( !options.prefill_depths.empty() )
        {
            measureDepths( *model, options, samples, label, prompt );
            printMemory( std::format( "{} {} after the runs", label, precisionName( TPrecision ) ), model->getDeploymentPlan() );

            return;
        }

        const std::vector<std::int32_t> one_token = randomTokens( 1, kPromptSeed + 1 );
        const std::vector<std::int32_t> deep = randomTokens( options.depth, kPromptSeed + 2 );

        (void)prefillRate( *model, prompt );
        (void)decodeRate( *model, one_token, options.decode_tokens );
        (void)decodeRate( *model, deep, options.decode_tokens );

        if ( options.profile_dispatches )
        {
            if ( options.profile_depths.empty() )
            {
                printDecodeProfile( *model, one_token, options.decode_tokens, label );
            }

            for ( const dim_t depth : options.profile_depths )
            {
                const std::vector<std::int32_t> held = depth == 0 ? one_token : randomTokens( depth, kPromptSeed + 2 );

                (void)decodeRate( *model, held, options.decode_tokens );
                printDecodeProfile( *model, held, options.decode_tokens, label );
            }

            return;
        }

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

    template<Family TFamily, TensorDataType TPrecision>
    void measurePlannerContext( const Options& options )
    {
        using Model = typename FamilyModel<TFamily, TPrecision>::type;

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

        if ( !options.prefill_depths.empty() )
        {
            for ( const auto& [depth, values] : samples.extend )
            {
                line( std::format( "prefill {} after {}", options.prefill_tokens, depth ), values, "tokens/s" );
            }

            for ( const auto& [depth, values] : samples.fill )
            {
                line( std::format( "prefill {} from empty", depth ), values, "tokens/s" );
            }
        } else
        {
            line( std::format( "prefill {}", options.prefill_tokens ), samples.prefill, "tokens/s" );
            line( std::format( "decode {}", options.decode_tokens ), samples.decode, "tokens/s" );
            line( std::format( "decode {} after {}", options.decode_tokens, options.depth ), samples.decode_at_depth, "tokens/s" );
        }

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
            } else if ( arg == "--family" )
            {
                const std::string_view family = next();

                if ( family == "llama" )
                {
                    options.family = Family::Llama;
                } else if ( family == "gemma" )
                {
                    options.family = Family::Gemma;
                } else
                {
                    throw std::invalid_argument( "--family is llama or gemma" );
                }
            } else if ( arg == "--context" )
            {
                options.context = std::stoll( std::string( next() ) );
            } else if ( arg == "--profile-dispatches" )
            {
                options.profile_dispatches = true;
            } else if ( arg == "--temperature" )
            {
                options.temperature = std::stof( std::string( next() ) );
            } else if ( arg == "--top-k" )
            {
                options.top_k = std::stoi( std::string( next() ) );
            } else if ( arg == "--top-p" )
            {
                options.top_p = std::stof( std::string( next() ) );
            } else if ( arg == "--rounds" )
            {
                options.rounds = std::stoi( std::string( next() ) );
            } else if ( arg == "--runs" )
            {
                options.runs = std::stoi( std::string( next() ) );
            } else if ( arg == "--depth" )
            {
                options.depth = std::stoll( std::string( next() ) );
            } else if ( arg == "--prefill-depths" )
            {
                for ( const auto part : std::views::split( next(), ',' ) )
                {
                    options.prefill_depths.push_back( std::stoll( std::string( std::string_view( part ) ) ) );
                }
            } else if ( arg == "--profile-depths" )
            {
                for ( const auto part : std::views::split( next(), ',' ) )
                {
                    options.profile_depths.push_back( std::stoll( std::string( std::string_view( part ) ) ) );
                }
            } else if ( arg == "--log-info" )
            {
                options.log_info = true;
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
            throw std::invalid_argument( "usage: mila-metal-model-timing --weights <weights> [--family llama|gemma] [--context 8192] "
                "[--rounds 3] [--runs 5] [--depth 4096] [--precision fp16|bf16] [--temperature 0] [--top-k 0] [--top-p 1] "
                "[--prefill-depths 0,4096,16384] [--profile-dispatches [--profile-depths 0,4096]] [--log-info] [--no-planner | --planner-only]" );
        }

        for ( const dim_t depth : options.profile_depths )
        {
            if ( depth < 0 || depth + options.decode_tokens > options.context )
            {
                throw std::invalid_argument( std::format( "--profile-depths: {} and the {} decoded do not fit a context of {}",
                    depth, options.decode_tokens, options.context ) );
            }
        }

        for ( const dim_t depth : options.prefill_depths )
        {
            if ( depth < 0 || depth + options.prefill_tokens > options.context )
            {
                throw std::invalid_argument( std::format( "--prefill-depths: {} and the {} added do not fit a context of {}",
                    depth, options.prefill_tokens, options.context ) );
            }
        }

        return options;
    }

    template<Family TFamily>
    void run( const Options& options )
    {
        Samples fp16;
        Samples bf16;

        for ( int round = 0; options.rates && round < options.rounds; ++round )
        {
            const std::string label = std::format( "round {}", round );

            auto measureFp16 = [&]()
            {
                if ( options.fp16 )
                {
                    measureLoad<TFamily, TensorDataType::FP16>( options, fp16, label );
                }
            };

            auto measureBf16 = [&]()
            {
                if ( options.bf16 )
                {
                    measureLoad<TFamily, TensorDataType::BF16>( options, bf16, label );
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
            measurePlannerContext<TFamily, TensorDataType::FP16>( options );
        }

        if ( options.planner && options.bf16 )
        {
            measurePlannerContext<TFamily, TensorDataType::BF16>( options );
        }
    }
}

int main( int argc, char** argv )
{
    try
    {
        const Options options = parse( argc, argv );
        sampling_options = options;

        Mila::initialize( 0, std::make_shared<Mila::Logging::ConsoleSink>(
            options.log_info ? Mila::Logging::LogLevel::Info : Mila::Logging::LogLevel::Warning ) );

        if ( options.family == Family::Gemma )
        {
            run<Family::Gemma>( options );
        } else
        {
            run<Family::Llama>( options );
        }

        return 0;
    } catch ( const std::exception& error )
    {
        std::cerr << "mila-metal-model-timing: " << error.what() << "\n";

        return 1;
    }
}
