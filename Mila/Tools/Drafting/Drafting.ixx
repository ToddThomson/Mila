/**
 * @file Drafting.ixx
 * @brief Measurements that decide whether a draft model pays on a target (Gemma4Mtp.md section 5.1).
 *
 * `verify-cost`: what checking K drafted tokens costs the target, against one decode, at a depth.
 * `drafter-parity`: the drafter's steps from the target's state, with every input dumped for HuggingFace's forward.
 * `acceptance`: how many drafts the target accepts. `speculate`: the greedy loop's rate against plain decoding.
 * `generate`: the same through GemmaModel::generate, the loop a program runs. `routing`: the experts R rows of the
 * 26B-A4B share.
 */

module;

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <utility>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

export module Tools.Drafting;

import Mila;
import Dnn.Samplers.TokenSampler;
import Dnn.Samplers.SamplingConfig;

#include "Measurement/LogLikelihoodHarness.h"
#include "Measurement/Pg19Books.h"

namespace Mila::Tools::Drafting
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using Mila::Deployment::DeploymentRequest;

    namespace fs = std::filesystem;

    using GemmaCuda = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

    // What GemmaModel's own dispatch builds for the Q4_0 12B: the target Google's drafter is paired with.
    using GemmaQ4_0Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy>;

    const DeviceId kDevice{ DeviceType::Cuda, 0 };

    // The same target with its global layers' cache in FP8, as KvCacheCompression::FP8 builds it.
    using GemmaQ4_0Fp8GlobalNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy, GemmaFeedForward::Dense,
        Quant::KvCache::PerTokenKvFp8<>>;

    // The 26B-A4B as ContextProfile measures it: routed, Q4_0 expert bank, with either global cache.
    using GemmaRoutedQ4_0Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy, GemmaFeedForward::Routed>;

    using GemmaRoutedQ4_0Fp8GlobalNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy, GemmaFeedForward::Routed,
        Quant::KvCache::PerTokenKvFp8<>>;

    using GemmaDrafterCuda = GemmaDrafter<DeviceType::Cuda, TensorDataType::BF16>;

    // The same drafter with its head at the 12B's six-bit table format, quantized on load (Gemma4Mtp.md decision 2).
    using GemmaDrafterSixBitHeadCuda = GemmaDrafter<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::NoWeightQuant, Quant::Weight::PerGroupInt6<32>>;

    struct Options
    {
        /// The 26B-A4B and its drafter instead of the 12B and its own.
        bool target_26b{ false };

        std::vector<dim_t> depths{ 8192, 65536 };
        int max_draft{ 8 };
        int repeats{ 20 };
        int decode_steps{ 64 };
        int prompt_tokens{ 1536 };
        int positions{ 256 };
        // Chat's sampling (Chat.Config.ixx) by default; Google's generation_config.json for Gemma 4 is 1.0, 64, 0.95.
        float temperature{ 0.8f };
        int top_k{ 40 };
        float top_p{ 1.0f };
        std::vector<int> drafts{ 2, 3, 4, 5 };
        int tokens{ 512 };
        int runs{ 3 };
        bool fp8_cache{ false };
        bool sample{ false };
        // generate: run r of a cell seeds the model's sampler with seed + r, so two builds sample the same replies.
        std::optional<std::uint64_t> seed;
        fs::path weights{ fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma4_12b_it_qat_q4_0.safetensors" };
        fs::path drafter{ fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma4_12b_it_qat_drafter_bf16.bin" };
        fs::path output{ "drafter_parity" };
        bool output_given{ false };
        /// The drafter's head at the six-bit table format instead of BF16.
        bool six_bit_drafter_head{ false };
    };

    void printUsage()
    {
        std::cout
            << "Drafting verify-cost [options]\n"
            << "  What a verify of K drafted tokens costs the Gemma 4 12B Q4_0 target, against one decode.\n"
            << "  --depths        Comma-separated context depths. Default: 8192,65536.\n"
            << "  --max-draft     Largest K measured; K runs from 1. Default: 8.\n"
            << "  --repeats       Timed repeats per cell; the median is reported. Default: 20.\n"
            << "  --decode-steps  Consecutive decodes per decode timing. Default: 64.\n"
            << "  --weights       Weights file. Default: Models/Gemma/gemma4_12b_it_qat_q4_0.safetensors.\n"
            << "\n"
            << "Drafting drafter-parity [options]\n"
            << "  The drafter's steps from the 12B's state after a PG-19 prompt, every input and output dumped for\n"
            << "  Tools/Converters/Gemma/hf_gemma_drafter_reference.py.\n"
            << "  --prompt-tokens Prompt length. Default: 1536, past the sliding window.\n"
            << "  --max-draft     Draft steps. Default: 8.\n"
            << "  --drafter       Drafter weights. Default: Models/Gemma/gemma4_12b_it_qat_drafter_bf16.bin.\n"
            << "  --output        Dump directory. Default: drafter_parity.\n"
            << "\n"
            << "Drafting acceptance [options]\n"
            << "  How many drafts the 12B accepts. The 12B decodes greedily; after each step the drafter drafts\n"
            << "  --max-draft tokens from its state, and a round's acceptance is how many lead the 12B's own tokens.\n"
            << "  Prompts: PG-19 prose, a C++ source file, and a chat question in Gemma's template.\n"
            << "  --positions     Rounds per prompt. Default: 256.\n"
            << "  --prompt-tokens Prose and code prompt length. Default: 1536.\n"
            << "  --temperature, --top-k, --top-p\n"
            << "                  The sampling the first draft's acceptance is measured at: the greedy draft's p(argmax q)\n"
            << "                  and a sampled draft's sum min(p, q), p and q both at these settings. Defaults: 0.8, 40, 1\n"
            << "                  (Chat's); Google's for Gemma 4 are 1.0, 64, 0.95.\n"
            << "  --kv-cache      bf16 | fp8 (the target's global layers). Default: bf16.\n"
            << "\n"
            << "Drafting speculate [options]\n"
            << "  Greedy decoding with the drafter against greedy decoding without it, on the acceptance prompts: each\n"
            << "  round drafts K tokens, verifies them in one multi-token decode and keeps the agreed prefix.\n"
            << "  --drafts        Comma-separated K, each 1 to 7. Default: 2,3,4,5.\n"
            << "  --sample        Sample at --temperature, --top-k and --top-p instead of decoding greedily.\n"
            << "  --tokens        Tokens generated after the prompt's first. Default: 512.\n"
            << "  --runs          Timed runs per cell; the median is reported. Default: 3.\n"
            << "  --prompt-tokens Prose and code prompt length. Default: 1536.\n"
            << "  --kv-cache      bf16 | fp8 (the target's global layers). Default: bf16.\n"
            << "\n"
            << "Drafting generate [options]\n"
            << "  The same comparison through GemmaModel::generate: the model loaded without a draft model, then with\n"
            << "  one at each K. Takes --drafts, --tokens, --runs, --prompt-tokens and --kv-cache as speculate does.\n"
            << "  --seed          Seed run r of every cell with seed + r and print a hash of each cell's tokens, so two\n"
            << "                  builds' sampled runs can be timed on the same replies and checked token for token.\n"
            << "\n"
            << "Drafting routing --target 26b [options]\n"
            << "  How many distinct experts R = 1..8 consecutive rows of a greedy reply route to, per layer: the union a\n"
            << "  verify of R rows reads. Takes --tokens, --prompt-tokens and --kv-cache.\n"
            << "  --output        Also write each reply's experts, INT32 [rows, layers, top_k], to <dir>/<prompt>.routing.\n"
            << "\n"
            << "acceptance and speculate take --drafter-head bf16 | int6 (default bf16): the drafter's head as shipped, or\n"
            << "at the six-bit table format, quantized on load.\n"
            << "\n"
            << "Every mode takes --target 12b | 26b (default 12b): the 26B-A4B runs Q4_0, quantized on load from\n"
            << "Models/Gemma/gemma4_26b_a4b_it_qat_bf16.bin, with its drafter gemma4_26b_a4b_it_qat_drafter_bf16.bin.\n";
    }

    int parseInt( std::string_view value, std::string_view flag )
    {
        int result = 0;
        const auto status = std::from_chars( value.data(), value.data() + value.size(), result );

        if ( status.ec != std::errc{} || status.ptr != value.data() + value.size() || result <= 0 )
            throw std::invalid_argument( std::format( "{} expects a positive integer, got '{}'", flag, value ) );

        return result;
    }

    Options parseOptions( int argc, char** argv )
    {
        Options options;
        bool weights_given = false;
        bool drafter_given = false;

        for ( int i = 2; i < argc; ++i )
        {
            const std::string_view arg = argv[ i ];

            const auto value = [&]() -> std::string_view
            {
                if ( i + 1 >= argc )
                    throw std::invalid_argument( std::format( "{} requires a value", arg ) );

                return argv[ ++i ];
            };

            if ( arg == "--depths" )
            {
                options.depths.clear();
                std::string_view list = value();

                while ( !list.empty() )
                {
                    const std::size_t comma = list.find( ',' );
                    options.depths.push_back( parseInt( list.substr( 0, comma ), "--depths" ) );
                    list = comma == std::string_view::npos ? std::string_view{} : list.substr( comma + 1 );
                }
            }
            else if ( arg == "--max-draft" )
            {
                options.max_draft = parseInt( value(), "--max-draft" );
            }
            else if ( arg == "--drafts" )
            {
                options.drafts.clear();
                std::string_view list = value();

                while ( !list.empty() )
                {
                    const std::size_t comma = list.find( ',' );
                    const int draft = parseInt( list.substr( 0, comma ), "--drafts" );

                    // A verify is K + 1 rows, and a network decodes at most 8 tokens in one call.
                    if ( draft > 7 )
                        throw std::invalid_argument( std::format( "--drafts: K is at most 7, got {}", draft ) );

                    options.drafts.push_back( draft );
                    list = comma == std::string_view::npos ? std::string_view{} : list.substr( comma + 1 );
                }
            }
            else if ( arg == "--tokens" )
            {
                options.tokens = parseInt( value(), "--tokens" );
            }
            else if ( arg == "--sample" )
            {
                options.sample = true;
            }
            else if ( arg == "--runs" )
            {
                options.runs = parseInt( value(), "--runs" );
            }
            else if ( arg == "--seed" )
            {
                options.seed = static_cast<std::uint64_t>( parseInt( value(), "--seed" ) );
            }
            else if ( arg == "--repeats" )
            {
                options.repeats = parseInt( value(), "--repeats" );
            }
            else if ( arg == "--decode-steps" )
            {
                options.decode_steps = parseInt( value(), "--decode-steps" );
            }
            else if ( arg == "--weights" )
            {
                options.weights = value();
                weights_given = true;
            }
            else if ( arg == "--prompt-tokens" )
            {
                options.prompt_tokens = parseInt( value(), "--prompt-tokens" );
            }
            else if ( arg == "--drafter" )
            {
                options.drafter = value();
                drafter_given = true;
            }
            else if ( arg == "--output" )
            {
                options.output = value();
                options.output_given = true;
            }
            else if ( arg == "--positions" )
            {
                options.positions = parseInt( value(), "--positions" );
            }
            else if ( arg == "--temperature" )
            {
                const std::string_view text = value();
                const auto status = std::from_chars( text.data(), text.data() + text.size(), options.temperature );

                if ( status.ec != std::errc{} || options.temperature <= 0.0f )
                    throw std::invalid_argument( std::format( "--temperature expects a positive number, got '{}'", text ) );
            }
            else if ( arg == "--top-k" )
            {
                options.top_k = parseInt( value(), "--top-k" );
            }
            else if ( arg == "--top-p" )
            {
                const std::string_view text = value();
                const auto status = std::from_chars( text.data(), text.data() + text.size(), options.top_p );

                if ( status.ec != std::errc{} || options.top_p <= 0.0f || options.top_p > 1.0f )
                    throw std::invalid_argument( std::format( "--top-p expects a number in (0, 1], got '{}'", text ) );
            }
            else if ( arg == "--target" )
            {
                const std::string_view target = value();

                if ( target != "12b" && target != "26b" )
                    throw std::invalid_argument( std::format( "--target expects 12b or 26b, got '{}'", target ) );

                options.target_26b = target == "26b";
            }
            else if ( arg == "--drafter-head" )
            {
                const std::string_view format = value();

                if ( format != "bf16" && format != "int6" )
                    throw std::invalid_argument( std::format( "--drafter-head expects bf16 or int6, got '{}'", format ) );

                options.six_bit_drafter_head = format == "int6";
            }
            else if ( arg == "--kv-cache" )
            {
                const std::string_view format = value();

                if ( format != "bf16" && format != "fp8" )
                    throw std::invalid_argument( std::format( "--kv-cache expects bf16 or fp8, got '{}'", format ) );

                options.fp8_cache = format == "fp8";
            }
            else
            {
                throw std::invalid_argument( std::format( "unknown option '{}'", arg ) );
            }
        }

        // The 26B-A4B quantizes to Q4_0 on load from its full-precision weights, as ContextProfile runs it.
        if ( options.target_26b )
        {
            const fs::path gemma = fs::path( MILA_DATA_DIR ) / "Models" / "Gemma";

            if ( !weights_given )
                options.weights = gemma / "gemma4_26b_a4b_it_qat_bf16.bin";

            if ( !drafter_given )
                options.drafter = gemma / "gemma4_26b_a4b_it_qat_drafter_bf16.bin";
        }

        return options;
    }

    /// Tokens drawn uniformly from the vocabulary with a fixed seed, so every run does the same work.
    std::vector<std::int32_t> syntheticTokens( dim_t vocabulary_size, std::size_t length )
    {
        std::mt19937 generator( 20261004u );
        std::uniform_int_distribution<std::int32_t> vocabulary( 0, static_cast<std::int32_t>( vocabulary_size ) - 1 );
        std::vector<std::int32_t> tokens( length );

        for ( std::int32_t& token : tokens )
            token = vocabulary( generator );

        return tokens;
    }

    double median( std::vector<double> values )
    {
        std::sort( values.begin(), values.end() );

        const std::size_t middle = values.size() / 2;

        return values.size() % 2 == 1 ? values[ middle ] : 0.5 * ( values[ middle - 1 ] + values[ middle ] );
    }

    template<typename TNetwork, typename TAction>
    double timedMilliseconds( TNetwork& network, TAction&& action )
    {
        network.synchronize();

        const auto start = std::chrono::steady_clock::now();

        action();
        network.synchronize();

        return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - start ).count();
    }

    template<typename TNetwork>
    void requireRewind( TNetwork& network, dim_t position )
    {
        if ( !network.rewindKvCache( position ) )
            throw std::runtime_error( std::format( "the network refused a rewind to {}", position ) );
    }

    /**
     * @brief Per-token time of `steps` consecutive replayed decodes from `depth`, as generate() runs them.
     *
     * The host enqueues every step before it waits, so the time is the device's, not the launch gaps'.
     */
    template<typename TNetwork>
    double decodeMilliseconds( TNetwork& network, const typename TNetwork::TokenIndexType& token, dim_t depth, int steps )
    {
        requireRewind( network, depth );

        return timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < steps; ++step )
                network.decode( token, depth + step );
        } ) / steps;
    }

    template<typename TNetwork>
    void measureVerifyCost( const Options& options )
    {
        const dim_t deepest = *std::max_element( options.depths.begin(), options.depths.end() );
        const dim_t context_length = deepest + std::max<dim_t>( options.max_draft + 1, options.decode_steps ) + 1;

        Serialization::WeightsReader reader( options.weights );
        auto config = GemmaCuda::configFromMetadata( reader.getWeightsMetadata() );

        // One head window holds every verified row, so the head runs once over all of them, as a verify would.
        config.withLogLikelihoodWindow( options.max_draft + 1 );

        PrefillChunking chunking;
        auto network = Measurement::buildMeasuredNetwork<TNetwork>(
            options.weights, config, kDevice, context_length, &chunking );
        network->setDecodeReplay( true );

        const auto tokens = syntheticTokens( config.getVocabSize(), static_cast<std::size_t>( context_length ) );
        const auto decode_token = Measurement::deviceTokens( *network, { tokens[ 0 ] } );

        std::cout << std::format( "weights {}\ncontext {}, prefill chunk {}, {} repeats, median reported\n\n",
            options.weights.string(), context_length, chunking.chunk_rows, options.repeats );
        std::cout << "| depth | K | decode ms | verify ms, last-row head | verify ms, head per row | per-row / decode |\n"
                  << "|---|---|---|---|---|---|\n";

        for ( const dim_t depth : options.depths )
        {
            const std::vector<std::int32_t> prompt( tokens.begin(), tokens.begin() + depth );
            network->prefill( Measurement::deviceTokens( *network, prompt ) );

            // Unmeasured: primes, records and checks the decode replay.
            decodeMilliseconds( *network, decode_token, depth, 4 );

            std::vector<double> decode_runs;

            for ( int repeat = 0; repeat < options.repeats; ++repeat )
                decode_runs.push_back( decodeMilliseconds( *network, decode_token, depth, options.decode_steps ) );

            const double decode_ms = median( decode_runs );

            for ( int draft = 1; draft <= options.max_draft; ++draft )
            {
                const std::vector<std::int32_t> sequence( tokens.begin(), tokens.begin() + depth + draft + 1 );
                const auto input = Measurement::deviceTokens( *network, sequence );

                std::vector<double> last_row_runs;
                std::vector<double> per_row_runs;

                for ( int repeat = 0; repeat <= options.repeats; ++repeat )
                {
                    requireRewind( *network, depth );
                    const double last_row = timedMilliseconds( *network, [&] { network->prefillFrom( input, depth ); } );

                    requireRewind( *network, depth );
                    const double per_row = timedMilliseconds( *network, [&] { network->sequenceLogLikelihoodFrom( input, depth ); } );

                    // The first repeat warms the cell's plans and buffers.
                    if ( repeat > 0 )
                    {
                        last_row_runs.push_back( last_row );
                        per_row_runs.push_back( per_row );
                    }
                }

                const double per_row_ms = median( per_row_runs );

                std::cout << std::format( "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |\n",
                    depth, draft, decode_ms, median( last_row_runs ), per_row_ms, per_row_ms / decode_ms );
            }
        }
    }

    void writeFile( const fs::path& path, const void* data, std::size_t bytes )
    {
        std::FILE* file = std::fopen( path.string().c_str(), "wb" );

        if ( file == nullptr || std::fwrite( data, 1, bytes, file ) != bytes )
            throw std::runtime_error( std::format( "cannot write {}", path.string() ) );

        std::fclose( file );
    }

    template<typename TTensor>
    void writeTensor( const fs::path& path, const TTensor& tensor, IExecutionContext* context )
    {
        context->synchronize();

        auto host = toHost<TensorDataType::FP32>( tensor );
        writeFile( path, host.data(), static_cast<std::size_t>( host.size() ) * sizeof( float ) );
    }

    /**
     * @brief A BF16 cache's rows for positions [first, end), per key/value head in position order, as raw BF16.
     *
     * Position q lives in row q % capacity, as every Mila cache lays it out.
     */
    std::vector<std::uint16_t> cacheRows( const void* device, const KvCacheView& view, dim_t first, dim_t end )
    {
        if ( view.fp8 )
            throw std::invalid_argument( "drafter-parity dumps a BF16 cache; run the target with a BF16 cache" );

        const std::size_t row = static_cast<std::size_t>( view.head_size );
        std::vector<std::uint16_t> whole( static_cast<std::size_t>( view.num_kv_heads ) * view.capacity * row );

        if ( cudaMemcpy( whole.data(), device, whole.size() * sizeof( std::uint16_t ), cudaMemcpyDeviceToHost ) != cudaSuccess )
            throw std::runtime_error( "cannot copy a KV cache to the host" );

        std::vector<std::uint16_t> ordered;
        ordered.reserve( static_cast<std::size_t>( view.num_kv_heads ) * static_cast<std::size_t>( end - first ) * row );

        for ( int head = 0; head < view.num_kv_heads; ++head )
        {
            for ( dim_t position = first; position < end; ++position )
            {
                const std::size_t offset = ( static_cast<std::size_t>( head ) * view.capacity
                    + static_cast<std::size_t>( position % view.capacity ) ) * row;

                ordered.insert( ordered.end(), whole.begin() + offset, whole.begin() + offset + row );
            }
        }

        return ordered;
    }

    void dumpCache( const fs::path& directory, std::string_view name, const KvCacheView& view, dim_t first, dim_t end )
    {
        const auto keys = cacheRows( view.keys, view, first, end );
        const auto values = cacheRows( view.values, view, first, end );

        writeFile( directory / std::format( "{}_k.bf16", name ), keys.data(), keys.size() * sizeof( std::uint16_t ) );
        writeFile( directory / std::format( "{}_v.bf16", name ), values.data(), values.size() * sizeof( std::uint16_t ) );
    }

    /// The first book of the PG-19 test split, as tokens.
    std::vector<std::int32_t> bookPrompt( int length )
    {
        const fs::path books = Measurement::pg19TestPath( fs::path( MILA_DATA_DIR ) );
        std::vector<fs::path> files;

        for ( const auto& entry : fs::directory_iterator( books ) )
            files.push_back( entry.path() );

        if ( files.empty() )
            throw std::runtime_error( std::format( "no PG-19 books under {}", books.string() ) );

        std::sort( files.begin(), files.end() );

        auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma_tokenizer.bin" );
        const auto encoded = tokenizer->encode(
            Measurement::joinWraps( Measurement::readBook( files.front(), static_cast<std::size_t>( length ) * 8 ) ) );
        std::vector<std::int32_t> tokens( encoded.begin(), encoded.end() );

        if ( static_cast<int>( tokens.size() ) < length )
            throw std::runtime_error( std::format( "{} holds fewer than {} tokens", files.front().string(), length ) );

        tokens.resize( static_cast<std::size_t>( length ) );

        return tokens;
    }

    /**
     * @brief Run the target over a prompt, then the drafter K steps from its state, dumping what HuggingFace needs.
     *
     * Each step's inputs are dumped as Mila produced them -- the target's scaled embedding of the step's token and
     * the hidden state it joins -- so the reference reproduces the drafter's forward alone (Gemma4Mtp.md 5.1).
     */
    template<typename TNetwork>
    void dumpDrafterParity( const Options& options )
    {
        const dim_t prompt_length = options.prompt_tokens;
        const dim_t context_length = prompt_length + options.max_draft + 64;

        Serialization::WeightsReader reader( options.weights );
        const GemmaConfig target_config = GemmaCuda::configFromMetadata( reader.getWeightsMetadata() );

        auto network = Measurement::buildMeasuredNetwork<TNetwork>(
            options.weights, target_config, kDevice, context_length );
        IExecutionContext* context = network->getExecutionContext();

        Serialization::WeightsReader drafter_reader( options.drafter );
        const GemmaConfig drafter_config = GemmaDrafterCuda::configFromMetadata(
            drafter_reader.getWeightsMetadata(), target_config );

        auto drafter = std::make_shared<GemmaDrafterCuda>( "drafter", drafter_config, target_config.getModelDim(), context );
        drafter->build( BuildContext( shape_t{ 1, context_length, drafter_config.getModelDim() }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( kDevice ) ) );
        drafter->loadParameters( drafter_reader );

        const std::vector<std::int32_t> prompt = bookPrompt( options.prompt_tokens );
        std::int32_t token = Measurement::argMax( Measurement::hostLogits( *network,
            network->prefill( Measurement::deviceTokens( *network, prompt ) ) ) );

        fs::create_directories( options.output );

        const KvCacheView sliding = network->lastSlidingLayerCache();
        const KvCacheView global = network->lastGlobalLayerCache();
        const dim_t sliding_first = std::max<dim_t>( 0, prompt_length - sliding.window );

        dumpCache( options.output, "sliding", sliding, sliding_first, prompt_length );
        dumpCache( options.output, "global", global, 0, prompt_length );

        // Every draft step runs at the position of the token the target chose and has not processed.
        context->setDecodePosition( prompt_length );

        std::vector<std::int32_t> drafts;
        const auto* hidden = &network->finalNormedHidden();
        drafter->startFrom( *hidden );

        for ( int step = 0; step < options.max_draft; ++step )
        {
            const auto token_tensor = Measurement::deviceTokens( *network, { token } );
            auto& embedding = network->embed( token_tensor );

            writeTensor( options.output / std::format( "embedding_{}.f32", step ), embedding, context );
            writeTensor( options.output / std::format( "hidden_{}.f32", step ), *hidden, context );

            auto& logits = drafter->decode( embedding, prompt_length, sliding, global );

            const std::vector<float> host_logits = Measurement::hostLogits( *network, logits );
            writeFile( options.output / std::format( "logits_{}.f32", step ), host_logits.data(), host_logits.size() * sizeof( float ) );

            token = Measurement::argMax( host_logits );
            drafts.push_back( token );
            hidden = &drafter->nextHidden();
        }

        std::string draft_list;

        for ( std::size_t i = 0; i < drafts.size(); ++i )
            draft_list += std::format( "{}{}", i == 0 ? "" : ", ", drafts[ i ] );

        const std::string manifest = std::format(
            "{{\n  \"position\": {},\n  \"steps\": {},\n  \"target_model_dim\": {},\n  \"vocab_size\": {},\n"
            "  \"sliding\": {{ \"num_kv_heads\": {}, \"head_size\": {}, \"first\": {}, \"count\": {} }},\n"
            "  \"global\": {{ \"num_kv_heads\": {}, \"head_size\": {}, \"first\": 0, \"count\": {} }},\n"
            "  \"drafts\": [ {} ]\n}}\n",
            prompt_length, options.max_draft, target_config.getModelDim(), drafter_config.getVocabSize(),
            sliding.num_kv_heads, sliding.head_size, sliding_first, prompt_length - sliding_first,
            global.num_kv_heads, global.head_size, prompt_length, draft_list );

        writeFile( options.output / "manifest.json", manifest.data(), manifest.size() );

        std::cout << std::format( "position {}, drafts [ {} ], dumped to {}\n", prompt_length, draft_list, options.output.string() );
    }

    struct PromptCase
    {
        std::string name;
        std::vector<std::int32_t> tokens;
    };

    /**
     * @brief Three chat turns, each answered in Gemma's template: continuing prose, explaining code, explaining a concept.
     *
     * Each is a request an instruct model answers as Chat or an agent would ask it. A raw-text continuation is not:
     * an instruct model run greedily on one falls into repetition, which any drafter predicts.
     */
    std::vector<PromptCase> acceptancePrompts( int length )
    {
        auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma_tokenizer.bin" );

        // The passage's first `length` tokens, as text, for a turn that quotes it.
        const auto passage = [&]( const std::string& text )
        {
            const auto encoded = tokenizer->encode( text );
            std::vector<std::int32_t> tokens( encoded.begin(), encoded.end() );

            if ( static_cast<int>( tokens.size() ) > length )
                tokens.resize( static_cast<std::size_t>( length ) );

            return tokenizer->decode( tokens );
        };

        const auto turn = [&]( const std::string& request )
        {
            const std::vector<Dnn::Conversation::Turn> history{ { Dnn::Conversation::Role::User, request } };
            const auto encoded = tokenizer->encode( Dnn::Gemma::formatPrompt( history ) );

            return std::vector<std::int32_t>( encoded.begin(), encoded.end() );
        };

        const fs::path books = Measurement::pg19TestPath( fs::path( MILA_DATA_DIR ) );
        std::vector<fs::path> files;

        for ( const auto& entry : fs::directory_iterator( books ) )
            files.push_back( entry.path() );

        std::sort( files.begin(), files.end() );

        const std::size_t characters = static_cast<std::size_t>( length ) * 8;
        const fs::path source = fs::path( MILA_DATA_DIR ).parent_path() / "Mila" / "Src" / "Dnn" / "Components" / "Linear" / "Linear.ixx";

        std::vector<PromptCase> cases;

        cases.push_back( { "prose", turn( "Here is the opening of a novel:\n\n"
            + passage( Measurement::joinWraps( Measurement::readBook( files.front(), characters ) ) )
            + "\n\nContinue the story for several paragraphs, in the same style." ) } );

        cases.push_back( { "code", turn( "Here is a C++ source file:\n\n```cpp\n"
            + passage( Measurement::readBook( source, characters ) )
            + "\n```\n\nExplain what this code does, then suggest one improvement and show it." ) } );

        cases.push_back( { "chat", turn( "Explain how a sliding-window KV cache works in a transformer, why it saves "
            "memory at long context, and what it costs. Use a short example." ) } );

        return cases;
    }

    /// Softmax of logits / temperature, after the cap the target's sampler applies (zero for none).
    std::vector<double> distribution( const std::vector<float>& logits, float temperature, float softcap )
    {
        std::vector<double> probabilities( logits.size() );
        double peak = -1e300;

        for ( std::size_t i = 0; i < logits.size(); ++i )
        {
            const double capped = softcap > 0.0f ? softcap * std::tanh( logits[ i ] / softcap ) : logits[ i ];
            probabilities[ i ] = capped / temperature;
            peak = std::max( peak, probabilities[ i ] );
        }

        double total = 0.0;

        for ( double& value : probabilities )
        {
            value = std::exp( value - peak );
            total += value;
        }

        for ( double& value : probabilities )
            value /= total;

        return probabilities;
    }

    /**
     * @brief The distribution the device sampler draws from: the cap, the temperature, the top_k largest kept, then the
     *        top_p nucleus of those.
     *
     * Greedy puts all of it on the argmax. The device's top-k threshold is found by search rather than exactly, so
     * the two can differ at a tie on the threshold.
     */
    std::vector<double> samplerDistribution( const std::vector<float>& logits, const SamplingParams& sampling, float softcap )
    {
        if ( sampling.temperature <= 0.0f || sampling.top_k == 1 )
        {
            std::vector<double> point( logits.size(), 0.0 );
            point[ static_cast<std::size_t>( Measurement::argMax( logits ) ) ] = 1.0;

            return point;
        }

        std::vector<double> probabilities = distribution( logits, sampling.temperature, softcap );

        if ( sampling.top_k > 0 && static_cast<std::size_t>( sampling.top_k ) < probabilities.size() )
        {
            std::vector<double> ordered( probabilities );
            std::nth_element( ordered.begin(), ordered.begin() + ( sampling.top_k - 1 ), ordered.end(), std::greater<double>() );
            const double threshold = ordered[ static_cast<std::size_t>( sampling.top_k - 1 ) ];
            double total = 0.0;

            for ( double& value : probabilities )
            {
                value = value < threshold ? 0.0 : value;
                total += value;
            }

            for ( double& value : probabilities )
                value /= total;
        }

        // The nucleus over what top-k kept: the smallest top set whose mass reaches top_p, the boundary value kept.
        if ( sampling.top_p < 1.0f )
        {
            std::vector<double> ordered( probabilities );
            std::sort( ordered.begin(), ordered.end(), std::greater<double>() );

            double mass = 0.0;
            double threshold = ordered.front();

            for ( double value : ordered )
            {
                threshold = value;
                mass += value;

                if ( mass >= sampling.top_p )
                    break;
            }

            double total = 0.0;

            for ( double& value : probabilities )
            {
                value = value < threshold ? 0.0 : value;
                total += value;
            }

            for ( double& value : probabilities )
                value /= total;
        }

        return probabilities;
    }

    std::string describeSampling( const Options& options )
    {
        return std::format( "temperature {}, top-k {}, top-p {}", options.temperature, options.top_k, options.top_p );
    }

    struct AcceptanceResult
    {
        std::vector<int> accepted;          // per round, at the longest K
        std::vector<double> first_greedy;   // per round, p(argmax q) for the first draft
        std::vector<double> first_sampled;  // per round, sum min(p, q) for the first draft
        std::vector<std::int32_t> continuation;
    };

    // Gemma 4's end-of-sequence and end-of-turn ids, as the drafter's generation_config.json lists them.
    constexpr std::int32_t kStopTokens[] = { 1, 106 };

    bool isStopToken( std::int32_t token )
    {
        return std::find( std::begin( kStopTokens ), std::end( kStopTokens ), token ) != std::end( kStopTokens );
    }

    /**
     * @brief Greedy decode from `prompt`, with a draft round of `max_draft` steps from the target's state before each step.
     *
     * The drafter reads the target's caches and writes nothing, so the target's decode is the plain one. A round at
     * position q starts from the token the target chose for q and its hidden state at q - 1, as the loop's would.
     */
    template<typename TNetwork, typename TDrafter>
    AcceptanceResult measureAcceptance( TNetwork& network, TDrafter& drafter, const std::vector<std::int32_t>& prompt,
        int rounds, int max_draft, const SamplingParams& sampling, float softcap )
    {
        IExecutionContext* context = network.getExecutionContext();
        const KvCacheView sliding = network.lastSlidingLayerCache();
        const KvCacheView global = network.lastGlobalLayerCache();

        std::vector<std::int32_t> target{ Measurement::argMax( Measurement::hostLogits( network,
            network.prefill( Measurement::deviceTokens( network, prompt ) ) ) ) };
        std::vector<std::vector<std::int32_t>> drafts;
        AcceptanceResult result;

        const dim_t start = static_cast<dim_t>( prompt.size() );

        // A round counts only while the target is still writing: the reply's own end stops the measurement, and the
        // max_draft decodes past the last round supply the tokens its drafts are checked against.
        for ( int round = 0; round < rounds + max_draft; ++round )
        {
            if ( round < rounds && isStopToken( target.back() ) )
                rounds = round;

            const dim_t position = start + round;
            std::vector<float> first_draft_logits;

            if ( round < rounds )
            {
                context->setDecodePosition( position );

                std::int32_t token = target.back();
                std::vector<std::int32_t> chain;

                drafter.startFrom( network.finalNormedRow( 0 ) );

                for ( int step = 0; step < max_draft; ++step )
                {
                    auto& embedding = network.embed( Measurement::deviceTokens( network, { token } ) );
                    std::vector<float> logits = Measurement::hostLogits( network,
                        drafter.decode( embedding, position, sliding, global ) );

                    token = Measurement::argMax( logits );
                    chain.push_back( token );

                    if ( step == 0 )
                        first_draft_logits = std::move( logits );
                }

                drafts.push_back( std::move( chain ) );
            }

            const std::vector<float> target_logits = Measurement::hostLogits( network,
                network.decode( Measurement::deviceTokens( network, { target.back() } ), position ) );

            if ( round < rounds )
            {
                const auto p = samplerDistribution( target_logits, sampling, softcap );
                const auto q = samplerDistribution( first_draft_logits, sampling, 0.0f );
                double overlap = 0.0;

                for ( std::size_t i = 0; i < p.size(); ++i )
                    overlap += std::min( p[ i ], q[ i ] );

                result.first_greedy.push_back( p[ static_cast<std::size_t>( Measurement::argMax( first_draft_logits ) ) ] );
                result.first_sampled.push_back( overlap );
            }

            target.push_back( Measurement::argMax( target_logits ) );
        }

        for ( int round = 0; round < rounds; ++round )
        {
            int accepted = 0;

            while ( accepted < max_draft && drafts[ round ][ accepted ] == target[ round + 1 + accepted ] )
                ++accepted;

            result.accepted.push_back( accepted );
        }

        result.first_greedy.resize( static_cast<std::size_t>( rounds ) );
        result.first_sampled.resize( static_cast<std::size_t>( rounds ) );
        result.continuation.assign( target.begin(), target.begin() + rounds );

        return result;
    }

    /// Per-step time of `steps` draft steps on a fixed input, and of `steps` target decodes, after `prompt`.
    template<typename TNetwork, typename TDrafter>
    std::pair<double, double> stepTimes( TNetwork& network, TDrafter& drafter, const std::vector<std::int32_t>& prompt,
        int steps )
    {
        const KvCacheView sliding = network.lastSlidingLayerCache();
        const KvCacheView global = network.lastGlobalLayerCache();
        const auto token = Measurement::deviceTokens( network, { 0 } );
        const dim_t position = static_cast<dim_t>( prompt.size() );

        network.prefill( Measurement::deviceTokens( network, prompt ) );

        network.getExecutionContext()->setDecodePosition( position );
        auto& embedding = network.embed( token );

        drafter.startFrom( network.finalNormedRow( 0 ) );
        drafter.decode( embedding, position, sliding, global );

        const double draft_ms = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < steps; ++step )
                drafter.decode( embedding, position, sliding, global );
        } ) / steps;

        requireRewind( network, position );

        const double decode_ms = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < steps; ++step )
                network.decode( token, position + step );
        } ) / steps;

        return { draft_ms, decode_ms };
    }

    template<typename TNetwork, typename TDrafter>
    void measureAcceptanceOn( const Options& options )
    {
        const auto prompts = acceptancePrompts( options.prompt_tokens );

        dim_t longest = 0;

        for ( const auto& prompt : prompts )
            longest = std::max<dim_t>( longest, static_cast<dim_t>( prompt.tokens.size() ) );

        const dim_t context_length = longest + options.positions + options.max_draft + 64;

        Serialization::WeightsReader reader( options.weights );
        const GemmaConfig target_config = GemmaCuda::configFromMetadata( reader.getWeightsMetadata() );

        auto network = Measurement::buildMeasuredNetwork<TNetwork>( options.weights, target_config, kDevice, context_length );

        Serialization::WeightsReader drafter_reader( options.drafter );
        auto drafter = std::make_shared<TDrafter>( "drafter",
            TDrafter::configFromMetadata( drafter_reader.getWeightsMetadata(), target_config ),
            target_config.getModelDim(), network->getExecutionContext() );
        drafter->build( BuildContext( shape_t{ 1, context_length, drafter->getConfig().getModelDim() }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( kDevice ) ) );
        drafter->loadParameters( drafter_reader );

        const float softcap = target_config.getFinalLogitSoftcapping();

        std::cout << std::format( "weights {}\ndrafter {}, {} head\ncache {}, {} rounds per prompt, first draft's acceptance at {}\n\n",
            options.weights.string(), options.drafter.string(), options.six_bit_drafter_head ? "six-bit" : "BF16",
            options.fp8_cache ? "FP8 global layers" : "BF16",
            options.positions, describeSampling( options ) );

        std::cout << "| prompt | tokens | rounds | E[accepted], K = 1..8 | first draft greedy, p(argmax q) | first draft "
            "sampled, sum min(p, q) |\n|---|---|---|---|---|---|\n";

        std::vector<std::pair<std::string, std::vector<std::int32_t>>> continuations;

        for ( const auto& prompt : prompts )
        {
            const AcceptanceResult result = measureAcceptance( *network, *drafter, prompt.tokens,
                options.positions, options.max_draft, SamplingParams{ options.temperature, options.top_k, options.top_p },
                softcap );

            std::string by_draft;

            for ( int k = 1; k <= options.max_draft; ++k )
            {
                double total = 0.0;

                for ( int accepted : result.accepted )
                    total += std::min( accepted, k );

                by_draft += std::format( "{}{:.2f}", k == 1 ? "" : " / ", total / result.accepted.size() );
            }

            double greedy = 0.0;
            double sampled = 0.0;

            for ( double value : result.first_greedy )
                greedy += value;

            for ( double value : result.first_sampled )
                sampled += value;

            std::cout << std::format( "| {} | {} | {} | {} | {:.3f} | {:.3f} |\n", prompt.name, prompt.tokens.size(),
                result.accepted.size(), by_draft, greedy / result.first_greedy.size(), sampled / result.first_sampled.size() );

            continuations.push_back( { prompt.name, result.continuation } );
        }

        // What each measurement decoded: acceptance on a repeating continuation measures the repetition.
        auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma_tokenizer.bin" );

        for ( const auto& [name, tokens] : continuations )
        {
            std::cout << std::format( "\n--- {} continuation ({} tokens) ---\n{}\n", name, tokens.size(),
                tokenizer->decode( tokens ) );
        }

        const auto [draft_ms, decode_ms] = stepTimes( *network, *drafter, prompts.front().tokens, 64 );

        std::cout << std::format( "\ndraft step {:.3f} ms, target decode {:.3f} ms ({:.3f} of a decode), at position {}\n",
            draft_ms, decode_ms, draft_ms / decode_ms, prompts.front().tokens.size() );
    }

    using SamplerCuda = TokenSampler<DeviceType::Cuda, TensorDataType::BF16>;
    using HostTokens = Tensor<TensorDataType::INT32, CpuMemoryResource>;

    const SamplingParams kGreedy{ 0.0f, 1, 1.0f };

    /// The sampling at --temperature, --top-k and --top-p, or greedy.
    SamplingParams samplingOf( const Options& options )
    {
        return options.sample ? SamplingParams{ options.temperature, options.top_k, options.top_p } : kGreedy;
    }

    /// One element of a device token sequence, as the [1, 1] tensor a pass reads or a sampler writes.
    template<typename TTokens>
    TTokens tokenSlot( const TTokens& tokens, dim_t index )
    {
        return tokens.view( shape_t{ 1, 1 }, index );
    }

    template<typename TTokens>
    void setToken( TTokens& device, std::int32_t id, IExecutionContext* context )
    {
        HostTokens host( Device::Cpu(), shape_t{ 1, 1 } );
        host.data()[ 0 ] = id;
        copy( host, device, context );
        context->synchronize();
    }

    template<typename TTokens>
    std::vector<std::int32_t> readTokens( const TTokens& device, IExecutionContext* context )
    {
        HostTokens host( Device::Cpu(), device.shape() );
        copy( device, host, context );
        context->synchronize();

        return std::vector<std::int32_t>( host.data(), host.data() + host.size() );
    }

    struct Generation
    {
        std::vector<std::int32_t> tokens;  // the prompt's next token, then one per token generated
        double milliseconds{ 0.0 };        // generating, not the prefill
        std::vector<int> accepted;         // drafts accepted per round; empty without the drafter
    };

    /**
     * @brief Greedy decoding as generate() runs it: replayed steps, each choosing its token on the device into the
     *        tensor the next step reads, the host waiting only at the end.
     */
    template<typename TNetwork>
    Generation plainDecode( TNetwork& network, SamplerCuda& sampler, const SamplingParams& sampling,
        typename TNetwork::TokenIndexType& decode_token,
        const std::vector<std::int32_t>& prompt, int tokens )
    {
        IExecutionContext* context = network.getExecutionContext();
        const dim_t start = static_cast<dim_t>( prompt.size() );

        auto history = Measurement::deviceTokens( network, std::vector<std::int32_t>( tokens + 1, 0 ) );
        auto first = tokenSlot( history, 0 );

        sampler.enqueueSampleOnDevice( network.prefill( Measurement::deviceTokens( network, prompt ) ), decode_token, sampling );
        copy( decode_token, first, context );
        network.synchronize();

        Generation result;
        result.milliseconds = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < tokens; ++step )
            {
                sampler.enqueueSampleOnDevice( network.decode( decode_token, start + step ), decode_token, sampling );

                auto slot = tokenSlot( history, step + 1 );
                copy( decode_token, slot, context );
            }
        } );
        result.tokens = readTokens( history, context );

        return result;
    }

    /**
     * @brief Greedy decoding with the drafter (Gemma4Mtp.md 4.2 b): a round at position p drafts K tokens from the
     *        target's state, verifies the token at p and the K drafts in one decodeTokens, and keeps the drafts the
     *        target agrees with plus the target's own token after them.
     *
     * Every token is chosen on the device into the slot the next pass reads, so the host waits once a round, to read
     * the drafts and the target's choices and decide how far to keep.
     */
    template<typename TNetwork, typename TDrafter>
    Generation speculativeDecode( TNetwork& network, TDrafter& drafter, SamplerCuda& sampler,
        const SamplingParams& sampling, const std::vector<std::int32_t>& prompt, int tokens, int draft, dim_t vocabulary, float softcap = 0.0f, std::vector<double>* first_draft_probability = nullptr )
    {
        IExecutionContext* context = network.getExecutionContext();
        const KvCacheView sliding = network.lastSlidingLayerCache();
        const KvCacheView global = network.lastGlobalLayerCache();
        const dim_t rows = draft + 1;

        // The verify's input: the token at p, then the drafts. The target's choice after each of them.
        auto verify = Measurement::deviceTokens( network, std::vector<std::int32_t>( rows, 0 ) );
        auto chosen = Measurement::deviceTokens( network, std::vector<std::int32_t>( rows, 0 ) );
        HostTokens host_verify( Device::Cpu(), verify.shape() );
        HostTokens host_chosen( Device::Cpu(), chosen.shape() );

        auto head = tokenSlot( verify, 0 );
        sampler.enqueueSampleOnDevice( network.prefill( Measurement::deviceTokens( network, prompt ) ), head, sampling );

        Generation result;
        result.tokens.push_back( readTokens( head, context )[ 0 ] );

        dim_t position = static_cast<dim_t>( prompt.size() );

        // The row of the target's final-normed hidden state at p - 1: the prefill's last, then the last token kept's.
        dim_t hidden_row = 0;

        const auto start = std::chrono::steady_clock::now();

        while ( static_cast<int>( result.tokens.size() ) <= tokens )
        {
            context->setDecodePosition( position );
            drafter.startFrom( network.finalNormedRow( hidden_row ) );

            for ( dim_t k = 0; k < draft; ++k )
            {
                auto& embedding = network.embed( tokenSlot( verify, k ) );
                auto& logits = drafter.decode( embedding, position, sliding, global );

                auto next = tokenSlot( verify, k + 1 );
                sampler.enqueueSampleOnDevice( logits, next, kGreedy );
            }

            auto& logits = network.decodeTokens( verify, position );

            for ( dim_t row = 0; row < rows; ++row )
            {
                auto target = tokenSlot( chosen, row );
                sampler.enqueueSampleOnDevice( logits.view( shape_t{ 1, 1, vocabulary }, row * vocabulary ), target, sampling );
            }

            copy( verify, host_verify, context );
            copy( chosen, host_chosen, context );
            context->synchronize();

            // The probability the model's sampler gives the first draft at row 0: its expected chance of being kept.
            if ( first_draft_probability != nullptr )
            {
                const auto row = toHost<TensorDataType::FP32>( logits.view( shape_t{ 1, 1, vocabulary }, 0 ) );
                const std::vector<float> host_row( row.data(), row.data() + row.size() );
                const auto probabilities = samplerDistribution( host_row, sampling, softcap );

                first_draft_probability->push_back( probabilities[ static_cast<std::size_t>( host_verify.data()[ 1 ] ) ] );
            }

            int accepted = 0;

            while ( accepted < draft && host_verify.data()[ accepted + 1 ] == host_chosen.data()[ accepted ] )
                ++accepted;

            for ( int i = 1; i <= accepted; ++i )
                result.tokens.push_back( host_verify.data()[ i ] );

            result.tokens.push_back( host_chosen.data()[ accepted ] );
            result.accepted.push_back( accepted );

            // The caches keep p .. p + accepted; the rejected drafts' rows are written over by the next round.
            position += accepted + 1;
            requireRewind( network, position );

            auto bonus = tokenSlot( chosen, accepted );
            copy( bonus, head, context );
            hidden_row = accepted;
        }

        context->synchronize();
        result.milliseconds = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - start ).count();
        result.tokens.resize( static_cast<std::size_t>( tokens ) + 1 );

        return result;
    }

    struct StepCosts
    {
        double decode_ms{ 0.0 };
        double called_decode_ms{ 0.0 };  // the same step called instead of replayed
        double draft_ms{ 0.0 };
        std::vector<double> verify_ms;  // indexed by rows, 2 to 8
    };

    /// What section 3 prices, at the prompt's depth: a replayed decode, a draft step, and a verify of each row count.
    template<typename TNetwork, typename TDrafter>
    StepCosts measureStepCosts( TNetwork& network, TDrafter& drafter, typename TNetwork::TokenIndexType& decode_token,
        const std::vector<std::int32_t>& prompt, const std::vector<std::int32_t>& continuation, int max_rows )
    {
        IExecutionContext* context = network.getExecutionContext();
        const dim_t start = static_cast<dim_t>( prompt.size() );
        const KvCacheView sliding = network.lastSlidingLayerCache();
        const KvCacheView global = network.lastGlobalLayerCache();

        StepCosts costs;
        network.prefill( Measurement::deviceTokens( network, prompt ) );

        context->setDecodePosition( start );
        auto& embedding = network.embed( decode_token );

        drafter.startFrom( network.finalNormedRow( 0 ) );
        drafter.decode( embedding, start, sliding, global );
        costs.draft_ms = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < 64; ++step )
                drafter.decode( embedding, start, sliding, global );
        } ) / 64;

        decodeMilliseconds( network, decode_token, start, 4 );

        std::vector<double> decode_runs;

        for ( int repeat = 0; repeat < 5; ++repeat )
            decode_runs.push_back( decodeMilliseconds( network, decode_token, start, 64 ) );

        costs.decode_ms = median( decode_runs );

        network.setDecodeReplay( false );
        decode_runs.clear();

        for ( int repeat = 0; repeat < 5; ++repeat )
            decode_runs.push_back( decodeMilliseconds( network, decode_token, start, 64 ) );

        costs.called_decode_ms = median( decode_runs );
        network.setDecodeReplay( true );

        costs.verify_ms.assign( static_cast<std::size_t>( max_rows ) + 1, 0.0 );

        for ( int rows = 2; rows <= max_rows; ++rows )
        {
            const auto input = Measurement::deviceTokens( network,
                std::vector<std::int32_t>( continuation.begin(), continuation.begin() + rows ) );
            std::vector<double> runs;

            for ( int repeat = 0; repeat <= 10; ++repeat )
            {
                requireRewind( network, start );
                const double ms = timedMilliseconds( network, [&] { network.decodeTokens( input, start ); } );

                if ( repeat > 0 )
                    runs.push_back( ms );
            }

            costs.verify_ms[ static_cast<std::size_t>( rows ) ] = median( runs );
        }

        return costs;
    }

    /// The largest difference over the reference's RMS, and how many rows' argmax differ.
    std::pair<double, int> logitDifference( const std::vector<float>& produced, const std::vector<float>& reference, dim_t vocabulary )
    {
        double square_sum = 0.0;
        double largest = 0.0;

        for ( std::size_t i = 0; i < reference.size(); ++i )
        {
            square_sum += static_cast<double>( reference[ i ] ) * reference[ i ];
            largest = std::max( largest, std::abs( static_cast<double>( produced[ i ] ) - reference[ i ] ) );
        }

        int differing = 0;

        for ( std::size_t row = 0; row * vocabulary < reference.size(); ++row )
        {
            const auto begin = static_cast<std::ptrdiff_t>( row * vocabulary );
            const std::vector<float> a( produced.begin() + begin, produced.begin() + begin + vocabulary );
            const std::vector<float> b( reference.begin() + begin, reference.begin() + begin + vocabulary );

            differing += Measurement::argMax( a ) != Measurement::argMax( b );
        }

        return { largest / std::sqrt( square_sum / reference.size() ), differing };
    }

    /**
     * @brief Gemma4Mtp.md 5.2's first gate on the target: a verify's logits against decodes of the same tokens, beside
     *        how far the prefill path -- a third arithmetic -- lies from the same decodes.
     */
    template<typename TNetwork>
    void compareVerifyWithDecode( TNetwork& network, typename TNetwork::TokenIndexType& decode_token,
        const std::vector<std::int32_t>& prompt, const std::vector<std::int32_t>& continuation, int rows, dim_t vocabulary )
    {
        IExecutionContext* context = network.getExecutionContext();
        const dim_t start = static_cast<dim_t>( prompt.size() );
        const std::vector<std::int32_t> run( continuation.begin(), continuation.begin() + rows );

        network.prefill( Measurement::deviceTokens( network, prompt ) );

        std::vector<float> decoded;

        for ( int row = 0; row < rows; ++row )
        {
            setToken( decode_token, run[ row ], context );
            const auto logits = Measurement::hostLogits( network, network.decode( decode_token, start + row ) );
            decoded.insert( decoded.end(), logits.begin(), logits.end() );
        }

        requireRewind( network, start );
        const auto verified = Measurement::hostLogits( network, network.decodeTokens( Measurement::deviceTokens( network, run ), start ) );

        // Each row through the prefill path: the prompt continued through that row, its last row's logits.
        std::vector<float> prefilled;

        for ( int row = 0; row < rows; ++row )
        {
            std::vector<std::int32_t> sequence( prompt );
            sequence.insert( sequence.end(), run.begin(), run.begin() + row + 1 );

            requireRewind( network, start );
            const auto logits = Measurement::hostLogits( network, network.prefillFrom( Measurement::deviceTokens( network, sequence ), start ) );
            prefilled.insert( prefilled.end(), logits.begin(), logits.end() );
        }

        const auto [verify_difference, verify_argmax] = logitDifference( verified, decoded, vocabulary );
        const auto [prefill_difference, prefill_argmax] = logitDifference( prefilled, decoded, vocabulary );

        std::cout << std::format( "  {} tokens against their decodes, largest difference over the logits' RMS: verify {:.2e} "
            "({} argmax differ), prefill path {:.2e} ({} argmax differ)\n",
            rows, verify_difference, verify_argmax, prefill_difference, prefill_argmax );
    }

    /// Where two greedy runs first part, with the plain run's logits there: a near-tie or a structural error.
    template<typename TNetwork>
    std::string describeDivergence( TNetwork& network, typename TNetwork::TokenIndexType& decode_token,
        const std::vector<std::int32_t>& prompt, const Generation& plain, const Generation& drafted, int draft )
    {
        std::size_t index = 0;

        while ( index < plain.tokens.size() && plain.tokens[ index ] == drafted.tokens[ index ] )
            ++index;

        if ( index == plain.tokens.size() )
            return {};

        IExecutionContext* context = network.getExecutionContext();
        const dim_t start = static_cast<dim_t>( prompt.size() );

        network.prefill( Measurement::deviceTokens( network, prompt ) );

        std::vector<float> logits;

        for ( std::size_t i = 0; i < index; ++i )
        {
            setToken( decode_token, plain.tokens[ i ], context );
            logits = Measurement::hostLogits( network, network.decode( decode_token, start + static_cast<dim_t>( i ) ) );
        }

        std::vector<float> sorted = logits;
        std::partial_sort( sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>() );

        return std::format( "K = {}: parted at token {}: plain chose {} (logit {:.4f}), drafted chose {} (logit {:.4f}); "
            "the plain run's top two differ by {:.4f}\n", draft, index, plain.tokens[ index ], logits[ plain.tokens[ index ] ],
            drafted.tokens[ index ], logits[ drafted.tokens[ index ] ], sorted[ 0 ] - sorted[ 1 ] );
    }

    template<typename TNetwork, typename TDrafter>
    void measureSpeculationOn( const Options& options )
    {
        const auto prompts = acceptancePrompts( options.prompt_tokens );
        const int max_draft = *std::max_element( options.drafts.begin(), options.drafts.end() );
        const int max_rows = max_draft + 1;

        dim_t longest = 0;

        for ( const auto& prompt : prompts )
            longest = std::max<dim_t>( longest, static_cast<dim_t>( prompt.tokens.size() ) );

        const dim_t context_length = longest + options.tokens + max_rows + 64;

        Serialization::WeightsReader reader( options.weights );
        GemmaConfig target_config = GemmaCuda::configFromMetadata( reader.getWeightsMetadata() );
        target_config.withDecodeTokens( max_rows );

        auto network = Measurement::buildMeasuredNetwork<TNetwork>( options.weights, target_config, kDevice, context_length );
        network->setDecodeReplay( true );

        Serialization::WeightsReader drafter_reader( options.drafter );
        auto drafter = std::make_shared<TDrafter>( "drafter",
            TDrafter::configFromMetadata( drafter_reader.getWeightsMetadata(), target_config ),
            target_config.getModelDim(), network->getExecutionContext() );
        drafter->build( BuildContext( shape_t{ 1, context_length, drafter->getConfig().getModelDim() }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( kDevice ) ) );
        drafter->loadParameters( drafter_reader );

        const dim_t vocabulary = target_config.getVocabSize();

        SamplerCuda sampler( network->getExecutionContext(), SamplingConfig{}
            .withVocabularySize( vocabulary )
            .withFinalLogitSoftcap( target_config.getFinalLogitSoftcapping() ) );

        // Every decode reads this one tensor, so the recording made of the first stays valid for the run.
        auto decode_token = Measurement::deviceTokens( *network, { 0 } );

        const SamplingParams sampling = samplingOf( options );
        const float softcap = target_config.getFinalLogitSoftcapping();

        std::cout << std::format( "weights {}\ndrafter {}, {} head\ncache {}, {} tokens after the prompt's first, median of {} runs, {}\n",
            options.weights.string(), options.drafter.string(), options.six_bit_drafter_head ? "six-bit" : "BF16",
            options.fp8_cache ? "FP8 global layers" : "BF16",
            options.tokens, options.runs, options.sample
                ? "sampled at " + describeSampling( options ) : std::string( "greedy" ) );

        for ( const auto& prompt : prompts )
        {
            std::vector<Generation> plain_runs;

            for ( int run = 0; run < options.runs; ++run )
                plain_runs.push_back( plainDecode( *network, sampler, sampling, decode_token, prompt.tokens, options.tokens ) );

            std::vector<double> plain_ms;

            for ( const auto& run : plain_runs )
                plain_ms.push_back( run.milliseconds );

            const Generation& plain = plain_runs.back();
            const double plain_per_token = median( plain_ms ) / options.tokens;

            const StepCosts costs = measureStepCosts( *network, *drafter, decode_token, prompt.tokens, plain.tokens, max_rows );

            std::cout << std::format( "\n## {} ({} prompt tokens)\n\nplain {:.3f} ms/token; at the prompt's depth a "
                "replayed decode {:.3f} ms ({:.3f} called), a draft step {:.3f} ms ({:.3f} of a decode), a verify of",
                prompt.name, prompt.tokens.size(), plain_per_token, costs.decode_ms, costs.called_decode_ms, costs.draft_ms,
                costs.draft_ms / costs.decode_ms );

            for ( int rows = 2; rows <= max_rows; ++rows )
                std::cout << std::format( " {} rows {:.3f} ms ({:.2f}){}", rows, costs.verify_ms[ rows ],
                    costs.verify_ms[ rows ] / costs.decode_ms, rows == max_rows ? "\n" : "," );

            for ( std::size_t i = 1; i < plain.tokens.size(); ++i )
            {
                if ( isStopToken( plain.tokens[ i ] ) )
                {
                    std::cout << std::format( "  NOTE: the plain reply ends at token {}; tokens past it continue a finished reply\n", i );
                    break;
                }
            }

            compareVerifyWithDecode( *network, decode_token, prompt.tokens, plain.tokens, max_rows, vocabulary );

            std::vector<std::string> divergences;

            std::cout << std::format( "\n| K | rounds | E[accepted] | ms/token | speedup | predicted | round ms, measured / "
                "priced | {} |\n|---|---|---|---|---|---|---|---|\n",
                options.sample ? "draft 1 kept, measured / expected" : "same tokens as plain" );

            for ( const int draft : options.drafts )
            {
                std::vector<Generation> runs;
                std::vector<double> ms;

                for ( int run = 0; run < options.runs; ++run )
                {
                    runs.push_back( speculativeDecode( *network, *drafter, sampler, sampling, prompt.tokens, options.tokens,
                        draft, vocabulary ) );
                    ms.push_back( runs.back().milliseconds );
                }

                const Generation& drafted = runs.back();
                const double per_token = median( ms ) / options.tokens;
                const double rounds = static_cast<double>( drafted.accepted.size() );

                double accepted = 0.0;

                for ( const int value : drafted.accepted )
                    accepted += value;

                accepted /= rounds;

                // Section 3, with this run's acceptance and the costs measured above.
                const double priced_round = draft * costs.draft_ms + costs.verify_ms[ draft + 1 ];
                const double predicted = ( accepted + 1.0 ) * costs.decode_ms / priced_round;

                std::string last_column;

                if ( options.sample )
                {
                    // A further, untimed run: each round's first draft is kept with probability p(draft) under the
                    // model's own sampler, so the kept fraction must match the mean of p within its standard error.
                    std::vector<double> probability;
                    const Generation checked = speculativeDecode( *network, *drafter, sampler, sampling, prompt.tokens,
                        options.tokens, draft, vocabulary, softcap, &probability );

                    double kept = 0.0;
                    double expected = 0.0;
                    double variance = 0.0;

                    for ( std::size_t round = 0; round < probability.size(); ++round )
                    {
                        kept += checked.accepted[ round ] >= 1 ? 1.0 : 0.0;
                        expected += probability[ round ];
                        variance += probability[ round ] * ( 1.0 - probability[ round ] );
                    }

                    const double n = static_cast<double>( probability.size() );
                    last_column = std::format( "{:.3f} / {:.3f} +- {:.3f}", kept / n, expected / n, std::sqrt( variance ) / n );
                }
                else
                {
                    std::size_t same = 0;

                    while ( same < plain.tokens.size() && plain.tokens[ same ] == drafted.tokens[ same ] )
                        ++same;

                    last_column = same == plain.tokens.size() ? std::string( "all" ) : std::format( "first {}", same );

                    if ( same < plain.tokens.size() )
                        divergences.push_back( describeDivergence( *network, decode_token, prompt.tokens, plain, drafted, draft ) );
                }

                std::cout << std::format( "| {} | {} | {:.2f} | {:.3f} | {:.2f}x | {:.2f}x | {:.2f} / {:.2f} | {} |\n",
                    draft, drafted.accepted.size(), accepted, per_token, plain_per_token / per_token, predicted,
                    median( ms ) / rounds, priced_round, last_column );
            }

            for ( const auto& divergence : divergences )
                std::cout << divergence;
        }
    }

    /**
     * @brief How many distinct experts R consecutive rows of the 26B-A4B route to, per layer: the union a verify of R
     *        rows reads (Gemma4Mtp.md 4.7, step 5).
     *
     * Each prompt's greedy reply is prefilled after it with the router projections observed; a row's experts are the
     * top_k of its logits. A window starts at every reply position. A verify's rows past a rejected draft are other
     * tokens than these, so the windows stand for rounds whose drafts were kept. With --output, every reply row's
     * experts are written as INT32 [rows, layers, top_k] to <output>/<prompt>.routing.
     */
    template<typename TNetwork>
    void measureRoutingOn( const Options& options )
    {
        const bool write_routing = options.output_given;

        using DeviceLogits = Tensor<TensorDataType::BF16, typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource>;

        const auto prompts = acceptancePrompts( options.prompt_tokens );

        dim_t longest = 0;

        for ( const auto& prompt : prompts )
            longest = std::max<dim_t>( longest, static_cast<dim_t>( prompt.tokens.size() ) );

        const dim_t context_length = longest + options.tokens + 64;

        Serialization::WeightsReader reader( options.weights );
        const GemmaConfig config = GemmaCuda::configFromMetadata( reader.getWeightsMetadata() );
        const int experts = static_cast<int>( config.getNumExperts() );
        const int top_k = static_cast<int>( config.getTopKExperts() );

        if ( experts == 0 )
            throw std::invalid_argument( "routing measures a mixture-of-experts target: pass --target 26b" );

        auto network = Measurement::buildMeasuredNetwork<TNetwork>( options.weights, config, kDevice, context_length );
        network->setDecodeReplay( true );

        IExecutionContext* context = network->getExecutionContext();

        SamplerCuda sampler( context, SamplingConfig{}
            .withVocabularySize( config.getVocabSize() )
            .withFinalLogitSoftcap( config.getFinalLogitSoftcapping() ) );

        auto decode_token = Measurement::deviceTokens( *network, { 0 } );

        if ( write_routing )
            fs::create_directories( options.output );

        std::cout << std::format( "weights {}\n{} experts, top {}; greedy replies of up to {} tokens, routing from their "
            "prefill\n", options.weights.string(), experts, top_k, options.tokens );

        for ( const auto& prompt : prompts )
        {
            const Generation plain = plainDecode( *network, sampler, kGreedy, decode_token, prompt.tokens, options.tokens );

            std::size_t reply_rows = plain.tokens.size();

            for ( std::size_t i = 0; i < plain.tokens.size(); ++i )
            {
                if ( isStopToken( plain.tokens[ i ] ) )
                {
                    reply_rows = i;
                    break;
                }
            }

            std::vector<std::int32_t> sequence = prompt.tokens;
            sequence.insert( sequence.end(), plain.tokens.begin(), plain.tokens.begin() + reply_rows );

            // Layer paths in the order the prefill reaches them; each layer's logits row-major over the sequence.
            std::vector<std::string> layers;
            std::vector<std::vector<float>> logits;

            const std::size_t observed = network->observe( "*.router.proj", ComputePassMask{ ComputePass::Forward },
                [&]( std::string_view path, ComputePass, std::string_view stage, const ITensor& value )
                {
                    if ( stage != "output" )
                        return;

                    const auto* typed = dynamic_cast<const DeviceLogits*>( &value );

                    if ( typed == nullptr )
                        throw std::runtime_error( std::format( "{}: router logits are not BF16 on the device", path ) );

                    const auto host = toHost<TensorDataType::FP32>( *typed, context );
                    context->synchronize();

                    auto layer = std::find( layers.begin(), layers.end(), path );

                    if ( layer == layers.end() )
                    {
                        layers.emplace_back( path );
                        logits.emplace_back();
                        layer = layers.end() - 1;
                    }

                    auto& rows = logits[ static_cast<std::size_t>( layer - layers.begin() ) ];
                    rows.insert( rows.end(), host.data(), host.data() + host.size() );
                } );

            if ( observed == 0 )
                throw std::runtime_error( "no component matched *.router.proj" );

            ( void )network->prefill( Measurement::deviceTokens( *network, sequence ) );
            network->synchronize();
            network->stopObserving();

            const std::size_t first = prompt.tokens.size();
            const std::size_t layer_count = layers.size();

            for ( const auto& rows : logits )
            {
                if ( rows.size() != sequence.size() * static_cast<std::size_t>( experts ) )
                    throw std::runtime_error( std::format( "a router published {} logits for {} rows of {} experts",
                        rows.size(), sequence.size(), experts ) );
            }

            // routing[ ( row * layers + layer ) * top_k + slot ], reply rows only, highest logit first.
            std::vector<std::int32_t> routing( reply_rows * layer_count * top_k );
            std::vector<int> order( experts );

            for ( std::size_t row = 0; row < reply_rows; ++row )
            {
                for ( std::size_t layer = 0; layer < layer_count; ++layer )
                {
                    const float* row_logits = logits[ layer ].data() + ( first + row ) * experts;

                    for ( int e = 0; e < experts; ++e )
                        order[ e ] = e;

                    std::partial_sort( order.begin(), order.begin() + top_k, order.end(), [&]( int a, int b )
                    {
                        return row_logits[ a ] > row_logits[ b ] || ( row_logits[ a ] == row_logits[ b ] && a < b );
                    } );

                    std::copy( order.begin(), order.begin() + top_k, routing.begin() + ( row * layer_count + layer ) * top_k );
                }
            }

            std::cout << std::format( "\n## {} ({} reply rows, {} layers)\n\n| R | union, mean | per layer, min / median / "
                "max of window means | all distinct | uniform routing | expert bytes vs R rows apart | vs one row |\n"
                "|---|---|---|---|---|---|---|\n", prompt.name, reply_rows, layer_count );

            for ( int rows = 1; rows <= 8; ++rows )
            {
                if ( reply_rows < static_cast<std::size_t>( rows ) )
                    break;

                const std::size_t windows = reply_rows - rows + 1;
                std::vector<double> layer_means( layer_count, 0.0 );
                double total = 0.0;

                for ( std::size_t start = 0; start < windows; ++start )
                {
                    for ( std::size_t layer = 0; layer < layer_count; ++layer )
                    {
                        std::vector<bool> seen( experts, false );
                        int distinct = 0;

                        for ( int r = 0; r < rows; ++r )
                        {
                            const std::int32_t* chosen = routing.data() + ( ( start + r ) * layer_count + layer ) * top_k;

                            for ( int slot = 0; slot < top_k; ++slot )
                            {
                                if ( !seen[ chosen[ slot ] ] )
                                {
                                    seen[ chosen[ slot ] ] = true;
                                    ++distinct;
                                }
                            }
                        }

                        layer_means[ layer ] += distinct;
                        total += distinct;
                    }
                }

                for ( double& mean : layer_means )
                    mean /= static_cast<double>( windows );

                std::sort( layer_means.begin(), layer_means.end() );

                const double mean = total / static_cast<double>( windows * layer_count );
                const double uniform = experts * ( 1.0 - std::pow( 1.0 - static_cast<double>( top_k ) / experts, rows ) );

                std::cout << std::format( "| {} | {:.2f} | {:.2f} / {:.2f} / {:.2f} | {} | {:.2f} | {:.2f} | {:.2f}x |\n",
                    rows, mean, layer_means.front(), layer_means[ layer_count / 2 ], layer_means.back(), rows * top_k,
                    uniform, mean / ( rows * top_k ), mean / top_k );
            }

            if ( write_routing )
                writeFile( options.output / ( prompt.name + ".routing" ), routing.data(), routing.size() * sizeof( std::int32_t ) );
        }
    }

    struct LibraryRun
    {
        std::vector<std::int32_t> tokens;
        double ms_per_token{ 0.0 };  // from the first token's callback to the last, so the prefill is not in it
        std::uint64_t tokens_hash{ 14695981039346656037ull };  // FNV-1a over every run's tokens, in run order
    };

    /// Generation through GemmaModel::generate, as a program runs it: median of `runs`.
    LibraryRun generateThroughTheModel( GemmaCuda& model, const SamplingParams& sampling, const std::vector<std::int32_t>& prompt,
        int tokens, int runs, std::optional<std::uint64_t> seed )
    {
        GenerateParams params;
        params.max_new_tokens = tokens + 1;
        params.sampling = sampling;

        LibraryRun result;
        std::vector<double> per_token;

        for ( int run = 0; run < runs; ++run )
        {
            if ( seed )
                model.seedSampler( *seed + static_cast<std::uint64_t>( run ) );

            std::vector<std::int32_t> generated;
            std::chrono::steady_clock::time_point first{};
            std::chrono::steady_clock::time_point last{};

            ( void )model.generate( prompt, [&]( std::int32_t token )
            {
                last = std::chrono::steady_clock::now();

                if ( generated.empty() )
                    first = last;

                generated.push_back( token );
            }, params );

            if ( generated.size() < 2 )
                throw std::runtime_error( "the reply ended before a second token" );

            for ( const std::int32_t token : generated )
            {
                result.tokens_hash ^= static_cast<std::uint32_t>( token );
                result.tokens_hash *= 1099511628211ull;
            }

            per_token.push_back( std::chrono::duration<double, std::milli>( last - first ).count() / ( generated.size() - 1 ) );
            result.tokens = std::move( generated );
        }

        result.ms_per_token = median( per_token );

        return result;
    }

    /// Gemma4Mtp.md 4.7, 6b-1's gate: the library's loop at the rate the tool's loop measured, with plain greedy's tokens.
    void measureGenerateThroughTheModel( const Options& options )
    {
        const auto prompts = acceptancePrompts( options.prompt_tokens );

        dim_t longest = 0;

        for ( const auto& prompt : prompts )
            longest = std::max<dim_t>( longest, static_cast<dim_t>( prompt.tokens.size() ) );

        const int max_draft = *std::max_element( options.drafts.begin(), options.drafts.end() );
        const dim_t context_length = ( ( longest + options.tokens + max_draft + 64 ) / 1024 + 1 ) * 1024;

        DeploymentRequest request;
        request.withWeightQuantization( WeightQuantization::Q4_0 )
            .withKvCacheCompression( options.fp8_cache ? KvCacheCompression::FP8 : KvCacheCompression::None )
            .withContextLength( context_length );

        // Sampled replies part from the first token whatever the draft does, so only greedy ones are compared.
        std::cout << std::format( "weights {}\ndrafter {}\ncontext {}, {} tokens after the first, median of {} runs, {}, "
            "through GemmaModel::generate{}\n\n| prompt | K | ms/token | speedup | same tokens as plain | tokens hash |\n"
            "|---|---|---|---|---|---|\n",
            options.weights.string(), options.drafter.string(), context_length, options.tokens, options.runs,
            options.sample ? "sampled at " + describeSampling( options ) : std::string( "greedy" ),
            options.seed ? std::format( ", run r seeded {} + r", *options.seed ) : std::string() );

        const auto hashOf = [&]( const LibraryRun& run )
        {
            return options.seed ? std::format( "{:016x}", run.tokens_hash ) : std::string( "-" );
        };

        std::vector<LibraryRun> plain;

        {
            auto model = GemmaCuda::load( options.weights, request );

            for ( const auto& prompt : prompts )
            {
                plain.push_back( generateThroughTheModel(
                    *model, samplingOf( options ), prompt.tokens, options.tokens, options.runs, options.seed ) );
                std::cout << std::format( "| {} | - | {:.3f} | 1.00x | - | {} |\n",
                    prompt.name, plain.back().ms_per_token, hashOf( plain.back() ) );
            }
        }

        for ( const int draft : options.drafts )
        {
            auto model = GemmaCuda::load( options.weights,
                DeploymentRequest( request ).withSpeculativeDecode( options.drafter, draft ) );

            for ( std::size_t i = 0; i < prompts.size(); ++i )
            {
                const LibraryRun drafted = generateThroughTheModel(
                    *model, samplingOf( options ), prompts[ i ].tokens, options.tokens, options.runs, options.seed );

                std::size_t same = 0;

                while ( same < plain[ i ].tokens.size() && same < drafted.tokens.size()
                    && plain[ i ].tokens[ same ] == drafted.tokens[ same ] )
                {
                    ++same;
                }

                std::cout << std::format( "| {} | {} | {:.3f} | {:.2f}x | {} | {} |\n", prompts[ i ].name, draft,
                    drafted.ms_per_token, plain[ i ].ms_per_token / drafted.ms_per_token,
                    options.sample ? std::string( "-" )
                        : same == plain[ i ].tokens.size() ? std::string( "all" ) : std::format( "first {}", same ),
                    hashOf( drafted ) );
            }
        }
    }

    export int run( int argc, char** argv )
    {
        try
        {
            const std::string_view command = argc > 1 ? argv[ 1 ] : "";

            if ( command != "verify-cost" && command != "drafter-parity" && command != "acceptance" && command != "speculate"
                && command != "generate" && command != "routing" )
            {
                printUsage();

                return command.empty() || command == "--help" ? 0 : 1;
            }

            const Options options = parseOptions( argc, argv );

            // Warnings shown: a decode step that stops replaying says so here, and its timing changes with it.
            Mila::initialize( 0, std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning ) );

            if ( command == "generate" )
            {
                measureGenerateThroughTheModel( options );

                return 0;
            }

            const auto measureWith = [&]<typename TNetwork, typename TDrafter>()
            {
                if ( command == "verify-cost" )
                    measureVerifyCost<TNetwork>( options );
                else if ( command == "drafter-parity" )
                    dumpDrafterParity<TNetwork>( options );
                else if ( command == "speculate" )
                    measureSpeculationOn<TNetwork, TDrafter>( options );
                else if ( command == "routing" )
                    measureRoutingOn<TNetwork>( options );
                else
                    measureAcceptanceOn<TNetwork, TDrafter>( options );
            };

            const auto measure = [&]<typename TNetwork>()
            {
                if ( options.six_bit_drafter_head )
                    measureWith.template operator()<TNetwork, GemmaDrafterSixBitHeadCuda>();
                else
                    measureWith.template operator()<TNetwork, GemmaDrafterCuda>();
            };

            if ( options.target_26b && options.fp8_cache )
                measure.template operator()<GemmaRoutedQ4_0Fp8GlobalNetwork>();
            else if ( options.target_26b )
                measure.template operator()<GemmaRoutedQ4_0Network>();
            else if ( options.fp8_cache )
                measure.template operator()<GemmaQ4_0Fp8GlobalNetwork>();
            else
                measure.template operator()<GemmaQ4_0Network>();

            return 0;
        }
        catch ( const std::exception& e )
        {
            std::cerr << "Drafting: " << e.what() << "\n";

            return 1;
        }
    }
}
