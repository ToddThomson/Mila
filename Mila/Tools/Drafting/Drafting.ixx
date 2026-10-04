/**
 * @file Drafting.ixx
 * @brief Measurements that decide whether a draft model pays on a target (Gemma4Mtp.md section 5.1).
 *
 * `verify-cost`: what checking K drafted tokens costs the target, against one decode, at a depth.
 * `drafter-parity`: the drafter's steps from the target's state, with every input dumped for HuggingFace's forward.
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
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

export module Tools.Drafting;

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Measurement/Pg19Books.h"

namespace Mila::Tools::Drafting
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

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
        float temperature{ 0.8f };
        bool fp8_cache{ false };
        fs::path weights{ fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma4_12b_it_qat_q4_0.safetensors" };
        fs::path drafter{ fs::path( MILA_DATA_DIR ) / "Models" / "Gemma" / "gemma4_12b_it_qat_drafter_bf16.bin" };
        fs::path output{ "drafter_parity" };
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
            << "  --temperature   For the first draft's sampled acceptance, sum min(p, q). Default: 0.8.\n"
            << "  --kv-cache      bf16 | fp8 (the target's global layers). Default: bf16.\n"
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
            else if ( arg == "--target" )
            {
                const std::string_view target = value();

                if ( target != "12b" && target != "26b" )
                    throw std::invalid_argument( std::format( "--target expects 12b or 26b, got '{}'", target ) );

                options.target_26b = target == "26b";
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

        for ( int step = 0; step < options.max_draft; ++step )
        {
            const auto token_tensor = Measurement::deviceTokens( *network, { token } );
            auto& embedding = network->embed( token_tensor );

            writeTensor( options.output / std::format( "embedding_{}.f32", step ), embedding, context );
            writeTensor( options.output / std::format( "hidden_{}.f32", step ), *hidden, context );

            auto& logits = drafter->decode( embedding, *hidden, prompt_length, sliding, global );

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

    struct AcceptanceResult
    {
        std::vector<int> accepted;          // per round, at the longest K
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
    template<typename TNetwork>
    AcceptanceResult measureAcceptance( TNetwork& network, GemmaDrafterCuda& drafter, const std::vector<std::int32_t>& prompt,
        int rounds, int max_draft, float temperature, float softcap )
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
                const auto* hidden = &network.finalNormedHidden();
                std::vector<std::int32_t> chain;

                for ( int step = 0; step < max_draft; ++step )
                {
                    auto& embedding = network.embed( Measurement::deviceTokens( network, { token } ) );
                    std::vector<float> logits = Measurement::hostLogits( network,
                        drafter.decode( embedding, *hidden, position, sliding, global ) );

                    token = Measurement::argMax( logits );
                    chain.push_back( token );
                    hidden = &drafter.nextHidden();

                    if ( step == 0 )
                        first_draft_logits = std::move( logits );
                }

                drafts.push_back( std::move( chain ) );
            }

            const std::vector<float> target_logits = Measurement::hostLogits( network,
                network.decode( Measurement::deviceTokens( network, { target.back() } ), position ) );

            if ( round < rounds )
            {
                const auto p = distribution( target_logits, temperature, softcap );
                const auto q = distribution( first_draft_logits, temperature, 0.0f );
                double overlap = 0.0;

                for ( std::size_t i = 0; i < p.size(); ++i )
                    overlap += std::min( p[ i ], q[ i ] );

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

        result.first_sampled.resize( static_cast<std::size_t>( rounds ) );
        result.continuation.assign( target.begin(), target.begin() + rounds );

        return result;
    }

    /// Per-step time of `steps` draft steps on a fixed input, and of `steps` target decodes, after `prompt`.
    template<typename TNetwork>
    std::pair<double, double> stepTimes( TNetwork& network, GemmaDrafterCuda& drafter, const std::vector<std::int32_t>& prompt,
        int steps )
    {
        const KvCacheView sliding = network.lastSlidingLayerCache();
        const KvCacheView global = network.lastGlobalLayerCache();
        const auto token = Measurement::deviceTokens( network, { 0 } );
        const dim_t position = static_cast<dim_t>( prompt.size() );

        network.prefill( Measurement::deviceTokens( network, prompt ) );

        network.getExecutionContext()->setDecodePosition( position );
        auto& embedding = network.embed( token );
        const auto& hidden = network.finalNormedHidden();

        drafter.decode( embedding, hidden, position, sliding, global );

        const double draft_ms = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < steps; ++step )
                drafter.decode( embedding, hidden, position, sliding, global );
        } ) / steps;

        requireRewind( network, position );

        const double decode_ms = timedMilliseconds( network, [&]
        {
            for ( int step = 0; step < steps; ++step )
                network.decode( token, position + step );
        } ) / steps;

        return { draft_ms, decode_ms };
    }

    template<typename TNetwork>
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
        auto drafter = std::make_shared<GemmaDrafterCuda>( "drafter",
            GemmaDrafterCuda::configFromMetadata( drafter_reader.getWeightsMetadata(), target_config ),
            target_config.getModelDim(), network->getExecutionContext() );
        drafter->build( BuildContext( shape_t{ 1, context_length, drafter->getConfig().getModelDim() }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( kDevice ) ) );
        drafter->loadParameters( drafter_reader );

        const float softcap = target_config.getFinalLogitSoftcapping();

        std::cout << std::format( "weights {}\ndrafter {}\ncache {}, {} rounds per prompt, temperature {} for the first "
            "draft's sampled acceptance\n\n", options.weights.string(), options.drafter.string(),
            options.fp8_cache ? "FP8 global layers" : "BF16", options.positions, options.temperature );

        std::cout << "| prompt | tokens | rounds | E[accepted], K = 1..8 | first draft, sampled |\n|---|---|---|---|---|\n";

        std::vector<std::pair<std::string, std::vector<std::int32_t>>> continuations;

        for ( const auto& prompt : prompts )
        {
            const AcceptanceResult result = measureAcceptance( *network, *drafter, prompt.tokens,
                options.positions, options.max_draft, options.temperature, softcap );

            std::string by_draft;

            for ( int k = 1; k <= options.max_draft; ++k )
            {
                double total = 0.0;

                for ( int accepted : result.accepted )
                    total += std::min( accepted, k );

                by_draft += std::format( "{}{:.2f}", k == 1 ? "" : " / ", total / result.accepted.size() );
            }

            double sampled = 0.0;

            for ( double value : result.first_sampled )
                sampled += value;

            std::cout << std::format( "| {} | {} | {} | {} | {:.3f} |\n", prompt.name, prompt.tokens.size(),
                result.accepted.size(), by_draft, sampled / result.first_sampled.size() );

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

    export int run( int argc, char** argv )
    {
        try
        {
            const std::string_view command = argc > 1 ? argv[ 1 ] : "";

            if ( command != "verify-cost" && command != "drafter-parity" && command != "acceptance" )
            {
                printUsage();

                return command.empty() || command == "--help" ? 0 : 1;
            }

            const Options options = parseOptions( argc, argv );

            // Warnings shown: a decode step that stops replaying says so here, and its timing changes with it.
            Mila::initialize( 0, std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning ) );

            const auto measure = [&]<typename TNetwork>()
            {
                if ( command == "verify-cost" )
                    measureVerifyCost<TNetwork>( options );
                else if ( command == "drafter-parity" )
                    dumpDrafterParity<TNetwork>( options );
                else
                    measureAcceptanceOn<TNetwork>( options );
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
