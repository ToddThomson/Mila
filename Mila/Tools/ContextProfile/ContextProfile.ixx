/**
 * @file ContextProfile.ixx
 * @brief What a model configuration is worth to an agent at each context length it can hold (ContextProfile.md).
 *
 * Phase 1: fit, loss by band and recall at depth, one JSON file and its Markdown rendering per run.
 */

module;

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

export module Tools.ContextProfile;

import Mila;
import nlohmann.json;

#include "Measurement/LogLikelihoodHarness.h"
#include "Measurement/Pg19Books.h"

namespace Mila::Tools::ContextProfile
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace fs = std::filesystem;
    namespace GemmaProtocol = ::Mila::Dnn::Gemma;
    namespace QwenProtocol = ::Mila::Dnn::Qwen;
    namespace Turns = ::Mila::Dnn::Conversation;

    using Json = nlohmann::ordered_json;

    // ====================================================================
    // Configurations
    // ====================================================================

    enum class Family { Gemma, Llama, Qwen };

    /// A model, its weight format and its KV-cache format; the card is whichever runs the tool.
    struct Configuration
    {
        std::string_view name;
        Family family;

        /// Under the models directory.
        std::string_view weights;

        std::string_view weight_format;
        std::string_view kv_cache;
        WeightQuantization weight_quantization;
        KvCacheCompression kv_cache_compression;
    };

    // The 26B-A4B and the Qwen FP4 build quantize on load from their full-precision weights, as the pricing did.
    constexpr Configuration kConfigurations[] = {
        { "gemma-4-12b-q4_0", Family::Gemma, "Gemma/gemma4_12b_it_qat_q4_0.safetensors", "Q4_0", "BF16",
            WeightQuantization::Q4_0, KvCacheCompression::None },
        { "gemma-4-12b-fp4", Family::Gemma, "Gemma/gemma4_12b_it_fp4.safetensors", "FP4", "BF16",
            WeightQuantization::FP4, KvCacheCompression::None },
        { "gemma-4-26b-a4b-q4_0", Family::Gemma, "Gemma/gemma4_26b_a4b_it_qat_bf16.bin", "Q4_0", "FP8 global layers",
            WeightQuantization::Q4_0, KvCacheCompression::FP8 },
        { "qwen-3.8-27b-fp4", Family::Qwen, "Qwen/qwen38_27b_fp4.safetensors", "FP4", "BF16",
            WeightQuantization::FP4, KvCacheCompression::None },
        { "qwen-3.8-27b-2.82-bit", Family::Qwen, "Qwen/qwen38_27b_cb2-3.safetensors", "2.82-bit codebook", "BF16",
            WeightQuantization::Plan, KvCacheCompression::None },
        { "llama-3.2-3b-fp4", Family::Llama, "LLaMa/llama32_3b_instruct_fp4.safetensors", "FP4", "BF16",
            WeightQuantization::FP4, KvCacheCompression::None },
        { "llama-3.1-8b-fp4", Family::Llama, "LLaMa/llama31_8b_instruct_fp4.safetensors", "FP4", "BF16",
            WeightQuantization::FP4, KvCacheCompression::None },
    };

    using GemmaCuda = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using LlamaCuda = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using QwenCuda = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

    // What each model's own dispatch builds for the configuration above; PlanEqualsBuild holds the planner to the
    // same types.
    using GemmaQ4_0Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy>;
    using GemmaFp4Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupFp4<128>, GemmaCuda::GemmaSlidingKvPolicy>;
    using GemmaRoutedQ4_0Fp8GlobalNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupInt4<32>, GemmaCuda::GemmaSlidingKvPolicy, GemmaFeedForward::Routed,
        Quant::KvCache::PerTokenKvFp8<>>;
    using QwenFp4Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
        QwenOraclePrecisionPlan, QwenCuda::QwenKvPolicy>;
    using QwenCodebookNetwork = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
        QwenPrecisionPlan, QwenCuda::QwenKvPolicy>;
    using LlamaFp4Network = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
        Quant::Weight::PerGroupFp4<128>, Quant::KvCache::NoKvCompression>;

    /// Runs `action.template operator()<TNetwork, TModel>()` for the named configuration.
    template<typename TAction>
    void dispatchConfiguration( std::string_view name, TAction&& action )
    {
        if ( name == "gemma-4-12b-q4_0" )
            return action.template operator()<GemmaQ4_0Network, GemmaCuda>();

        if ( name == "gemma-4-12b-fp4" )
            return action.template operator()<GemmaFp4Network, GemmaCuda>();

        if ( name == "gemma-4-26b-a4b-q4_0" )
            return action.template operator()<GemmaRoutedQ4_0Fp8GlobalNetwork, GemmaCuda>();

        if ( name == "qwen-3.8-27b-fp4" )
            return action.template operator()<QwenFp4Network, QwenCuda>();

        if ( name == "qwen-3.8-27b-2.82-bit" )
            return action.template operator()<QwenCodebookNetwork, QwenCuda>();

        if ( name == "llama-3.2-3b-fp4" || name == "llama-3.1-8b-fp4" )
            return action.template operator()<LlamaFp4Network, LlamaCuda>();

        throw std::invalid_argument( std::format( "no network type for configuration '{}'", name ) );
    }

    const Configuration& configurationNamed( std::string_view name )
    {
        for ( const Configuration& configuration : kConfigurations )
        {
            if ( configuration.name == name )
                return configuration;
        }

        throw std::invalid_argument( std::format( "unknown configuration '{}'; `ContextProfile list` names them", name ) );
    }

    std::string_view familyName( Family family )
    {
        switch ( family )
        {
            case Family::Gemma: return "gemma";
            case Family::Llama: return "llama";
            case Family::Qwen: return "qwen";
        }

        return "unknown";
    }

    // ====================================================================
    // Options
    // ====================================================================

    /// The bands of section 3, before the configuration's largest planned context is added.
    constexpr dim_t kStandardBands[] = { 16384, 32768, 65536, 131072 };

    struct Options
    {
        std::string configuration;

        /// Empty: the standard bands and the configuration's largest planned context.
        std::vector<dim_t> bands;

        bool fit{ true };
        bool loss{ true };
        bool recall{ true };

        /// Loss: PG-19 books that fill the largest band. Five, as G2's decision 6 runs.
        std::size_t books{ 5 };

        /// Recall: conversations a band. Twelve, eight records at five depths: 96 trials a cell (section 9).
        std::size_t conversations{ 12 };

        /// Loss: a fixed prefill chunk instead of the planner's, so two runs differ in nothing else.
        std::optional<dim_t> loss_chunk;

        /// Measure bands the planner refuses for memory too: scores do not depend on the fit, rates would.
        bool past_free_memory{ false };

        /// Recall: let the model think before it answers, as long as it chooses (section 4.3).
        bool thinking{ false };

        fs::path data_directory{ MILA_DATA_DIR };

        /// Empty: the configuration's own, under the data directory.
        fs::path weights;

        /// Empty: <configuration>.json in the working directory. The Markdown rendering is written beside it.
        fs::path output;
    };

    void printUsage()
    {
        std::cerr
            << "Usage:\n"
            << "  ContextProfile list\n"
            << "  ContextProfile run <configuration> [options]\n"
            << "\n"
            << "Options for run:\n"
            << "  --bands 16384,32768,...  Context lengths to profile. Default: 16K, 32K, 64K, 128K and the largest\n"
            << "                           context the planner chooses, every one the card fits.\n"
            << "  --arms fit,loss,recall   Which arms run. Default: all three.\n"
            << "  --books N                Loss: PG-19 books filling the largest band. Default: 5.\n"
            << "  --conversations N        Recall: conversations a band. Default: 12.\n"
            << "  --loss-chunk N           Loss: this prefill chunk instead of the planner's.\n"
            << "  --past-free-memory       Also measure bands refused for memory. On WDDM they spill to host\n"
            << "                           memory, so their scores stand and their timings do not.\n"
            << "  --thinking               Recall: the model thinks before answering, as long as it chooses; the\n"
            << "                           answer log-likelihood is not scored, the thinking tokens are counted.\n"
            << "  --weights PATH           Weights file. Default: the configuration's, under the data directory.\n"
            << "  --data PATH              Data directory (models, PG-19). Default: the build's.\n"
            << "  --output PATH            JSON output; the Markdown goes beside it. Default: <configuration>.json.\n";
    }

    std::vector<std::string_view> splitList( std::string_view text )
    {
        std::vector<std::string_view> parts;

        while ( !text.empty() )
        {
            const std::size_t comma = text.find( ',' );
            parts.push_back( text.substr( 0, comma ) );
            text = comma == std::string_view::npos ? std::string_view{} : text.substr( comma + 1 );
        }

        return parts;
    }

    std::size_t parseCount( std::string_view text, std::string_view flag )
    {
        std::size_t value = 0;
        const auto parsed = std::from_chars( text.data(), text.data() + text.size(), value );

        if ( parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0 )
            throw std::invalid_argument( std::format( "{} expects a positive integer, got '{}'", flag, text ) );

        return value;
    }

    Options parseRunOptions( int argc, char** argv )
    {
        if ( argc < 3 )
            throw std::invalid_argument( "run needs a configuration" );

        Options options;
        options.configuration = argv[ 2 ];

        for ( int index = 3; index < argc; ++index )
        {
            const std::string_view flag = argv[ index ];

            if ( flag == "--past-free-memory" )
            {
                options.past_free_memory = true;
                continue;
            }

            if ( flag == "--thinking" )
            {
                options.thinking = true;
                continue;
            }

            if ( index + 1 >= argc )
                throw std::invalid_argument( std::format( "{} needs a value", flag ) );

            const std::string_view value = argv[ ++index ];

            if ( flag == "--bands" )
            {
                for ( const std::string_view band : splitList( value ) )
                    options.bands.push_back( static_cast<dim_t>( parseCount( band, flag ) ) );

                std::ranges::sort( options.bands );
            }
            else if ( flag == "--arms" )
            {
                options.fit = options.loss = options.recall = false;

                for ( const std::string_view arm : splitList( value ) )
                {
                    if ( arm == "fit" )
                        options.fit = true;
                    else if ( arm == "loss" )
                        options.loss = true;
                    else if ( arm == "recall" )
                        options.recall = true;
                    else
                        throw std::invalid_argument( std::format( "unknown arm '{}'", arm ) );
                }
            }
            else if ( flag == "--books" )
                options.books = parseCount( value, flag );
            else if ( flag == "--conversations" )
                options.conversations = parseCount( value, flag );
            else if ( flag == "--loss-chunk" )
                options.loss_chunk = static_cast<dim_t>( parseCount( value, flag ) );
            else if ( flag == "--weights" )
                options.weights = value;
            else if ( flag == "--data" )
                options.data_directory = value;
            else if ( flag == "--output" )
                options.output = value;
            else
                throw std::invalid_argument( std::format( "unknown option '{}'", flag ) );
        }

        return options;
    }

    // ====================================================================
    // Family: tokenizer, stop tokens, the book turn
    // ====================================================================

    const DeviceId kDevice{ DeviceType::Cuda, 0 };

    std::shared_ptr<Mila::Data::BpeTokenizer> loadTokenizer( Family family, const fs::path& models )
    {
        switch ( family )
        {
            case Family::Gemma: return Mila::Data::BpeTokenizer::loadGemma( models / "Gemma" / "gemma_tokenizer.bin" );
            case Family::Llama: return Mila::Data::BpeTokenizer::loadLlama32( models / "LLaMa" / "llama32_tokenizer.bin" );
            case Family::Qwen: return Mila::Data::BpeTokenizer::loadQwen( models / "Qwen" / "qwen38_tokenizer.bin" );
        }

        throw std::logic_error( "loadTokenizer: unknown family" );
    }

    /// Each model's own stop set; generation here runs at the network layer, which has none.
    std::unordered_set<std::int32_t> stopTokens( Family family )
    {
        switch ( family )
        {
            case Family::Gemma: return { 1, 106 };
            case Family::Llama: return { 128001, 128009, 128008 };
            case Family::Qwen: return { 248046, 248044 };
        }

        return {};
    }

    // G2's and L3's book turns, token for token: the loss arm reproduces those gates.
    constexpr std::int32_t kGemmaBos = 2;
    constexpr std::string_view kGemmaBookTurn =
        "<|turn>user\nContinue this book.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";

    constexpr std::int32_t kLlamaBeginOfText = 128000;
    constexpr std::string_view kLlamaBookTurn =
        "<|start_header_id|>user<|end_header_id|>\n\nContinue this book.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    /// The start of every book segment: the sequence start and a user turn asking for the book, thinking off.
    std::vector<std::int32_t> bookOpening( Family family, Mila::Data::BpeTokenizer& tokenizer )
    {
        std::vector<std::int32_t> opening;
        std::string turn;

        switch ( family )
        {
            case Family::Gemma:
                opening.push_back( kGemmaBos );
                turn = kGemmaBookTurn;
                break;

            case Family::Llama:
                opening.push_back( kLlamaBeginOfText );
                turn = kLlamaBookTurn;
                break;

            case Family::Qwen:
            {
                // No sequence start: the checkpoint sets add_bos_token false.
                const std::vector<Turns::Turn> history{ { Turns::Role::User, "Continue this book." } };
                turn = QwenProtocol::formatPrompt( history, false );
                break;
            }
        }

        const std::vector<std::int32_t> encoded = tokenizer.encode( turn );
        opening.insert( opening.end(), encoded.begin(), encoded.end() );

        return opening;
    }

    /// The PG-19 test split, sorted by file name.
    std::vector<fs::path> pg19Books( const fs::path& data_directory )
    {
        std::vector<fs::path> books;

        for ( const auto& entry : fs::directory_iterator( Measurement::pg19TestPath( data_directory ) ) )
        {
            if ( entry.path().extension() == ".txt" )
                books.push_back( entry.path() );
        }

        std::ranges::sort( books );

        return books;
    }

    template<typename TModel>
    auto measuredConfig( const fs::path& weights, dim_t log_likelihood_window )
    {
        Serialization::WeightsReader reader( weights );

        auto config = TModel::configFromMetadata( reader.getWeightsMetadata() );
        config.withLogLikelihoodWindow( log_likelihood_window );

        return config;
    }

    // ====================================================================
    // Fit (section 4.1)
    // ====================================================================

    Json footprintOf( const MemoryStats& footprint )
    {
        Json json;
        json[ "weights_bytes" ] = footprint.device_parameter_bytes;
        json[ "state_bytes" ] = footprint.device_state_bytes;
        json[ "scratch_bytes" ] = footprint.device_scratch_bytes;
        json[ "total_bytes" ] = footprint.totalDeviceBytes();

        return json;
    }

    struct FitCell
    {
        Json json;
        bool fits{ false };

        /// Refused because this context fits at no chunk -- the weights fit -- so a build past free memory can run.
        bool refused_for_memory{ false };

        std::size_t free_bytes{ 0 };
    };

    DeploymentRequest requestFor( const Configuration& configuration )
    {
        DeploymentRequest request;
        request.withWeightQuantization( configuration.weight_quantization );
        request.withKvCacheCompression( configuration.kv_cache_compression );

        return request;
    }

    /// The planner's answer for the request a user's load at `context_length` makes: window 1, the model's own types.
    template<typename TModel>
    FitCell fitAt( const Configuration& configuration, const fs::path& weights, dim_t context_length )
    {
        FitCell cell;
        cell.json[ "band" ] = context_length;

        DeploymentRequest request = requestFor( configuration );
        request.withContextLength( context_length );

        try
        {
            const auto planned = TModel::planDeployment( weights, request );

            if ( planned )
            {
                const DeploymentPlan& plan = planned->best();

                cell.fits = true;
                cell.free_bytes = plan.reading().free_bytes;
                cell.json[ "fits" ] = true;
                cell.json[ "prefill_chunk" ] = plan.prefillChunkRows();
                cell.json[ "prefill_chunk_limited_by" ] = DeploymentPlan::nameOf( plan.prefillChunkLimitedBy() );
                cell.json[ "footprint" ] = footprintOf( plan.footprint() );
            }
            else
            {
                cell.free_bytes = planned.error().reading().free_bytes;
                cell.refused_for_memory =
                    planned.error().reason() == DeploymentRefusal::Reason::FixedContextDoesNotFit;
                cell.json[ "fits" ] = false;
                cell.json[ "refused" ] = DeploymentRefusal::nameOf( planned.error().reason() );
                cell.json[ "footprint" ] = footprintOf( planned.error().footprint() );
            }
        } catch ( const std::invalid_argument& )
        {
            cell.json[ "fits" ] = false;
            cell.json[ "refused" ] = "BeyondTrainedMaximum";
        }

        return cell;
    }

    /// The context the planner chooses when the request leaves it automatic, or nothing when it refuses.
    template<typename TModel>
    std::optional<dim_t> plannedContext( const Configuration& configuration, const fs::path& weights )
    {
        DeploymentRequest request = requestFor( configuration );
        request.withAutomaticContextLength();

        const auto planned = TModel::planDeployment( weights, request );

        if ( !planned )
            return std::nullopt;

        return planned->best().contextLength();
    }

    // ====================================================================
    // Loss by band (section 4.2): G2's protocol unchanged
    // ====================================================================

    constexpr dim_t kShortestPrefix = 8192;
    constexpr dim_t kLossWindow = 64;
    constexpr dim_t kShortContext = 1024;

    /// The opening and then the book: `length` tokens in all. Empty when the book is too short to fill it.
    std::vector<std::int32_t> bookSegment( const std::vector<std::int32_t>& opening, const fs::path& book,
        Mila::Data::BpeTokenizer& tokenizer, dim_t length )
    {
        const std::size_t text_tokens = static_cast<std::size_t>( length ) - opening.size();

        // Six characters per token over-reads English prose, which runs about four; the tokens are truncated.
        const std::vector<std::int32_t> text =
            tokenizer.encode( Measurement::joinWraps( Measurement::readBook( book, text_tokens * 6 ) ) );

        if ( text.size() < text_tokens )
            return {};

        std::vector<std::int32_t> segment = opening;
        segment.insert( segment.end(), text.begin(), text.begin() + static_cast<std::ptrdiff_t>( text_tokens ) );

        return segment;
    }

    /**
     * The first `books` PG-19 test books that fill `context_length`, each scored whole at prefixes of 8192 doubling
     * to the context (and the context itself when the doubling misses it); a band is one prefix less the one before.
     * Each band's targets are also scored in blocks of 1024 after only the 1024 book tokens before them -- gate 1
     * holds when the whole book predicts no worse. Every score counts book tokens only.
     */
    template<typename TNetwork, typename TModel>
    Json lossByBand( const Configuration& configuration, const fs::path& weights, const Options& options,
        Mila::Data::BpeTokenizer& tokenizer, dim_t context_length )
    {
        std::vector<dim_t> prefixes;

        for ( dim_t prefix = kShortestPrefix; prefix <= context_length; prefix *= 2 )
            prefixes.push_back( prefix );

        if ( prefixes.empty() || prefixes.back() != context_length )
            prefixes.push_back( context_length );

        PrefillChunking chunking;
        std::unique_ptr<TNetwork> network;

        if ( options.loss_chunk )
        {
            Serialization::WeightsReader reader( weights );

            network = std::make_unique<TNetwork>( reader.getWeightsMetadata().model_name,
                measuredConfig<TModel>( weights, kLossWindow ), kDevice );
            network->build( BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( kDevice ) )
                .withPrefillSize( *options.loss_chunk ) );
            network->loadParameters( reader );
            chunking.chunk_rows = *options.loss_chunk;
        }
        else
        {
            network = Measurement::buildMeasuredNetwork<TNetwork>( weights,
                measuredConfig<TModel>( weights, kLossWindow ), kDevice, context_length, &chunking );
        }

        std::cout << std::format( "[loss] context {}, window {}, prefill chunk {}{}\n", context_length, kLossWindow,
            chunking.chunk_rows, chunking.fits_available_memory ? "" : " (does not fit free memory)" ) << std::flush;

        const std::vector<std::int32_t> opening = bookOpening( configuration.family, tokenizer );

        const auto bandOf = [&]( dim_t position )
        {
            return static_cast<std::size_t>( std::ranges::upper_bound( prefixes, position ) - prefixes.begin() );
        };

        Json result;
        result[ "context" ] = context_length;
        result[ "log_likelihood_window" ] = kLossWindow;
        result[ "prefill_chunk" ] = chunking.chunk_rows;
        result[ "prefill_chunk_fixed" ] = options.loss_chunk.has_value();
        result[ "fits_free_memory" ] = chunking.fits_available_memory;
        result[ "prefixes" ] = prefixes;
        result[ "books" ] = Json::array();

        std::vector<double> pooled_whole( prefixes.size(), 0.0 );
        std::vector<double> pooled_short( prefixes.size(), 0.0 );
        std::vector<dim_t> pooled_positions( prefixes.size(), 0 );
        std::size_t scored = 0;

        for ( const fs::path& book : pg19Books( options.data_directory ) )
        {
            if ( scored == options.books )
                break;

            const std::vector<std::int32_t> segment = bookSegment( opening, book, tokenizer, context_length );

            if ( segment.empty() )
                continue;

            ++scored;

            const SequenceLogLikelihood prompt = Measurement::sequenceLogLikelihoodOf( *network, opening );

            std::vector<double> whole( prefixes.size(), 0.0 );
            std::vector<double> short_context( prefixes.size(), 0.0 );
            std::vector<dim_t> positions( prefixes.size(), 0 );

            SequenceLogLikelihood previous = prompt;

            for ( std::size_t index = 0; index < prefixes.size(); ++index )
            {
                const SequenceLogLikelihood prefix = Measurement::sequenceLogLikelihoodOf( *network,
                    std::vector<std::int32_t>( segment.begin(), segment.begin() + static_cast<std::ptrdiff_t>( prefixes[ index ] ) ) );

                whole[ index ] = prefix.total_log_probability - previous.total_log_probability;
                positions[ index ] = prefix.scored_positions - previous.scored_positions;
                previous = prefix;
            }

            const dim_t opening_length = static_cast<dim_t>( opening.size() );

            for ( dim_t block = 0; block < context_length; block += kShortContext )
            {
                const dim_t first_target = std::max( block, opening_length );
                const dim_t context_start = std::max( opening_length, block - kShortContext );
                const dim_t block_end = std::min( block + kShortContext, context_length );

                std::vector<std::int32_t> sequence = opening;
                sequence.insert( sequence.end(), segment.begin() + static_cast<std::ptrdiff_t>( context_start ),
                    segment.begin() + static_cast<std::ptrdiff_t>( first_target ) );

                const SequenceLogLikelihood before = Measurement::sequenceLogLikelihoodOf( *network, sequence );

                sequence.insert( sequence.end(), segment.begin() + static_cast<std::ptrdiff_t>( first_target ),
                    segment.begin() + static_cast<std::ptrdiff_t>( block_end ) );

                const SequenceLogLikelihood after = Measurement::sequenceLogLikelihoodOf( *network, sequence );

                short_context[ bandOf( block ) ] += after.total_log_probability - before.total_log_probability;
            }

            Json bands = Json::array();
            std::string line = std::format( "[loss] {}:", book.stem().string() );

            for ( std::size_t index = 0; index < prefixes.size(); ++index )
            {
                const double whole_nats = -whole[ index ] / static_cast<double>( positions[ index ] );
                const double short_nats = -short_context[ index ] / static_cast<double>( positions[ index ] );

                Json band;
                band[ "start" ] = index == 0 ? 0 : prefixes[ index - 1 ];
                band[ "end" ] = prefixes[ index ];
                band[ "positions" ] = positions[ index ];
                band[ "whole_book_nats" ] = whole_nats;
                band[ "short_context_nats" ] = short_nats;
                band[ "gate_1" ] = whole_nats <= short_nats;
                bands.push_back( band );

                pooled_whole[ index ] += whole[ index ];
                pooled_short[ index ] += short_context[ index ];
                pooled_positions[ index ] += positions[ index ];

                line += std::format( " {:.4f}", whole_nats );
            }

            Json entry;
            entry[ "book" ] = book.stem().string();
            entry[ "bands" ] = bands;
            result[ "books" ].push_back( entry );

            std::cout << line << "\n" << std::flush;
        }

        if ( scored < options.books )
        {
            throw std::runtime_error( std::format( "only {} PG-19 test books fill {} tokens; {} were asked for", scored,
                context_length, options.books ) );
        }

        Json pooled = Json::array();

        for ( std::size_t index = 0; index < prefixes.size(); ++index )
        {
            Json band;
            band[ "start" ] = index == 0 ? 0 : prefixes[ index - 1 ];
            band[ "end" ] = prefixes[ index ];
            band[ "positions" ] = pooled_positions[ index ];
            band[ "whole_book_nats" ] = -pooled_whole[ index ] / static_cast<double>( pooled_positions[ index ] );
            band[ "short_context_nats" ] = -pooled_short[ index ] / static_cast<double>( pooled_positions[ index ] );
            pooled.push_back( band );
        }

        result[ "pooled" ] = pooled;

        return result;
    }

    // ====================================================================
    // Recall at depth (section 4.3)
    // ====================================================================

    constexpr std::size_t kRecordsPerDepth = 8;
    constexpr std::size_t kAbsentPerConversation = 8;
    constexpr double kDepths[] = { 0.0, 0.25, 0.5, 0.75 };
    constexpr std::string_view kDepthNames[] = { "0%", "25%", "50%", "75%", "end" };
    constexpr std::size_t kDepthCount = std::size( kDepthNames );

    // What the conversation leaves free in the band: the question, the reply and the scored answer.
    constexpr dim_t kQuestionReserve = 512;
    constexpr int kReplyTokens = 24;

    // With thinking on: room for the reasoning and the answer. A reply that reaches it unclosed is a miss, counted apart.
    constexpr int kThinkingReplyTokens = 4096;

    // A read returns about this much book, cut at the next paragraph.
    constexpr std::size_t kChunkCharacters = 6000;

    constexpr std::uint64_t kRecallSeed = 0x436f6e7465787450ull;

    constexpr std::string_view kTask =
        "Read the documents and records with your tools. When you are done, I will ask you about the records.";
    constexpr std::string_view kClosing = "I have read the documents and the records.";

    /// splitmix64: the same identifiers on every platform and every run.
    class IdentifierSource
    {
    public:

        explicit IdentifierSource( std::uint64_t seed )
            : state_( seed )
        {
        }

        std::uint64_t next()
        {
            std::uint64_t z = ( state_ += 0x9e3779b97f4a7c15ull );
            z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ull;
            z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebull;

            return z ^ ( z >> 31 );
        }

        /// Letters and digits a reader cannot confuse: no 0, 1, I, L or O.
        std::string draw( std::size_t length )
        {
            constexpr std::string_view kAlphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";

            std::string text;

            for ( std::size_t index = 0; index < length; ++index )
                text.push_back( kAlphabet[ next() % kAlphabet.size() ] );

            return text;
        }

    private:

        std::uint64_t state_;
    };

    struct Record
    {
        std::string key;
        std::string value;

        /// Index into kDepthNames, or kDepthCount for a key never planted.
        std::size_t depth;
    };

    /// One tool call and what the tool returned.
    struct ToolExchange
    {
        std::string tool;
        std::string arguments;
        std::string result;
    };

    ToolExchange recordExchange( const Record& record )
    {
        Json arguments;
        arguments[ "record" ] = record.key;

        Json result;
        result[ "record" ] = record.key;
        result[ "value" ] = record.value;

        return { "get_record", arguments.dump(), result.dump() };
    }

    /// Book text read in order, a paragraph-aligned chunk at a time, from the first book onward and then the next.
    class BookText
    {
    public:

        BookText( std::vector<fs::path> books, std::size_t first )
            : books_( std::move( books ) ), book_( first % books_.size() )
        {
        }

        ToolExchange next()
        {
            while ( text_.empty() || offset_ + kChunkCharacters / 6 >= text_.size() )
            {
                if ( !text_.empty() )
                    book_ = ( book_ + 1 ) % books_.size();

                text_ = Measurement::joinWraps( Measurement::readBook( books_[ book_ ], 1u << 24 ) );
                offset_ = 0;
                part_ = 0;
            }

            std::size_t end = std::min( offset_ + kChunkCharacters, text_.size() );
            const std::size_t paragraph = text_.find( "\n\n", end );

            if ( paragraph != std::string::npos && paragraph < end + kChunkCharacters / 2 )
                end = paragraph;

            std::string chunk = text_.substr( offset_, end - offset_ );
            offset_ = end;

            const std::size_t start = chunk.find_first_not_of( " \n" );
            chunk = start == std::string::npos ? std::string{} : chunk.substr( start );

            Json arguments;
            arguments[ "document" ] = "pg19-" + books_[ book_ ].stem().string();
            arguments[ "part" ] = ++part_;

            return { "read_document", arguments.dump(), chunk };
        }

    private:

        std::vector<fs::path> books_;
        std::size_t book_;
        std::string text_;
        std::size_t offset_{ 0 };
        int part_{ 0 };
    };

    // Gemma 4's thinking trigger leads a system turn of its own (GemmaChatProtocol.md); with no instruction beside it,
    // the model decides how long to think. Qwen's is the open reasoning span its formatPrompt leaves, at the
    // checkpoint's default effort, which adds no instruction either.
    constexpr std::string_view kGemmaThink = "<|think|>";

    std::string gemmaModelTurn( const std::vector<ToolExchange>& exchanges )
    {
        // Gemma splices each response into the model turn that made the call (GemmaChatProtocol.md).
        std::string model_turn;

        for ( const ToolExchange& exchange : exchanges )
        {
            model_turn += GemmaProtocol::formatToolCall( exchange.tool, exchange.arguments );
            model_turn += GemmaProtocol::formatToolResponse( exchange.tool, exchange.result );
        }

        return model_turn + std::string( kClosing );
    }

    /// A conversation rendered in the family's grammar: everything before the questions, as text.
    std::string conversationText( Family family, const std::vector<ToolExchange>& exchanges, bool thinking )
    {
        if ( family == Family::Gemma )
        {
            std::string text( GemmaProtocol::kBos );

            if ( thinking )
                text += GemmaProtocol::formatTurn( { Turns::Role::System, std::string( kGemmaThink ) } );

            return text
                + GemmaProtocol::formatTurn( { Turns::Role::User, std::string( kTask ) } )
                + GemmaProtocol::formatTurn( { Turns::Role::Assistant, gemmaModelTurn( exchanges ) } );
        }

        // Qwen: each call an assistant turn, each result a user turn carrying a tool_response span.
        std::string text = QwenProtocol::formatTurn( { Turns::Role::User, std::string( kTask ) } );

        for ( const ToolExchange& exchange : exchanges )
        {
            text += QwenProtocol::formatTurn( { Turns::Role::Assistant, {}, { { exchange.tool, exchange.arguments } } } );
            text += QwenProtocol::formatTurn( { Turns::Role::Tool, exchange.result } );
        }

        text += QwenProtocol::formatTurn( { Turns::Role::Assistant, std::string( kClosing ) } );

        return text;
    }

    std::string questionFor( const Record& record )
    {
        return std::format( "What is the value of record {}? Reply with the value only.", record.key );
    }

    /// The question's turn and the primed model turn, as the family's formatPrompt renders them after `conversation`.
    std::string questionText( Family family, const std::vector<ToolExchange>& exchanges, const std::string& conversation,
        const Record& record, bool thinking )
    {
        std::string full;

        if ( family == Family::Gemma )
        {
            std::vector<Turns::Turn> history;

            if ( thinking )
                history.push_back( { Turns::Role::System, std::string( kGemmaThink ) } );

            history.push_back( { Turns::Role::User, std::string( kTask ) } );
            history.push_back( { Turns::Role::Assistant, gemmaModelTurn( exchanges ) } );
            history.push_back( { Turns::Role::User, questionFor( record ) } );

            full = GemmaProtocol::formatPrompt( history );

            // Mila/Src's formatPrompt always closes an empty thought channel, the thinking-off primer; with thinking on
            // the model opens the channel itself, so the primer stops at the model turn, as Chat renders it.
            if ( thinking )
            {
                if ( !full.ends_with( GemmaProtocol::kThoughtPrime ) )
                    throw std::logic_error( "Gemma's prompt no longer ends with the thinking-off primer" );

                full.resize( full.size() - GemmaProtocol::kThoughtPrime.size() );
            }
        }
        else
        {
            std::vector<Turns::Turn> history{ { Turns::Role::User, std::string( kTask ) } };

            for ( const ToolExchange& exchange : exchanges )
            {
                history.push_back( { Turns::Role::Assistant, {}, { { exchange.tool, exchange.arguments } } } );
                history.push_back( { Turns::Role::Tool, exchange.result } );
            }

            history.push_back( { Turns::Role::Assistant, std::string( kClosing ) } );
            history.push_back( { Turns::Role::User, questionFor( record ) } );

            full = QwenProtocol::formatPrompt( history, thinking );
        }

        if ( !full.starts_with( conversation ) )
            throw std::logic_error( "the family's prompt does not begin with the conversation it was given" );

        return full.substr( conversation.size() );
    }

    struct RecallConversation
    {
        std::vector<ToolExchange> exchanges;
        std::vector<Record> records;
        std::string text;
        std::vector<std::int32_t> tokens;

        /// Each depth's records begin about here, in tokens: an estimate from the pieces' own encodings.
        std::vector<dim_t> depth_positions;
    };

    /**
     * Book reads to fill the band less kQuestionReserve, with eight records at each of 0, 25, 50 and 75% of it and
     * eight just before the question, and eight keys never planted. Deterministic in the seed, the band and the
     * conversation's index.
     */
    RecallConversation buildConversation( Family family, Mila::Data::BpeTokenizer& tokenizer,
        const std::vector<fs::path>& books, dim_t band, std::size_t index )
    {
        IdentifierSource identifiers( kRecallSeed ^ ( static_cast<std::uint64_t>( band ) << 20 ) ^ index );
        RecallConversation conversation;

        for ( std::size_t depth = 0; depth <= kDepthCount; ++depth )
        {
            const std::size_t count = depth == kDepthCount ? kAbsentPerConversation : kRecordsPerDepth;

            for ( std::size_t record = 0; record < count; ++record )
            {
                std::string value = identifiers.draw( 4 ) + "-" + identifiers.draw( 4 );
                conversation.records.push_back( { identifiers.draw( 6 ), std::move( value ), depth } );
            }
        }

        const auto tokensOf = [&]( const std::vector<ToolExchange>& pieces )
        {
            return static_cast<dim_t>( tokenizer.encode( conversationText( family, pieces, false ) ).size() );
        };

        const auto groupOf = [&]( std::size_t depth )
        {
            std::vector<ToolExchange> group;

            for ( const Record& record : conversation.records )
            {
                if ( record.depth == depth )
                    group.push_back( recordExchange( record ) );
            }

            return group;
        };

        const dim_t budget = band - kQuestionReserve;
        const dim_t empty = tokensOf( {} );
        const dim_t final_group = tokensOf( groupOf( kDepthCount - 1 ) ) - empty;

        BookText text( books, index );
        std::vector<ToolExchange> exchanges;
        std::vector<std::size_t> read_indices;
        dim_t total = empty + final_group;
        std::size_t next_depth = 0;

        conversation.depth_positions.assign( kDepthCount, 0 );

        while ( true )
        {
            while ( next_depth < std::size( kDepths ) && total >= static_cast<dim_t>( kDepths[ next_depth ] * budget ) )
            {
                conversation.depth_positions[ next_depth ] = total - final_group;

                const std::vector<ToolExchange> group = groupOf( next_depth );
                exchanges.insert( exchanges.end(), group.begin(), group.end() );
                total += tokensOf( group ) - empty;
                ++next_depth;
            }

            ToolExchange read = text.next();
            const dim_t cost = tokensOf( { read } ) - empty;

            if ( total + cost > budget )
                break;

            read_indices.push_back( exchanges.size() );
            exchanges.push_back( std::move( read ) );
            total += cost;
        }

        if ( next_depth < std::size( kDepths ) )
            throw std::runtime_error( std::format( "band {} is too short to place records at every depth", band ) );

        const std::vector<ToolExchange> final_records = groupOf( kDepthCount - 1 );
        conversation.depth_positions[ kDepthCount - 1 ] = total - final_group;

        // The pieces' encodings sum to about the whole's; a read is dropped until the whole fits.
        while ( true )
        {
            std::vector<ToolExchange> all = exchanges;
            all.insert( all.end(), final_records.begin(), final_records.end() );

            conversation.text = conversationText( family, all, false );
            conversation.tokens = tokenizer.encode( conversation.text );

            if ( static_cast<dim_t>( conversation.tokens.size() ) <= budget )
            {
                conversation.exchanges = std::move( all );
                break;
            }

            if ( read_indices.empty() )
                throw std::runtime_error( std::format( "band {} holds no book text beside its records", band ) );

            exchanges.erase( exchanges.begin() + static_cast<std::ptrdiff_t>( read_indices.back() ) );
            read_indices.pop_back();
        }

        return conversation;
    }

    /// Greedy decode through one persistent token tensor, so a decode step replays from its recording.
    template<typename TNetwork>
    class GreedyDecoder
    {
    public:

        explicit GreedyDecoder( TNetwork& network )
            : network_( network ),
              host_( Device::Cpu(), shape_t{ 1, 1 } ),
              device_( network.getDeviceId(), shape_t{ 1, 1 } )
        {
        }

        /// The reply after `logits`, the prefill's last row; the stop token that ends it is not part of it.
        std::vector<std::int32_t> reply( const std::vector<float>& logits, dim_t position,
            const std::unordered_set<std::int32_t>& stops, dim_t context_length, int reply_tokens )
        {
            std::vector<std::int32_t> tokens;
            std::int32_t token = Measurement::argMax( logits );

            while ( !stops.contains( token ) )
            {
                tokens.push_back( token );

                if ( static_cast<int>( tokens.size() ) >= reply_tokens || position >= context_length )
                    break;

                host_.data()[ 0 ] = token;
                copy( host_, device_ );

                token = Measurement::argMax( Measurement::hostLogits( network_, network_.decode( device_, position ) ) );
                ++position;
            }

            return tokens;
        }

        /**
         * The log-probability of `answer` after `logits`, the prefill's last row, each token forced in by a decode
         * step: the path the reply came from. A difference of two prefill scores is not used, because a prefill's row
         * can depend on the other rows of its chunk (FP4's activations share a scale across it), so the question's
         * own score differs between the two calls and leaks into the answer's.
         */
        double answerLogProbability( std::vector<float> logits, dim_t position, const std::vector<std::int32_t>& answer,
            float final_logit_softcap )
        {
            double total = 0.0;

            for ( std::size_t index = 0; index < answer.size(); ++index )
            {
                total += nextTokenLogProbability( logits.data(), static_cast<dim_t>( logits.size() ), answer[ index ],
                    final_logit_softcap );

                if ( index + 1 == answer.size() )
                    break;

                host_.data()[ 0 ] = answer[ index ];
                copy( host_, device_ );

                logits = Measurement::hostLogits( network_, network_.decode( device_, position ) );
                ++position;
            }

            return total;
        }

    private:

        TNetwork& network_;
        Tensor<TensorDataType::INT32, CpuMemoryResource> host_;
        typename TNetwork::TokenIndexType device_;
    };

    /// The 95% Wilson interval of a proportion.
    std::pair<double, double> wilsonInterval( std::size_t successes, std::size_t trials )
    {
        if ( trials == 0 )
            return { 0.0, 1.0 };

        constexpr double z = 1.959963984540054;

        const double n = static_cast<double>( trials );
        const double p = static_cast<double>( successes ) / n;
        const double denominator = 1.0 + z * z / n;
        const double center = ( p + z * z / ( 2.0 * n ) ) / denominator;
        const double half = z * std::sqrt( p * ( 1.0 - p ) / n + z * z / ( 4.0 * n * n ) ) / denominator;

        return { std::max( 0.0, center - half ), std::min( 1.0, center + half ) };
    }

    template<typename TNetwork, typename TModel>
    Json recallAtBand( const Configuration& configuration, const fs::path& weights, const Options& options,
        Mila::Data::BpeTokenizer& tokenizer, dim_t band, const std::function<void( const Json& )>& on_conversation )
    {
        PrefillChunking chunking;

        // Window 1 and the planner's chunk: the network a user's load at this context builds. With thinking on it holds
        // the reasoning's room beyond the band, so the conversations are the thinking-off run's, record for record.
        const bool thinking = options.thinking;
        const int reply_tokens = thinking ? kThinkingReplyTokens : kReplyTokens;
        const dim_t network_context = band + ( thinking ? kThinkingReplyTokens : 0 );

        const auto config = measuredConfig<TModel>( weights, 1 );
        auto network = Measurement::buildMeasuredNetwork<TNetwork>( weights, config, kDevice, network_context, &chunking );
        network->setDecodeReplay( true );

        // Decode returns logits before the cap a Gemma samples through; the other families have none.
        float final_logit_softcap = 0.0f;

        if constexpr ( requires { config.getFinalLogitSoftcapping(); } )
            final_logit_softcap = config.getFinalLogitSoftcapping();

        std::cout << std::format( "[recall] band {}, prefill chunk {}\n", band, chunking.chunk_rows ) << std::flush;

        const std::vector<fs::path> books = pg19Books( options.data_directory );
        const std::unordered_set<std::int32_t> stops = stopTokens( configuration.family );

        GreedyDecoder decoder( *network );

        // The token that closes the reasoning: what comes after it is the answer.
        std::int32_t thinking_close = -1;

        if ( thinking )
        {
            const std::vector<std::int32_t> marker = tokenizer.encode( configuration.family == Family::Gemma
                ? std::string( GemmaProtocol::kChannelClose )
                : std::string( QwenProtocol::kThinkClose ) );

            if ( marker.size() != 1 )
                throw std::logic_error( "the reasoning's closing marker is not one token in this vocabulary" );

            thinking_close = marker.front();
        }

        std::vector<double> thinking_tokens( kDepthCount + 1, 0.0 );
        std::vector<std::size_t> unclosed( kDepthCount + 1, 0 );
        std::vector<std::size_t> tool_calls( kDepthCount + 1, 0 );
        std::size_t refills = 0;

        // A thinking model unsure of a record reaches for the tool. An agent's harness stops at the call and runs it;
        // here it stops at the call and counts it, an attempt to recover rather than a wrong answer.
        std::unordered_set<std::int32_t> reply_stops = stops;
        const std::string tool_call_open = configuration.family == Family::Gemma
            ? std::string( GemmaProtocol::kToolCallOpen )
            : std::string( QwenProtocol::kToolCallOpen );

        if ( thinking )
        {
            const std::vector<std::int32_t> close = tokenizer.encode( configuration.family == Family::Gemma
                ? std::string( GemmaProtocol::kToolCallClose )
                : std::string( QwenProtocol::kToolCallClose ) );

            if ( close.size() != 1 )
                throw std::logic_error( "the tool call's closing marker is not one token in this vocabulary" );

            reply_stops.insert( close.front() );
        }

        std::vector<std::size_t> exact( kDepthCount + 1, 0 );
        std::vector<std::size_t> trials( kDepthCount + 1, 0 );
        std::vector<double> answer_log_likelihood( kDepthCount + 1, 0.0 );
        std::vector<dim_t> answer_tokens( kDepthCount + 1, 0 );

        Json conversations = Json::array();

        // The band so far: written after every conversation, so a long run stopped early keeps what it measured.
        const auto summarize = [ & ]
        {
            Json cells = Json::array();

            for ( std::size_t depth = 0; depth <= kDepthCount; ++depth )
            {
                const auto [ low, high ] = wilsonInterval( exact[ depth ], trials[ depth ] );

                Json cell;
                cell[ "depth" ] = depth < kDepthCount ? kDepthNames[ depth ] : "absent";
                cell[ "trials" ] = trials[ depth ];
                cell[ "exact" ] = exact[ depth ];
                cell[ "exact_proportion" ] = static_cast<double>( exact[ depth ] ) / static_cast<double>( trials[ depth ] );
                cell[ "exact_interval" ] = { low, high };

                if ( thinking )
                {
                    cell[ "mean_thinking_tokens" ] = thinking_tokens[ depth ] / static_cast<double>( trials[ depth ] );
                    cell[ "unclosed" ] = unclosed[ depth ];
                    cell[ "called_tool" ] = tool_calls[ depth ];
                }
                else
                {
                    cell[ "mean_answer_log_likelihood" ] = answer_log_likelihood[ depth ] / static_cast<double>( trials[ depth ] );
                    cell[ "answer_nats_per_token" ] = -answer_log_likelihood[ depth ] / static_cast<double>( answer_tokens[ depth ] );
                }

                cells.push_back( cell );
            }

            Json result;
            result[ "band" ] = band;
            result[ "prefill_chunk" ] = chunking.chunk_rows;
            result[ "fits_free_memory" ] = chunking.fits_available_memory;
            result[ "thinking" ] = thinking;
            result[ "conversation_refills" ] = refills;
            result[ "cells" ] = cells;
            result[ "conversations" ] = conversations;

            return result;
        };


        for ( std::size_t index = 0; index < options.conversations; ++index )
        {
            RecallConversation conversation = buildConversation( configuration.family, tokenizer, books, band, index );

            if ( thinking )
            {
                conversation.text = conversationText( configuration.family, conversation.exchanges, true );
                conversation.tokens = tokenizer.encode( conversation.text );
            }

            const dim_t end = static_cast<dim_t>( conversation.tokens.size() );

            const auto fill = [ & ]
            {
                (void)Measurement::hostLogits( *network,
                    network->prefill( Measurement::deviceTokens( *network, conversation.tokens ) ) );

                if ( network->savePosition() != end )
                    throw std::logic_error( "the network's fill is not the conversation's length" );
            };

            // Back to the end of the conversation for the next question. A reply longer than Gemma's sliding ring
            // holds overwrites the positions a rewind needs, and the rewind is refused; the conversation is then
            // prefilled again, as a model does when it cannot reuse a prefix. Only a long reasoning reply does this.
            const auto returnToEnd = [ & ]
            {
                if ( !network->rewindKvCache( end ) )
                {
                    fill();
                    ++refills;
                }
            };

            fill();

            Json answers = Json::array();

            for ( const Record& record : conversation.records )
            {
                const std::string question_text =
                    questionText( configuration.family, conversation.exchanges, conversation.text, record, thinking );
                const std::vector<std::int32_t> question = tokenizer.encode( question_text );

                std::vector<std::int32_t> prompt = conversation.tokens;
                prompt.insert( prompt.end(), question.begin(), question.end() );

                // Encoding the two halves apart must give the whole's encoding, or the questions are asked of a
                // different conversation than the one prefilled. Checked once a conversation; the boundary is a
                // turn marker each time.
                if ( &record == &conversation.records.front()
                    && tokenizer.encode( conversation.text + question_text ) != prompt )
                {
                    throw std::logic_error( "the conversation and its question encode differently apart and together" );
                }

                returnToEnd();

                const std::vector<float> logits = Measurement::hostLogits( *network,
                    network->prefillFrom( Measurement::deviceTokens( *network, prompt ), end ) );

                const std::vector<std::int32_t> reply =
                    decoder.reply( logits, static_cast<dim_t>( prompt.size() ), reply_stops, network_context, reply_tokens );

                // Thinking on: the answer is what follows the closing marker. Gemma may answer without opening the
                // channel at all; Qwen's prompt opens the span, so no marker means no answer.
                std::size_t reasoning = 0;
                bool closed = true;
                std::vector<std::int32_t> answered = reply;

                if ( thinking )
                {
                    const auto marker = std::ranges::find( reply, thinking_close );

                    if ( marker != reply.end() )
                    {
                        reasoning = static_cast<std::size_t>( marker - reply.begin() ) + 1;
                        answered.assign( marker + 1, reply.end() );
                    }
                    else if ( configuration.family == Family::Qwen || static_cast<int>( reply.size() ) >= reply_tokens )
                    {
                        reasoning = reply.size();
                        closed = false;
                        answered.clear();
                    }
                }

                const std::string reply_text = tokenizer.decode( answered );
                const bool recalled = reply_text.find( record.value ) != std::string::npos;

                const std::vector<std::int32_t> value = tokenizer.encode( record.value );

                // Teacher-forcing the value after free-running reasoning is not defined, so thinking runs do not score it.
                double log_likelihood = 0.0;

                if ( !thinking )
                {
                    returnToEnd();

                    log_likelihood = decoder.answerLogProbability( Measurement::hostLogits( *network,
                        network->prefillFrom( Measurement::deviceTokens( *network, prompt ), end ) ),
                        static_cast<dim_t>( prompt.size() ), value, final_logit_softcap );

                    // A sum of log-probabilities: above zero means the score read something other than the answer.
                    if ( !( log_likelihood <= 1e-6 ) )
                    {
                        throw std::logic_error( std::format( "record {}: an answer log-likelihood of {} is not a "
                            "log-probability", record.key, log_likelihood ) );
                    }
                }

                exact[ record.depth ] += recalled ? 1 : 0;
                trials[ record.depth ] += 1;
                answer_log_likelihood[ record.depth ] += log_likelihood;
                answer_tokens[ record.depth ] += static_cast<dim_t>( value.size() );
                thinking_tokens[ record.depth ] += static_cast<double>( reasoning );
                unclosed[ record.depth ] += closed ? 0 : 1;

                const bool called_tool = thinking && tokenizer.decode( reply ).find( tool_call_open ) != std::string::npos;
                tool_calls[ record.depth ] += called_tool ? 1 : 0;

                Json answer;
                answer[ "depth" ] = record.depth < kDepthCount ? kDepthNames[ record.depth ] : "absent";
                answer[ "key" ] = record.key;
                answer[ "value" ] = record.value;
                answer[ "recalled" ] = recalled;
                answer[ "answer_log_likelihood" ] = thinking ? Json( nullptr ) : Json( log_likelihood );
                answer[ "answer_tokens" ] = value.size();
                answer[ "reply" ] = reply_text;

                if ( thinking )
                {
                    answer[ "thinking_tokens" ] = reasoning;
                    answer[ "reasoning_closed" ] = closed;
                    answer[ "called_tool" ] = called_tool;
                    answer[ "reasoning" ] = tokenizer.decode( std::vector<std::int32_t>( reply.begin(),
                        reply.begin() + static_cast<std::ptrdiff_t>( reasoning ) ) );
                }
                answers.push_back( answer );
            }

            Json entry;
            entry[ "conversation" ] = index;
            entry[ "tokens" ] = end;
            entry[ "depth_positions" ] = conversation.depth_positions;
            entry[ "answers" ] = answers;
            conversations.push_back( entry );

            std::size_t planted = 0, recalled = 0;

            for ( const Json& answer : answers )
            {
                if ( answer[ "depth" ] != "absent" )
                {
                    ++planted;
                    recalled += answer[ "recalled" ].get<bool>() ? 1 : 0;
                }
            }

            std::cout << std::format( "[recall] band {} conversation {}: {} tokens, {} of {} planted recalled\n", band,
                index, end, recalled, planted ) << std::flush;

            on_conversation( summarize() );
        }

        return summarize();
    }

    // ====================================================================
    // Output (section 6)
    // ====================================================================

    constexpr double kReliableRecall = 0.9;

    /// The deepest band whose every planted depth recalls at least 90% (section 9, decision 5); recall alone in Phase 1.
    std::optional<dim_t> reliableDepthOnRecall( const Json& recall )
    {
        std::optional<dim_t> deepest;

        for ( const Json& band : recall )
        {
            bool holds = true;

            for ( const Json& cell : band[ "cells" ] )
            {
                if ( cell[ "depth" ] != "absent" && cell[ "exact_proportion" ].get<double>() < kReliableRecall )
                    holds = false;
            }

            if ( holds )
                deepest = band[ "band" ].get<dim_t>();
        }

        return deepest;
    }

    std::string mebibytes( const Json& bytes )
    {
        return std::format( "{:.0f}", bytes.get<double>() / ( 1024.0 * 1024.0 ) );
    }

    std::string markdownOf( const Json& profile )
    {
        const Json& configuration = profile[ "configuration" ];

        std::string text = std::format( "# Context profile: {}\n\n{} weights, {} cache, on {}. Mila {}.\n\n",
            configuration[ "name" ].get<std::string>(), configuration[ "weight_format" ].get<std::string>(),
            configuration[ "kv_cache" ].get<std::string>(), profile[ "card" ][ "name" ].get<std::string>(),
            profile[ "build" ][ "mila" ].get<std::string>() );

        if ( profile.contains( "fit" ) )
        {
            text += "## Fit\n\n| Band | Fits | Prefill chunk | Weights MiB | State MiB | Scratch MiB | Total MiB |\n"
                "|---|---|---|---|---|---|---|\n";

            for ( const Json& cell : profile[ "fit" ] )
            {
                const bool fits = cell[ "fits" ].get<bool>();
                const bool priced = cell.contains( "footprint" );

                text += std::format( "| {} | {} | {} | {} | {} | {} | {} |\n", cell[ "band" ].get<dim_t>(),
                    fits ? "yes" : "refused, " + cell[ "refused" ].get<std::string>(),
                    fits ? std::to_string( cell[ "prefill_chunk" ].get<dim_t>() ) : "--",
                    priced ? mebibytes( cell[ "footprint" ][ "weights_bytes" ] ) : "--",
                    priced ? mebibytes( cell[ "footprint" ][ "state_bytes" ] ) : "--",
                    priced ? mebibytes( cell[ "footprint" ][ "scratch_bytes" ] ) : "--",
                    priced ? mebibytes( cell[ "footprint" ][ "total_bytes" ] ) : "--" );
            }

            text += "\n";
        }

        if ( profile.contains( "loss" ) )
        {
            const Json& loss = profile[ "loss" ];

            text += std::format( "## Loss by band\n\nNats a token over the book tokens, {} books, context {}, prefill "
                "chunk {}. Short context: the 1024 book tokens before each block of 1024 targets.\n\n"
                "| Band | Positions | Whole book | Short context |\n|---|---|---|---|\n",
                loss[ "books" ].size(), loss[ "context" ].get<dim_t>(), loss[ "prefill_chunk" ].get<dim_t>() );

            for ( const Json& band : loss[ "pooled" ] )
            {
                text += std::format( "| {} - {} | {} | {:.4f} | {:.4f} |\n", band[ "start" ].get<dim_t>(),
                    band[ "end" ].get<dim_t>(), band[ "positions" ].get<dim_t>(),
                    band[ "whole_book_nats" ].get<double>(), band[ "short_context_nats" ].get<double>() );
            }

            text += "\n";
        }

        if ( profile.contains( "recall" ) )
        {
            text += "## Recall at depth\n\nExact recall, percent of trials, by where the record sat in the conversation; "
                "absent asks for a key never planted.\n\n| Band |";

            for ( const std::string_view depth : kDepthNames )
                text += std::format( " {} |", depth );

            text += " absent |\n|---|";

            for ( std::size_t column = 0; column <= kDepthCount; ++column )
                text += "---|";

            text += "\n";

            for ( const Json& band : profile[ "recall" ] )
            {
                text += std::format( "| {}{} |", band[ "band" ].get<dim_t>(),
                    band.value( "fits_free_memory", true ) ? "" : " (past free memory)" );

                for ( const Json& cell : band[ "cells" ] )
                {
                    text += std::format( " {:.0f} ({}) |", 100.0 * cell[ "exact_proportion" ].get<double>(),
                        cell[ "trials" ].get<std::size_t>() );
                }

                text += "\n";
            }

            const bool thinking = profile[ "recall" ].front().value( "thinking", false );

            text += thinking
                ? "\nMean thinking tokens before the answer; in brackets the replies that never closed it, then those that "
                  "called the tool instead of answering:\n\n| Band |"
                : "\nMean answer log-likelihood, nats a token of the value:\n\n| Band |";

            for ( const std::string_view depth : kDepthNames )
                text += std::format( " {} |", depth );

            text += " absent |\n|---|";

            for ( std::size_t column = 0; column <= kDepthCount; ++column )
                text += "---|";

            text += "\n";

            for ( const Json& band : profile[ "recall" ] )
            {
                text += std::format( "| {}{} |", band[ "band" ].get<dim_t>(),
                    band.value( "fits_free_memory", true ) ? "" : " (past free memory)" );

                for ( const Json& cell : band[ "cells" ] )
                {
                    if ( thinking )
                    {
                        const std::size_t open = cell[ "unclosed" ].get<std::size_t>();

                        const std::size_t calls = cell[ "called_tool" ].get<std::size_t>();

                        text += std::format( " {:.0f}{}{} |", cell[ "mean_thinking_tokens" ].get<double>(),
                            open == 0 ? std::string{} : std::format( " ({})", open ),
                            calls == 0 ? std::string{} : std::format( ", {} called the tool", calls ) );
                    }
                    else
                    {
                        text += std::format( " {:.3f} |", cell[ "answer_nats_per_token" ].get<double>() );
                    }
                }

                text += "\n";
            }

            text += "\n";
        }

        if ( profile.contains( "reliable_depth" ) )
        {
            const Json& reliable = profile[ "reliable_depth" ];

            text += std::format( "## Reliable depth\n\n{}, on recall alone: instruction retention and tool-call fidelity "
                "are not yet measured.\n", reliable[ "band" ].is_null()
                    ? std::string( "No band" )
                    : std::to_string( reliable[ "band" ].get<dim_t>() ) );
        }

        return text;
    }

    void writeText( const fs::path& path, const std::string& text )
    {
        std::FILE* file = std::fopen( path.string().c_str(), "wb" );

        if ( file == nullptr )
            throw std::runtime_error( std::format( "cannot write {}", path.string() ) );

        std::fwrite( text.data(), 1, text.size(), file );
        std::fclose( file );
    }

    // ====================================================================
    // Run
    // ====================================================================

    template<typename TNetwork, typename TModel>
    void profileConfiguration( const Configuration& configuration, const Options& options )
    {
        const fs::path models = options.data_directory / "Models";
        const fs::path weights = options.weights.empty() ? models / configuration.weights : options.weights;
        const fs::path output = options.output.empty()
            ? fs::path( std::string( configuration.name ) + ".json" )
            : options.output;

        if ( !fs::exists( weights ) )
            throw std::runtime_error( std::format( "no weights at {}", weights.string() ) );

        const auto device = DeviceRegistry::instance().getDevice( kDevice );

        Json profile;
        profile[ "configuration" ][ "name" ] = configuration.name;
        profile[ "configuration" ][ "family" ] = familyName( configuration.family );
        profile[ "configuration" ][ "weights" ] = weights.filename().string();
        profile[ "configuration" ][ "weight_format" ] = configuration.weight_format;
        profile[ "configuration" ][ "kv_cache" ] = configuration.kv_cache;
        profile[ "configuration" ][ "thinking" ] = options.thinking ? "on, as long as the model chooses" : "off";
        profile[ "build" ][ "mila" ] = Mila::getAPIVersion().toString();
        const auto* cuda_device = dynamic_cast<const CudaDevice*>( device.get() );
        profile[ "card" ][ "name" ] = cuda_device ? cuda_device->getProperties().getName() : std::string( "unknown" );
        profile[ "card" ][ "total_bytes" ] = DeviceReading::take( kDevice ).total_bytes;

        Json run_record;
        run_record[ "readings_free_bytes" ] = Json::array();

        // The output is rewritten as the run goes, marked incomplete until it ends: stopping a long run keeps
        // every band and conversation it finished.
        const auto checkpoint = [ & ]
        {
            Json snapshot = profile;
            snapshot[ "run" ] = run_record;
            snapshot[ "run" ][ "complete" ] = false;
            writeText( output, snapshot.dump( 2 ) + "\n" );
        };

        const auto started = std::chrono::steady_clock::now();
        const auto secondsSince = []( std::chrono::steady_clock::time_point since )
        {
            return std::chrono::duration<double>( std::chrono::steady_clock::now() - since ).count();
        };

        std::vector<dim_t> bands = options.bands;

        if ( bands.empty() )
        {
            bands.assign( std::begin( kStandardBands ), std::end( kStandardBands ) );

            if ( const auto planned = plannedContext<TModel>( configuration, weights ) )
            {
                if ( std::ranges::find( bands, *planned ) == bands.end() )
                    bands.push_back( *planned );

                profile[ "planned_context" ] = *planned;
            }

            std::ranges::sort( bands );
        }

        std::vector<dim_t> measured = bands;

        if ( options.fit )
        {
            const auto arm_started = std::chrono::steady_clock::now();

            Json fit = Json::array();
            measured.clear();

            for ( const dim_t band : bands )
            {
                FitCell cell = fitAt<TModel>( configuration, weights, band );

                run_record[ "readings_free_bytes" ].push_back( cell.free_bytes );
                fit.push_back( cell.json );

                if ( cell.fits || ( options.past_free_memory && cell.refused_for_memory ) )
                    measured.push_back( band );

                if ( !cell.fits && options.past_free_memory && cell.refused_for_memory )
                    profile[ "measured_past_free_memory" ].push_back( band );

                std::cout << std::format( "[fit] band {}: {}\n", band, cell.fits ? "fits" : "refused" ) << std::flush;
            }

            profile[ "fit" ] = fit;
            run_record[ "fit_seconds" ] = secondsSince( arm_started );
            checkpoint();
        }

        profile[ "measured_bands" ] = measured;

        if ( measured.empty() )
        {
            std::cout << "no band fits; nothing further to measure\n";
        }

        const std::shared_ptr<Mila::Data::BpeTokenizer> tokenizer = loadTokenizer( configuration.family, models );

        if ( options.recall && !measured.empty() )
        {
            if ( configuration.family == Family::Llama )
            {
                // Llama's tool turn lives in Chat and the server, not in Mila/Src (ContextProfile.md 4.3).
                profile[ "recall_skipped" ] = "Llama's tool grammar is not in Mila/Src yet";
            }
            else
            {
                const auto arm_started = std::chrono::steady_clock::now();

                profile[ "recall" ] = Json::array();

                for ( const dim_t band : measured )
                {
                    const std::size_t slot = profile[ "recall" ].size();
                    profile[ "recall" ].push_back( Json::object() );

                    profile[ "recall" ][ slot ] = recallAtBand<TNetwork, TModel>( configuration, weights, options, *tokenizer,
                        band, [ & ]( const Json& band_so_far )
                        {
                            profile[ "recall" ][ slot ] = band_so_far;
                            checkpoint();
                        } );
                }

                const Json& recall = profile[ "recall" ];

                const std::optional<dim_t> reliable = reliableDepthOnRecall( recall );
                profile[ "reliable_depth" ][ "band" ] = reliable ? Json( *reliable ) : Json( nullptr );
                profile[ "reliable_depth" ][ "on" ] = "recall";

                run_record[ "recall_seconds" ] = secondsSince( arm_started );
            }
        }

        if ( options.loss && !measured.empty() )
        {
            const auto arm_started = std::chrono::steady_clock::now();

            profile[ "loss" ] = lossByBand<TNetwork, TModel>( configuration, weights, options, *tokenizer, measured.back() );
            run_record[ "loss_seconds" ] = secondsSince( arm_started );
            checkpoint();
        }

        run_record[ "seconds" ] = secondsSince( started );
        run_record[ "complete" ] = true;

        // The run's own record last, apart from the scores: two runs of one configuration differ only here.
        profile[ "run" ] = run_record;

        writeText( output, profile.dump( 2 ) + "\n" );

        fs::path markdown = output;
        markdown.replace_extension( ".md" );
        writeText( markdown, markdownOf( profile ) );

        std::cout << std::format( "wrote {} and {}\n", output.string(), markdown.string() );
    }

    export int run( int argc, char** argv )
    {
        try
        {
            const std::string_view command = argc > 1 ? argv[ 1 ] : "";

            if ( command == "list" )
            {
                for ( const Configuration& configuration : kConfigurations )
                {
                    std::cout << std::format( "{:<24} {:<18} cache {:<18} {}\n", configuration.name,
                        configuration.weight_format, configuration.kv_cache, configuration.weights );
                }

                return 0;
            }

            if ( command != "run" )
            {
                printUsage();

                return command.empty() || command == "--help" ? 0 : 2;
            }

            const Options options = parseRunOptions( argc, argv );
            const Configuration& configuration = configurationNamed( options.configuration );

            // Warnings shown: a decode step that stops replaying says so here, and its timing changes with it.
            Mila::initialize( 0, std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning ) );

            dispatchConfiguration( configuration.name, [&]<typename TNetwork, typename TModel>()
            {
                profileConfiguration<TNetwork, TModel>( configuration, options );
            } );

            return 0;
        } catch ( const std::exception& error )
        {
            std::cerr << "ContextProfile: " << error.what() << "\n";

            return 1;
        }
    }
}
