/**
 * @file HuggingFaceReference.h
 * @brief A model against a HuggingFace capture of it, for any family, device, precision, weight format and cache format.
 *
 * A family gives its capture, its network and its load as a `TFamily` (Llama32Reference.h, Gemma12bReference.h); the
 * checks here are the gates of Notebooks/MetalFirstModel.md and Notebooks/MetalGemma.md. Include after `import Mila;`.
 */

#pragma once

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models::HuggingFaceReference
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::KvCache;

    namespace fs = std::filesystem;

    template<typename TValue>
    std::vector<TValue> valuesIn( const fs::path& capture, const std::string& name )
    {
        Serialization::WeightsReader reference( capture );
        auto blob = reference.readTensorBlob<CpuMemoryResource>( name );

        std::vector<TValue> values( blob.sizeBytes() / sizeof( TValue ) );
        std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

        return values;
    }

    template<typename TFamily, typename TValue>
    std::vector<TValue> referenceValues( const std::string& name )
    {
        return valuesIn<TValue>( TFamily::referencePath(), name );
    }

    inline std::string precisionName( TensorDataType precision )
    {
        return precision == TensorDataType::FP16 ? "FP16" : precision == TensorDataType::BF16 ? "BF16" : "FP32";
    }

    /// What a result line names: the precision, the weights' file, and the cache when it is FP8.
    template<TensorDataType TPrecision, typename TKvPolicy>
    std::string runName( const fs::path& weights )
    {
        return std::format( "{} {}{}", precisionName( TPrecision ), weights.filename().string(),
            std::is_same_v<TKvPolicy, PerTokenKvFp8<>> ? " fp8-cache" : "" );
    }

    /// Per step of the chat continuation, against HuggingFace's FP32 logits and its BF16 ones.
    struct LogitsResult
    {
        double worst_relative_l2{ 0.0 };
        double mean_relative_l2{ 0.0 };
        double worst_relative_l2_to_bf16{ 0.0 };
        std::int64_t non_finite{ 0 };

        /// Steps whose argmax differs from HuggingFace's where HuggingFace's two are further apart than the step's largest
        /// error: logits off by up to that much anywhere cannot order two tokens nearer than it.
        std::vector<std::int64_t> argmax_disagreements;

        /// Each step's largest error against HuggingFace FP32.
        std::vector<double> largest_error;
    };

    struct StepDistance
    {
        double relative_l2{ 0.0 };
        double largest_error{ 0.0 };
        std::int64_t non_finite{ 0 };
    };

    inline StepDistance distance( const float* got, const float* want, std::int64_t vocab )
    {
        StepDistance result;
        double difference_squares = 0.0;
        double reference_squares = 0.0;

        for ( std::int64_t column = 0; column < vocab; ++column )
        {
            if ( !std::isfinite( got[ column ] ) )
            {
                ++result.non_finite;
                continue;
            }

            const double difference = static_cast<double>( got[ column ] ) - want[ column ];
            result.largest_error = std::max( result.largest_error, std::fabs( difference ) );
            difference_squares += difference * difference;
            reference_squares += static_cast<double>( want[ column ] ) * want[ column ];
        }

        result.relative_l2 = std::sqrt( difference_squares / reference_squares );

        return result;
    }

    /// Each step against HuggingFace FP32 and BF16, with the argmax rule of LogitsResult; `label` names the result lines.
    inline LogitsResult compareSteps( const std::string& label, const std::vector<float>& actual, const std::vector<float>& fp32,
        const std::vector<float>& bf16, std::int64_t vocab )
    {
        const std::int64_t steps = static_cast<std::int64_t>( fp32.size() ) / vocab;

        LogitsResult result;

        for ( std::int64_t step = 0; step < steps; ++step )
        {
            const float* got = actual.data() + step * vocab;
            const float* want = fp32.data() + step * vocab;

            const StepDistance to_fp32 = distance( got, want, vocab );
            const StepDistance to_bf16 = distance( got, bf16.data() + step * vocab, vocab );

            const auto got_argmax = std::max_element( got, got + vocab ) - got;
            const auto want_argmax = std::max_element( want, want + vocab ) - want;
            const double gap = static_cast<double>( want[ want_argmax ] ) - want[ got_argmax ];

            if ( got_argmax != want_argmax && gap >= to_fp32.largest_error )
            {
                result.argmax_disagreements.push_back( step );
            }

            result.largest_error.push_back( to_fp32.largest_error );

            result.worst_relative_l2 = std::max( result.worst_relative_l2, to_fp32.relative_l2 );
            result.worst_relative_l2_to_bf16 = std::max( result.worst_relative_l2_to_bf16, to_bf16.relative_l2 );
            result.mean_relative_l2 += to_fp32.relative_l2 / static_cast<double>( steps );
            result.non_finite += to_fp32.non_finite;

            std::cout << std::format( "[ {} ] step {}: relative L2 {:.3e} to HF FP32, {:.3e} to HF BF16, largest error {:.3e}, argmax {} vs {}\n",
                label, step, to_fp32.relative_l2, to_bf16.relative_l2, to_fp32.largest_error, got_argmax, want_argmax );
        }

        std::cout << std::format( "[ {} ] worst {:.3e}, mean {:.3e}, worst to HF BF16 {:.3e}, {} argmax disagreements, {} non-finite\n",
            label, result.worst_relative_l2, result.mean_relative_l2, result.worst_relative_l2_to_bf16,
            result.argmax_disagreements.size(), result.non_finite );

        return result;
    }

    /// Prefill the chat prompt, then decode HuggingFace's own greedy continuation, teacher-forced.
    /// `position_offset` decodes each token one position away from where it is, which the gate must catch.
    template<typename TFamily, DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
    LogitsResult measureLogits( DeviceId device, dim_t position_offset, const fs::path& weights )
    {
        auto network = TFamily::template loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );

        const std::vector<std::int32_t> prompt = referenceValues<TFamily, std::int32_t>( "prompt" );
        const std::vector<std::int32_t> continuation = referenceValues<TFamily, std::int32_t>( "greedy_tokens" );
        const std::vector<float> fp32 = referenceValues<TFamily, float>( "fp32.logits" );
        const std::vector<float> bf16 = valuesIn<float>( TFamily::bf16ReferencePath(), "bf16.logits" );

        const std::int64_t vocab = TFamily::template capturedConfig<TDeviceType, TPrecision>( weights ).getVocabSize();
        const std::int64_t steps = static_cast<std::int64_t>( fp32.size() ) / vocab;

        std::vector<float> actual;

        auto append = [&]( const auto& logits )
        {
            const std::vector<float> host = Measurement::hostLogits( *network, logits );
            actual.insert( actual.end(), host.begin(), host.end() );
        };

        append( network->prefill( Measurement::deviceTokens( *network, prompt ) ) );

        for ( std::int64_t step = 1; step < steps; ++step )
        {
            const dim_t position = static_cast<dim_t>( prompt.size() ) + step - 1;

            append( network->decode( Measurement::deviceTokens( *network, { continuation[ static_cast<std::size_t>( step - 1 ) ] } ),
                position + position_offset ) );
        }

        return compareSteps( "logits " + runName<TPrecision, TKvPolicy>( weights ), actual, fp32, bf16, vocab );
    }

    /// One block's output over the sequence: its largest magnitude and how many values are not finite.
    struct LayerRange
    {
        double largest{ 0.0 };
        std::int64_t non_finite{ 0 };
    };

    struct SequenceResult
    {
        double mean_negative_log_likelihood{ 0.0 };
        double reference_mean_negative_log_likelihood{ 0.0 };
        std::vector<LayerRange> layers;
    };

    /// The mean negative log-likelihood over the captured sequence, with every block's output range observed.
    template<typename TFamily, DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
    SequenceResult measureSequence( DeviceId device, const fs::path& weights )
    {
        auto network = TFamily::template loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        const std::string run = runName<TPrecision, TKvPolicy>( weights );

        using NetworkType = typename decltype( network )::element_type;
        using TensorType = typename NetworkType::TensorType;

        const std::int64_t layers = TFamily::template capturedConfig<TDeviceType, TPrecision>( weights ).getNumLayers();

        SequenceResult result;
        result.layers.resize( static_cast<std::size_t>( layers ) );

        const std::string block_prefix = network->getName() + ".tf_layer_";

        const std::size_t observed = network->observe( block_prefix + "*", ComputePassMask::inference(),
            [&]( std::string_view path, ComputePass, std::string_view stage, const ITensor& value )
            {
                const auto* typed = dynamic_cast<const TensorType*>( &value );

                // A block's own output, not one of its parts: nothing after the layer number.
                const std::string_view rest = path.substr( block_prefix.size() );

                if ( stage != "output" || typed == nullptr || rest.find( '.' ) != std::string_view::npos )
                {
                    return;
                }

                network->synchronize();

                const auto host = toHost<TensorDataType::FP32>( *typed );
                LayerRange& range = result.layers[ static_cast<std::size_t>( std::stoll( std::string( rest ) ) ) ];

                for ( std::size_t i = 0; i < host.size(); ++i )
                {
                    const float x = host.data()[ i ];

                    if ( !std::isfinite( x ) )
                    {
                        ++range.non_finite;
                    }
                    else
                    {
                        range.largest = std::max( range.largest, static_cast<double>( std::fabs( x ) ) );
                    }
                }
            } );

        if ( observed == 0 )
        {
            throw std::runtime_error( "no block was selected, so no ranges will arrive" );
        }

        const std::vector<std::int32_t> sequence = referenceValues<TFamily, std::int32_t>( "sequence" );
        const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, sequence );

        network->stopObserving();

        const std::vector<float> reference = referenceValues<TFamily, float>( "fp32.next_token_log_probabilities" );
        double reference_total = 0.0;

        for ( const float log_probability : reference )
        {
            reference_total += log_probability;
        }

        result.mean_negative_log_likelihood = -measured.total_log_probability / static_cast<double>( measured.scored_positions );
        result.reference_mean_negative_log_likelihood = -reference_total / static_cast<double>( reference.size() );

        std::cout << std::format( "[ sequence ] {} mean negative log-likelihood {:.6f} against HF FP32 {:.6f}, difference {:+.3e}, {} positions\n",
            run, result.mean_negative_log_likelihood, result.reference_mean_negative_log_likelihood,
            result.mean_negative_log_likelihood - result.reference_mean_negative_log_likelihood, measured.scored_positions );

        for ( std::size_t layer = 0; layer < result.layers.size(); ++layer )
        {
            std::cout << std::format( "[ range ] {} layer {}: largest {:.4g}, {} non-finite\n",
                run, layer, result.layers[ layer ].largest, result.layers[ layer ].non_finite );
        }

        return result;
    }

    /// The log-probabilities of a row of logits after the final softcap, when the model has one (`softcap` > 0).
    inline std::vector<double> logProbabilities( const float* logits, std::int64_t vocab, double softcap )
    {
        std::vector<double> values( static_cast<std::size_t>( vocab ) );
        double largest = -std::numeric_limits<double>::infinity();

        for ( std::int64_t column = 0; column < vocab; ++column )
        {
            const double x = logits[ column ];
            values[ static_cast<std::size_t>( column ) ] = softcap > 0.0 ? std::tanh( x / softcap ) * softcap : x;
            largest = std::max( largest, values[ static_cast<std::size_t>( column ) ] );
        }

        double sum = 0.0;

        for ( const double value : values )
        {
            sum += std::exp( value - largest );
        }

        const double normalizer = largest + std::log( sum );

        for ( double& value : values )
        {
            value -= normalizer;
        }

        return values;
    }

    /// KL( reference || got ) over one row, in nats.
    inline double klDivergence( const float* got, const float* reference, std::int64_t vocab, double softcap )
    {
        const std::vector<double> p = logProbabilities( reference, vocab, softcap );
        const std::vector<double> q = logProbabilities( got, vocab, softcap );
        double kl = 0.0;

        for ( std::size_t column = 0; column < p.size(); ++column )
        {
            kl += std::exp( p[ column ] ) * ( p[ column ] - q[ column ] );
        }

        return kl;
    }

    struct PiecesResult
    {
        double mean_kl{ 0.0 };
        double worst_kl{ 0.0 };
        double mean_kl_of_bf16{ 0.0 };
        std::int64_t non_finite{ 0 };
    };

    /// The captured sequence prefilled in pieces, each piece's last-position logits against HuggingFace FP32's by KL.
    template<typename TFamily, DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
    PiecesResult measurePieces( DeviceId device, const fs::path& weights, dim_t piece_tokens )
    {
        auto network = TFamily::template loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        const std::string run = runName<TPrecision, TKvPolicy>( weights );

        const auto config = TFamily::template capturedConfig<TDeviceType, TPrecision>( weights );
        const std::int64_t vocab = config.getVocabSize();
        const double softcap = TFamily::softcap( config );

        const std::vector<std::int32_t> sequence = referenceValues<TFamily, std::int32_t>( "sequence" );
        const std::vector<float> fp32 = referenceValues<TFamily, float>( "fp32.piece_logits" );
        const std::vector<float> bf16 = valuesIn<float>( TFamily::bf16ReferencePath(), "bf16.piece_logits" );
        const std::int64_t pieces = static_cast<std::int64_t>( fp32.size() ) / vocab;

        PiecesResult result;

        for ( std::int64_t piece = 0; piece < pieces; ++piece )
        {
            // prefillFrom takes the prompt from its start, with the positions before `start_offset` already in the cache.
            const std::vector<std::int32_t> tokens( sequence.begin(), sequence.begin() + ( piece + 1 ) * piece_tokens );
            const std::vector<float> got = Measurement::hostLogits( *network,
                network->prefillFrom( Measurement::deviceTokens( *network, tokens ), static_cast<dim_t>( piece * piece_tokens ) ) );

            const float* want = fp32.data() + piece * vocab;
            result.non_finite += std::count_if( got.begin(), got.end(), []( float x ) { return !std::isfinite( x ); } );

            const double kl = klDivergence( got.data(), want, vocab, softcap );
            const double kl_of_bf16 = klDivergence( bf16.data() + piece * vocab, want, vocab, softcap );

            result.mean_kl += kl / static_cast<double>( pieces );
            result.mean_kl_of_bf16 += kl_of_bf16 / static_cast<double>( pieces );
            result.worst_kl = std::max( result.worst_kl, kl );

            std::cout << std::format( "[ pieces ] {} position {}: KL {:.3e} from HF FP32, HF BF16's {:.3e}\n",
                run, ( piece + 1 ) * piece_tokens - 1, kl, kl_of_bf16 );
        }

        std::cout << std::format( "[ pieces ] {} mean KL {:.3e}, worst {:.3e}, HF BF16's mean {:.3e}, {} non-finite\n",
            run, result.mean_kl, result.worst_kl, result.mean_kl_of_bf16, result.non_finite );

        return result;
    }

    /// Prefill the sequence up to the capture's first edge row, then decode across the sliding window's edge, teacher-forced.
    template<typename TFamily, DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
    LogitsResult measureWindowEdge( DeviceId device, const fs::path& weights, dim_t first_row )
    {
        auto network = TFamily::template loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );

        const std::vector<std::int32_t> sequence = referenceValues<TFamily, std::int32_t>( "sequence" );
        const std::vector<float> fp32 = referenceValues<TFamily, float>( "fp32.window_edge_logits" );
        const std::vector<float> bf16 = valuesIn<float>( TFamily::bf16ReferencePath(), "bf16.window_edge_logits" );

        const std::int64_t vocab = TFamily::template capturedConfig<TDeviceType, TPrecision>( weights ).getVocabSize();
        const std::int64_t rows = static_cast<std::int64_t>( fp32.size() ) / vocab;

        // The prefill's last position is the first edge row; each decode then gives the next.
        const std::vector<std::int32_t> prefix( sequence.begin(), sequence.begin() + first_row + 1 );
        std::vector<float> actual = Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, prefix ) ) );

        for ( std::int64_t row = 1; row < rows; ++row )
        {
            const dim_t position = first_row + static_cast<dim_t>( row );
            const std::vector<float> host = Measurement::hostLogits( *network,
                network->decode( Measurement::deviceTokens( *network, { sequence[ static_cast<std::size_t>( position ) ] } ), position ) );
            actual.insert( actual.end(), host.begin(), host.end() );
        }

        return compareSteps( "window edge " + runName<TPrecision, TKvPolicy>( weights ), actual, fp32, bf16, vocab );
    }

    /// How many of HuggingFace's greedy tokens a generation reproduces before a divergence, and whether that divergence is
    /// at a step where HuggingFace's two tokens are closer than the largest error the teacher-forced run measured there.
    struct GreedyAgreement
    {
        std::size_t agreed{ 0 };
        bool diverged_at_a_tie{ true };
    };

    template<typename TFamily>
    GreedyAgreement greedyAgreement( TensorDataType precision, const std::vector<std::int32_t>& generated,
        const std::vector<double>& largest_error, std::int64_t vocab )
    {
        const std::vector<std::int32_t> reference = referenceValues<TFamily, std::int32_t>( "greedy_tokens" );
        const std::vector<float> fp32 = referenceValues<TFamily, float>( "fp32.logits" );

        GreedyAgreement result;

        while ( result.agreed < reference.size() && result.agreed < generated.size()
            && generated[ result.agreed ] == reference[ result.agreed ] )
        {
            ++result.agreed;
        }

        if ( result.agreed < reference.size() && result.agreed < generated.size() )
        {
            const float* want = fp32.data() + static_cast<std::int64_t>( result.agreed ) * vocab;
            const double gap = static_cast<double>( want[ reference[ result.agreed ] ] ) - want[ generated[ result.agreed ] ];

            result.diverged_at_a_tie = gap < largest_error.at( result.agreed );
        }
        else if ( result.agreed < reference.size() )
        {
            result.diverged_at_a_tie = false;
        }

        std::cout << std::format( "[ greedy ] {} agrees with HF FP32 for {} of {} tokens{}\n", precisionName( precision ),
            result.agreed, reference.size(),
            result.agreed == reference.size() ? "" : result.diverged_at_a_tie ? ", then diverges at a tie" : ", then diverges" );

        return result;
    }
}
