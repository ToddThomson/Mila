/**
 * @file Llama32Reference.h
 * @brief Llama 3.2 3B Instruct against HuggingFace's FP32 run on its BF16 weights, for any device, precision, weight
 *        format and cache format.
 *
 * The capture is Lib/Tools/Converters/Llama/hf_llama32_3b_reference.py. CUDA's run of each format sets the bars Metal is
 * held to (Notebooks/MetalFirstModel.md, Notebooks/MetalQuantizedWeights.md). Include after `import Mila;`.
 */

#pragma once

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    namespace fs = std::filesystem;

    inline constexpr dim_t kContext = 1024;

    inline fs::path weightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_bf16.bin";
    }

    /// The published FP4 package's weights, and FP8 weights `ExportArtifact --quantization fp8` wrote from the BF16 ones.
    inline fs::path fp4WeightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_fp4.safetensors";
    }

    inline fs::path fp8WeightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_fp8.safetensors";
    }

    /// The runtime setting a load is given for a weight policy, and for a cache policy.
    template<typename TWeightPolicy>
    constexpr WeightQuantization weightQuantizationOf()
    {
        if constexpr ( std::is_same_v<TWeightPolicy, PerChannelFp8<>> )
        {
            return WeightQuantization::FP8;
        }
        else if constexpr ( std::is_same_v<TWeightPolicy, PerGroupFp4<128>> )
        {
            return WeightQuantization::FP4;
        } else {
            static_assert( std::is_same_v<TWeightPolicy, NoWeightQuant> );

            return WeightQuantization::None;
        }
    }

    template<typename TKvPolicy>
    constexpr KvCacheCompression kvCacheCompressionOf()
    {
        return std::is_same_v<TKvPolicy, PerTokenKvFp8<>> ? KvCacheCompression::FP8 : KvCacheCompression::None;
    }

    inline fs::path referencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_reference.safetensors";
    }

    inline bool captureExists()
    {
        return fs::exists( weightsPath() ) && fs::exists( referencePath() );
    }

    template<typename TValue>
    std::vector<TValue> referenceValues( const std::string& name )
    {
        Serialization::WeightsReader reference( referencePath() );
        auto blob = reference.readTensorBlob<CpuMemoryResource>( name );

        std::vector<TValue> values( blob.sizeBytes() / sizeof( TValue ) );
        std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

        return values;
    }

    inline std::string precisionName( TensorDataType precision )
    {
        return precision == TensorDataType::FP16 ? "FP16" : precision == TensorDataType::BF16 ? "BF16" : "FP32";
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    using Network = LlamaTransformer<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>;

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    LlamaConfig capturedConfig( const fs::path& weights = weightsPath() )
    {
        Serialization::WeightsReader reader( weights );

        return LlamaModel<TDeviceType, TPrecision>::configFromMetadata( reader.getWeightsMetadata() );
    }

    /// What a result line names: the precision, the weights' file, and the cache when it is FP8.
    template<TensorDataType TPrecision, typename TKvPolicy>
    std::string runName( const fs::path& weights )
    {
        return std::format( "{} {}{}", TPrecision == TensorDataType::FP16 ? "FP16" : TPrecision == TensorDataType::BF16 ? "BF16" : "FP32",
            weights.filename().string(), std::is_same_v<TKvPolicy, PerTokenKvFp8<>> ? " fp8-cache" : "" );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    std::unique_ptr<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>> loadedNetwork( DeviceId device, const fs::path& weights )
    {
        return Measurement::buildMeasuredNetwork<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>>(
            weights, capturedConfig<TDeviceType, TPrecision>( weights ), device, kContext );
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

    /// Prefill the chat prompt, then decode HuggingFace's own greedy continuation, teacher-forced.
    /// `position_offset` decodes each token one position away from where it is, which the gate must catch.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    LogitsResult measureLogits( DeviceId device, dim_t position_offset = 0, const fs::path& weights = weightsPath() )
    {
        auto network = loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        const std::string run = runName<TPrecision, TKvPolicy>( weights );

        const std::vector<std::int32_t> prompt = referenceValues<std::int32_t>( "prompt" );
        const std::vector<std::int32_t> continuation = referenceValues<std::int32_t>( "greedy_tokens" );
        const std::vector<float> fp32 = referenceValues<float>( "fp32.logits" );
        const std::vector<float> bf16 = referenceValues<float>( "bf16.logits" );

        const std::int64_t vocab = capturedConfig<TDeviceType, TPrecision>().getVocabSize();
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

            std::cout << std::format( "[ logits ] {} step {}: relative L2 {:.3e} to HF FP32, {:.3e} to HF BF16, largest error {:.3e}, argmax {} vs {}\n",
                run, step, to_fp32.relative_l2, to_bf16.relative_l2, to_fp32.largest_error, got_argmax, want_argmax );
        }

        std::cout << std::format( "[ logits ] {} worst {:.3e}, mean {:.3e}, worst to HF BF16 {:.3e}, {} argmax disagreements, {} non-finite\n",
            run, result.worst_relative_l2, result.mean_relative_l2, result.worst_relative_l2_to_bf16,
            result.argmax_disagreements.size(), result.non_finite );

        return result;
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

    /// The mean negative log-likelihood over the 1024-token wikitext sequence, with every block's output range observed.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    SequenceResult measureSequence( DeviceId device, const fs::path& weights = weightsPath() )
    {
        auto network = loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        const std::string run = runName<TPrecision, TKvPolicy>( weights );

        using TensorType = typename Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>::TensorType;

        const std::int64_t layers = capturedConfig<TDeviceType, TPrecision>().getNumLayers();

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

        const std::vector<std::int32_t> sequence = referenceValues<std::int32_t>( "sequence" );
        const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, sequence );

        network->stopObserving();

        const std::vector<float> reference = referenceValues<float>( "fp32.next_token_log_probabilities" );
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

    /// Greedy generation through the model's own load and loop, which plans the deployment as an application's would.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    std::vector<std::int32_t> generateGreedy( DeviceId device, int max_new_tokens, const fs::path& weights = weightsPath() )
    {
        LlamaModelConfig config( kContext );
        config.withWeightQuantization( weightQuantizationOf<TWeightPolicy>() ).withKvCacheCompression( kvCacheCompressionOf<TKvPolicy>() );

        auto model = LlamaModel<TDeviceType, TPrecision>::load( weights, config, device );

        GenerateParams params;
        params.max_new_tokens = max_new_tokens;
        params.sampling.temperature = 0.0f;
        params.sampling.top_k = 0;

        std::vector<std::int32_t> tokens;

        (void)model->generate( referenceValues<std::int32_t>( "prompt" ),
            [&]( std::int32_t token ) { tokens.push_back( token ); }, params, std::stop_token{} );

        return tokens;
    }

    /// How many of HuggingFace's greedy tokens a generation reproduces before a divergence, and whether that divergence is
    /// at a step where HuggingFace's two tokens are closer than the largest error the teacher-forced run measured there.
    struct GreedyAgreement
    {
        std::size_t agreed{ 0 };
        bool diverged_at_a_tie{ true };
    };

    inline GreedyAgreement greedyAgreement( TensorDataType precision, const std::vector<std::int32_t>& generated,
        const std::vector<double>& largest_error, std::int64_t vocab )
    {
        const std::vector<std::int32_t> reference = referenceValues<std::int32_t>( "greedy_tokens" );
        const std::vector<float> fp32 = referenceValues<float>( "fp32.logits" );

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
