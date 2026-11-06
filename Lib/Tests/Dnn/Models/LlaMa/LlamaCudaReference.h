/**
 * @file LlamaCudaReference.h
 * @brief Mila's CUDA run of a Llama package, captured to a file, and another device's run compared with it.
 *
 * For a model with no HuggingFace capture: Llama 3.1 8B's FP32 run does not fit the machine that makes them
 * (Notebooks/MetalQuantizedWeights.md section 1.2). The capture holds the Llama 3.2 3B capture's prompt and sequence,
 * CUDA's own greedy continuation, its logits at each step, and its mean negative log-likelihood. Include after
 * Llama32Reference.h.
 */

#pragma once

namespace Mila::Tests::Dnn::Models::Llama32
{
    inline fs::path llama31Fp4WeightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_fp4.safetensors";
    }

    /// Where a package's CUDA capture is written: beside it, named for it.
    inline fs::path cudaCapturePath( const fs::path& weights )
    {
        return weights.parent_path() / ( weights.stem().string() + "_cuda_reference.safetensors" );
    }

    template<typename TValue>
    std::vector<TValue> captureValues( const fs::path& capture, const std::string& name )
    {
        Serialization::WeightsReader reader( capture );
        auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

        std::vector<TValue> values( blob.sizeBytes() / sizeof( TValue ) );
        std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

        return values;
    }

    inline std::int32_t argmaxOf( const std::vector<float>& logits )
    {
        return static_cast<std::int32_t>( std::max_element( logits.begin(), logits.end() ) - logits.begin() );
    }

    /// Run the package on CUDA at BF16 and write what a comparison needs. Returns the capture's path.
    template<typename TWeightPolicy>
    fs::path writeCudaCapture( const fs::path& weights, DeviceId device, int steps = 33 )
    {
        auto network = loadedNetwork<DeviceType::Cuda, TensorDataType::BF16, TWeightPolicy, NoKvCompression>( device, weights );

        const std::vector<std::int32_t> prompt = referenceValues<std::int32_t>( "prompt" );
        const std::vector<std::int32_t> sequence = referenceValues<std::int32_t>( "sequence" );

        std::vector<float> logits;
        std::vector<std::int32_t> greedy;

        std::vector<float> step = Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, prompt ) ) );
        logits.insert( logits.end(), step.begin(), step.end() );

        for ( int s = 1; s < steps; ++s )
        {
            greedy.push_back( argmaxOf( step ) );

            const dim_t position = static_cast<dim_t>( prompt.size() ) + s - 1;

            step = Measurement::hostLogits( *network, network->decode( Measurement::deviceTokens( *network, { greedy.back() } ), position ) );
            logits.insert( logits.end(), step.begin(), step.end() );
        }

        const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, sequence );
        const float mean_negative_log_likelihood = static_cast<float>( -measured.total_log_probability / static_cast<double>( measured.scored_positions ) );

        const dim_t vocab = static_cast<dim_t>( step.size() );
        const fs::path capture = cudaCapturePath( weights );

        Serialization::SafeTensorsWriter writer( capture );
        writer.declareTensor( "prompt", TensorDataType::INT32, shape_t{ static_cast<dim_t>( prompt.size() ) } );
        writer.declareTensor( "greedy_tokens", TensorDataType::INT32, shape_t{ static_cast<dim_t>( greedy.size() ) } );
        writer.declareTensor( "cuda.logits", TensorDataType::FP32, shape_t{ static_cast<dim_t>( steps ), vocab } );
        writer.declareTensor( "sequence", TensorDataType::INT32, shape_t{ static_cast<dim_t>( sequence.size() ) } );
        writer.declareTensor( "cuda.mean_negative_log_likelihood", TensorDataType::FP32, shape_t{ 1 } );
        writer.beginData();
        writer.writeTensorData( "prompt", prompt.data(), prompt.size() * sizeof( std::int32_t ) );
        writer.writeTensorData( "greedy_tokens", greedy.data(), greedy.size() * sizeof( std::int32_t ) );
        writer.writeTensorData( "cuda.logits", logits.data(), logits.size() * sizeof( float ) );
        writer.writeTensorData( "sequence", sequence.data(), sequence.size() * sizeof( std::int32_t ) );
        writer.writeTensorData( "cuda.mean_negative_log_likelihood", &mean_negative_log_likelihood, sizeof( float ) );
        writer.close();

        std::cout << std::format( "[ capture ] {}: {} steps, greedy starts {} {} {}, mean negative log-likelihood {:.6f}\n",
            capture.filename().string(), steps, greedy.at( 0 ), greedy.at( 1 ), greedy.at( 2 ), mean_negative_log_likelihood );

        return capture;
    }

    struct CaptureComparison
    {
        double worst_relative_l2{ 0.0 };
        double mean_relative_l2{ 0.0 };
        std::int64_t non_finite{ 0 };
        std::vector<std::int64_t> argmax_disagreements;
        std::vector<double> largest_error;
        double negative_log_likelihood_difference{ 0.0 };
        GreedyAgreement greedy;
    };

    /// Another device's run of the same package against CUDA's capture: logits teacher-forced on CUDA's continuation, the
    /// sequence's mean negative log-likelihood, and greedy generation through the model's own load.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy>
    CaptureComparison compareWithCudaCapture( const fs::path& weights, DeviceId device )
    {
        const fs::path capture = cudaCapturePath( weights );
        const std::vector<std::int32_t> prompt = captureValues<std::int32_t>( capture, "prompt" );
        const std::vector<std::int32_t> continuation = captureValues<std::int32_t>( capture, "greedy_tokens" );
        const std::vector<float> cuda = captureValues<float>( capture, "cuda.logits" );
        const std::vector<std::int32_t> sequence = captureValues<std::int32_t>( capture, "sequence" );
        const double cuda_mean = captureValues<float>( capture, "cuda.mean_negative_log_likelihood" ).at( 0 );

        const std::string run = runName<TPrecision, NoKvCompression>( weights );
        const std::int64_t steps = static_cast<std::int64_t>( continuation.size() ) + 1;
        const std::int64_t vocab = static_cast<std::int64_t>( cuda.size() ) / steps;

        CaptureComparison result;

        {
            auto network = loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, NoKvCompression>( device, weights );

            std::vector<float> actual = Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, prompt ) ) );

            for ( std::int64_t s = 1; s < steps; ++s )
            {
                const dim_t position = static_cast<dim_t>( prompt.size() ) + s - 1;
                const std::vector<float> step = Measurement::hostLogits( *network,
                    network->decode( Measurement::deviceTokens( *network, { continuation[ static_cast<std::size_t>( s - 1 ) ] } ), position ) );

                actual.insert( actual.end(), step.begin(), step.end() );
            }

            for ( std::int64_t s = 0; s < steps; ++s )
            {
                const float* got = actual.data() + s * vocab;
                const float* want = cuda.data() + s * vocab;
                const StepDistance d = distance( got, want, vocab );

                const auto got_argmax = std::max_element( got, got + vocab ) - got;
                const auto want_argmax = std::max_element( want, want + vocab ) - want;

                if ( got_argmax != want_argmax && static_cast<double>( want[ want_argmax ] ) - want[ got_argmax ] >= d.largest_error )
                {
                    result.argmax_disagreements.push_back( s );
                }

                result.largest_error.push_back( d.largest_error );
                result.worst_relative_l2 = std::max( result.worst_relative_l2, d.relative_l2 );
                result.mean_relative_l2 += d.relative_l2 / static_cast<double>( steps );
                result.non_finite += d.non_finite;
            }

            const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, sequence );
            const double mean = -measured.total_log_probability / static_cast<double>( measured.scored_positions );

            result.negative_log_likelihood_difference = mean - cuda_mean;
        }

        LlamaModelConfig config( kContext );
        config.withWeightQuantization( weightQuantizationOf<TWeightPolicy>() );

        auto model = LlamaModel<TDeviceType, TPrecision>::load( weights, config, device );

        GenerateParams params;
        params.max_new_tokens = static_cast<int>( continuation.size() );
        params.sampling.temperature = 0.0f;
        params.sampling.top_k = 0;

        std::vector<std::int32_t> generated;
        (void)model->generate( prompt, [&]( std::int32_t token ) { generated.push_back( token ); }, params, std::stop_token{} );

        while ( result.greedy.agreed < continuation.size() && result.greedy.agreed < generated.size()
            && generated[ result.greedy.agreed ] == continuation[ result.greedy.agreed ] )
        {
            ++result.greedy.agreed;
        }

        if ( result.greedy.agreed < continuation.size() && result.greedy.agreed < generated.size() )
        {
            const float* want = cuda.data() + static_cast<std::int64_t>( result.greedy.agreed ) * vocab;
            const double gap = static_cast<double>( want[ continuation[ result.greedy.agreed ] ] ) - want[ generated[ result.greedy.agreed ] ];

            result.greedy.diverged_at_a_tie = gap < result.largest_error.at( result.greedy.agreed );
        }
        else if ( result.greedy.agreed < continuation.size() )
        {
            result.greedy.diverged_at_a_tie = false;
        }

        std::cout << std::format( "[ against CUDA ] {} worst {:.3e}, mean {:.3e}, {} argmax disagreements, {} non-finite, "
            "mean negative log-likelihood {:+.3e} from CUDA's {:.6f}, greedy agrees for {} of {}{}\n",
            run, result.worst_relative_l2, result.mean_relative_l2, result.argmax_disagreements.size(), result.non_finite,
            result.negative_log_likelihood_difference, cuda_mean, result.greedy.agreed, continuation.size(),
            result.greedy.agreed == continuation.size() ? "" : result.greedy.diverged_at_a_tie ? ", then diverges at a tie" : ", then diverges" );

        return result;
    }
}
