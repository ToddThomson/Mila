/**
 * @file DecodeLaunch.Cuda.cpp
 * @brief The most that fewer launches can buy a decode step: the step as launched against the same step replayed
 *        as one CUDA graph.
 *
 * ModelFamilyParity.md 8.4, L5 "After the first two levers". A measurement, not a gate: it needs the published Q4_0
 * weights and never runs in CI. Pin the 16 GB card by UUID.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

import Mila;
import Compute.CudaExecutionContext;

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using LlamaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        using LlamaQ4_0 = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, Quant::KvCache::NoKvCompression>;
        using GemmaQ4_0 = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy>;

        constexpr dim_t kContextLength = 2048;
        constexpr int kPrompt = 16;
        constexpr int kWarmupSteps = 8;
        constexpr int kTimedSteps = 200;

        float elapsedMilliseconds( cudaEvent_t start, cudaEvent_t stop )
        {
            float milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start, stop );

            return milliseconds;
        }

        /**
         * Both arms decode at one fixed position after the same prompt, so they do the same work each step: the cache
         * write lands on the same slot and attention reads the same length. The tokens are not a generation; only the
         * time is.
         */
        template<typename TNetwork, typename TConfig>
        void reportDecodeStepAsOneGraph( const fs::path& weights, const TConfig& config )
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 || !fs::exists( weights ) )
            {
                GTEST_SKIP() << "Needs a CUDA device and " << weights.string();
            }

            auto network = Measurement::buildMeasuredNetwork<TNetwork>(
                weights, config, DeviceId{ DeviceType::Cuda, 0 }, kContextLength );

            auto* context = dynamic_cast<CudaExecutionContext*>( network->getExecutionContext() );
            ASSERT_NE( context, nullptr );
            const cudaStream_t stream = context->getStream();

            std::vector<std::int32_t> prompt( kPrompt );

            for ( int i = 0; i < kPrompt; ++i )
            {
                prompt[ static_cast<std::size_t>( i ) ] = 1000 + 37 * i;
            }

            (void)network->prefill( Measurement::deviceTokens( *network, prompt ) );

            const auto token = Measurement::deviceTokens( *network, { 1234 } );
            dim_t position = kPrompt;

            // Grows whatever scratch the step asks for, so neither arm allocates and the capture records final pointers.
            for ( int step = 0; step < kWarmupSteps; ++step )
            {
                (void)network->decode( token, position++ );
            }

            network->synchronize();

            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            ASSERT_EQ( cudaEventCreate( &start ), cudaSuccess );
            ASSERT_EQ( cudaEventCreate( &stop ), cudaSuccess );

            cudaEventRecord( start, stream );

            for ( int step = 0; step < kTimedSteps; ++step )
            {
                (void)network->decode( token, position );
            }

            cudaEventRecord( stop, stream );
            cudaEventSynchronize( stop );

            const double launched_ms = elapsedMilliseconds( start, stop ) / kTimedSteps;

            cudaGraph_t graph = nullptr;
            std::string capture_failure;

            ASSERT_EQ( cudaStreamBeginCapture( stream, cudaStreamCaptureModeThreadLocal ), cudaSuccess );

            try
            {
                (void)network->decode( token, position );
            }
            catch ( const std::exception& error )
            {
                capture_failure = error.what();
            }

            const cudaError_t ended = cudaStreamEndCapture( stream, &graph );

            if ( !capture_failure.empty() || ended != cudaSuccess )
            {
                cudaGetLastError();

                FAIL() << "the decode step cannot be captured: "
                       << ( capture_failure.empty() ? cudaGetErrorString( ended ) : capture_failure );
            }

            std::size_t nodes = 0;
            cudaGraphGetNodes( graph, nullptr, &nodes );

            cudaGraphExec_t executable = nullptr;
            ASSERT_EQ( cudaGraphInstantiate( &executable, graph, 0 ), cudaSuccess );

            for ( int step = 0; step < kWarmupSteps; ++step )
            {
                cudaGraphLaunch( executable, stream );
            }

            cudaEventRecord( start, stream );

            for ( int step = 0; step < kTimedSteps; ++step )
            {
                cudaGraphLaunch( executable, stream );
            }

            cudaEventRecord( stop, stream );
            cudaEventSynchronize( stop );

            const double graph_ms = elapsedMilliseconds( start, stop ) / kTimedSteps;

            ASSERT_EQ( cudaGetLastError(), cudaSuccess );

            std::cout << std::format(
                "  {}: {} kernels a step; launched {:.3f} ms ({:.1f} tokens/s), one graph {:.3f} ms ({:.1f} tokens/s), "
                "{:.3f} ms a step saved ({:.2f} us a kernel)\n",
                weights.filename().string(), nodes, launched_ms, 1000.0 / launched_ms, graph_ms, 1000.0 / graph_ms,
                launched_ms - graph_ms, 1000.0 * ( launched_ms - graph_ms ) / static_cast<double>( nodes ) ) << std::flush;

            cudaGraphExecDestroy( executable );
            cudaGraphDestroy( graph );
            cudaEventDestroy( start );
            cudaEventDestroy( stop );
        }
    }

    TEST( DecodeLaunchCudaTests, DISABLED_DecodeStepAsOneGraph_LlamaQ4_0 )
    {
        const fs::path weights = fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_q4_0.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Needs " << weights.string();
        }

        Serialization::WeightsReader reader( weights );

        reportDecodeStepAsOneGraph<LlamaQ4_0>( weights, LlamaBf16::configFromMetadata( reader.getWeightsMetadata() ) );
    }

    TEST( DecodeLaunchCudaTests, DISABLED_DecodeStepAsOneGraph_GemmaQ4_0 )
    {
        const fs::path weights = fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_qat_q4_0.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Needs " << weights.string();
        }

        Serialization::WeightsReader reader( weights );

        reportDecodeStepAsOneGraph<GemmaQ4_0>( weights, GemmaBf16::configFromMetadata( reader.getWeightsMetadata() ) );
    }
}
