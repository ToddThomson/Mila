/**
 * @file LlamaModel.Generation.Cuda.cpp
 * @brief LlamaModel::generate on the tiny HuggingFace Llama: device sampling and the decode-ahead loop.
 *
 * ModelFamilyParity.md 8.4, L5. Uses the tiny capture Llama.HuggingFaceReference.Cuda.cpp uses, and skips without it.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <vector>

import Mila;

#include "Common/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models::Llama
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using LlamaFp32 = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>;

        constexpr int64_t kContext = 32;
        constexpr int kGenerated = 12;

        fs::path weightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama_tiny" / "llama_tiny_fp32.bin";
        }

        // Inside the tiny vocabulary of 256, and none of them a Llama 3 stop token.
        std::vector<std::int32_t> promptOf( std::size_t length )
        {
            std::vector<std::int32_t> prompt( length );

            for ( std::size_t i = 0; i < length; ++i )
            {
                prompt[ i ] = static_cast<std::int32_t>( ( 7 * i + 3 ) % 256 );
            }

            return prompt;
        }

        struct Generation
        {
            std::vector<std::int32_t> tokens;
            GenerateStatus status{};
        };

        Generation generate( LlamaFp32& model, const std::vector<std::int32_t>& prompt, const GenerateParams& params )
        {
            Generation result;
            result.status = model.generate( prompt, [&]( std::int32_t token ) { result.tokens.push_back( token ); },
                params, std::stop_token{} );

            return result;
        }

        GenerateParams greedy( std::optional<int> max_new_tokens )
        {
            GenerateParams params;
            params.max_new_tokens = max_new_tokens;
            params.sampling.temperature = 0.0f;
            params.sampling.top_k = 0;

            return params;
        }

        GenerateParams sampled( float top_p )
        {
            GenerateParams params;
            params.max_new_tokens = kGenerated;
            params.sampling.temperature = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.top_p = top_p;

            return params;
        }
    }

    class LlamaModelGenerationCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            if ( !fs::exists( weightsPath() ) )
            {
                GTEST_SKIP() << "tiny Llama capture not present at: " << weightsPath().string();
            }

            model_ = LlamaFp32::load( weightsPath(), LlamaModelConfig( kContext ), Device::Cuda( 0 ) );
        }

        std::unique_ptr<LlamaFp32> model_;
    };

    // The model's pipelined loop -- sampling on the device, the next forward enqueued before the token is read back --
    // picks the tokens a plain prefill-then-argmax-then-decode walk of the same network picks.
    TEST_F( LlamaModelGenerationCudaTests, Greedy_MatchesTheNetworksGreedyContinuation )
    {
        const std::vector<std::int32_t> prompt = promptOf( 12 );

        const Generation generated = generate( *model_, prompt, greedy( kGenerated ) );

        Serialization::WeightsReader weights( weightsPath() );
        const LlamaConfig config = LlamaFp32::configFromMetadata( weights.getWeightsMetadata() );

        LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32> network( "llama", config, Device::Cuda( 0 ) );
        network.build( BuildContext( shape_t{ 1, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
        network.loadParameters( weights );

        const Common::GreedyContinuation reference = Common::greedyContinuationOf( network, prompt, kGenerated, {}, kContext );

        EXPECT_EQ( generated.status, GenerateStatus::MaxNewTokensReached );
        EXPECT_EQ( generated.tokens, reference.tokens );
    }

    TEST_F( LlamaModelGenerationCudaTests, SeedSampler_SameSeedGivesTheSameTokens )
    {
        const std::vector<std::int32_t> prompt = promptOf( 12 );

        model_->seedSampler( 7 );
        const Generation first = generate( *model_, prompt, sampled( 1.0f ) );

        model_->seedSampler( 7 );
        const Generation again = generate( *model_, prompt, sampled( 1.0f ) );

        model_->seedSampler( 8 );
        const Generation other = generate( *model_, prompt, sampled( 1.0f ) );

        ASSERT_EQ( first.tokens.size(), static_cast<std::size_t>( kGenerated ) );
        EXPECT_EQ( first.tokens, again.tokens );
        EXPECT_NE( first.tokens, other.tokens ) << "a different seed drew the same twelve tokens, so the seed is not reaching the sampler";
    }

    // A nucleus smaller than any one token's probability keeps only the most likely token, whatever the temperature.
    TEST_F( LlamaModelGenerationCudaTests, TopP_ANucleusOfOneTokenIsGreedy )
    {
        const std::vector<std::int32_t> prompt = promptOf( 12 );

        const Generation nucleus = generate( *model_, prompt, sampled( 1e-6f ) );
        const Generation argmax = generate( *model_, prompt, greedy( kGenerated ) );

        EXPECT_EQ( nucleus.tokens, argmax.tokens );
    }

    // The token sampled from the last position's logits is still reported; nothing is decoded past the context.
    TEST_F( LlamaModelGenerationCudaTests, ContextBound_ReportsEveryPositionThenOverflows )
    {
        constexpr std::size_t kPrompt = 20;

        const Generation generated = generate( *model_, promptOf( kPrompt ), greedy( std::nullopt ) );

        EXPECT_EQ( generated.status, GenerateStatus::ContextOverflow );
        EXPECT_EQ( generated.tokens.size(), static_cast<std::size_t>( kContext ) - kPrompt + 1 );
    }
}
