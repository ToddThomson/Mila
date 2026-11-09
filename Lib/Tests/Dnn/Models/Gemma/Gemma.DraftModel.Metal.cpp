/**
 * @file Gemma.DraftModel.Metal.cpp
 * @brief A Gemma network built with a draft model beside it on Metal (MetalBackend.md section 4.6, Gemma4Mtp.md 4.1), and
 *        greedy generation through it.
 *
 * The seeded tiny Gemma carries a seeded tiny draft model, for what needs no weights: the plan prices exactly what the build
 * allocates, and a draft reads the caches without writing them. The 12B and Google's draft model, where they are on disk, for
 * what does: greedy generation with the draft model gives the tokens it gives without one.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr TensorDataType kPrecision = TensorDataType::FP16;

        using GemmaFp16 = GemmaModel<DeviceType::Metal, kPrecision>;
        using TinyGemma = GemmaTransformer<DeviceType::Metal, kPrecision, Quant::Weight::NoWeightQuant, GemmaFp16::GemmaSlidingKvPolicy>;

        constexpr dim_t kContextLength = 256;
        constexpr dim_t kDraftTokens = 4;

        /// A draft model the tiny Gemma can carry: its head sizes, window and key/value heads are the network's.
        GemmaConfig tinyDrafterConfig( const GemmaConfig& target )
        {
            return GemmaConfig( 128, 2 )
                .withVocabularyLength( Common::kTinyVocabulary )
                .withMaxSequenceLength( target.getMaxSequenceLength() )
                .withNumHeads( 4 )
                .withNumKVHeads( target.getNumKVHeads() )
                .withHeadDim( target.getHeadDim() )
                .withGlobalHeadDim( target.getGlobalHeadDim() )
                .withNumGlobalKVHeads( target.getNumGlobalKVHeads() )
                .withHiddenDimension( 256 )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( target.getWindow() )
                .withSlidingWindowPattern( 2 )
                .withGlobalRotaryDim( target.getGlobalRotaryDim() )
                .withRoPETheta( target.getRoPEThetaLocal() )
                .withGlobalRoPETheta( target.getRoPEThetaGlobal() )
                .withTieWordEmbeddings( true );
        }

        GemmaConfig tinyConfig()
        {
            return Common::tinyGemmaConfig( kContextLength ).withDecodeTokens( kDraftTokens + 1 );
        }

        /// The tiny Gemma, with the tiny draft model when asked, its parameters from the tiny networks' seed.
        std::unique_ptr<TinyGemma> buildTinyGemma( bool with_drafter )
        {
            auto& generator = Core::RandomGenerator::getInstance();
            const unsigned int previous = generator.getSeed();
            generator.setSeed( Common::kTinyDecodeSeed );

            auto network = std::make_unique<TinyGemma>( "tiny", tinyConfig(), Device::Metal( 0 ) );

            if ( with_drafter )
                network->addDrafter( tinyDrafterConfig( network->getConfig() ) );

            network->build( BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, true )
                .withPrefillSize( Common::kTinyPrefillRows ) );
            network->synchronize();

            generator.setSeed( previous );

            return network;
        }

        std::vector<std::int32_t> tokensFrom( std::size_t first, std::size_t count )
        {
            std::vector<std::int32_t> tokens( count );

            for ( std::size_t i = 0; i < count; ++i )
                tokens[ i ] = static_cast<std::int32_t>( ( 7 + 31 * ( first + i ) ) % Common::kTinyVocabulary );

            return tokens;
        }

        fs::path gemmaData( const char* file )
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / file;
        }
    }

    class GemmaDraftModelMetalTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Metal ) == 0 )
            {
                GTEST_SKIP() << "Needs a Metal device";
            }
        }
    };

    TEST_F( GemmaDraftModelMetalTests, ADraftModelRunsAtFp16AndBf16 )
    {
        EXPECT_TRUE( ( GemmaTransformer<DeviceType::Metal, TensorDataType::FP16, Quant::Weight::NoWeightQuant, GemmaFp16::GemmaSlidingKvPolicy>::kDraftModelRuns ) );
        EXPECT_TRUE( ( GemmaTransformer<DeviceType::Metal, TensorDataType::BF16, Quant::Weight::NoWeightQuant,
            GemmaModel<DeviceType::Metal, TensorDataType::BF16>::GemmaSlidingKvPolicy>::kDraftModelRuns ) );
    }

    // Deployment.md 2.1: a selection is priced exactly, category by category.
    TEST_F( GemmaDraftModelMetalTests, PricedAsBuilt )
    {
        const BuildContext priced = BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) )
            .withPrefillSize( Common::kTinyPrefillRows );

        MemoryStats predicted_with;
        MemoryStats predicted_without;

        {
            TinyGemma predictor( "tiny", tinyConfig(), Device::Metal( 0 ) );
            predicted_without = predictor.getRequiredMemory( priced );
        }

        {
            TinyGemma predictor( "tiny", tinyConfig(), Device::Metal( 0 ) );
            predictor.addDrafter( tinyDrafterConfig( predictor.getConfig() ) );
            predicted_with = predictor.getRequiredMemory( priced );
        }

        MemoryStats built;

        {
            TinyGemma network( "tiny", tinyConfig(), Device::Metal( 0 ) );
            network.addDrafter( tinyDrafterConfig( network.getConfig() ) );
            network.build( priced );
            built = network.getMemoryStats();
        }

        EXPECT_EQ( predicted_with.device_parameter_bytes, built.device_parameter_bytes );
        EXPECT_EQ( predicted_with.device_state_bytes, built.device_state_bytes );
        EXPECT_EQ( predicted_with.device_scratch_bytes, built.device_scratch_bytes );
        EXPECT_GT( predicted_with.device_parameter_bytes, predicted_without.device_parameter_bytes );
    }

    // Gemma4Mtp.md 4.3: a draft reads the caches and writes nothing, so the decode after it is the decode without it, bit for
    // bit. Its tokens are a function of the network's state: two drafts from one state agree.
    TEST_F( GemmaDraftModelMetalTests, ADraftLeavesTheCachesAsTheyWere )
    {
        const auto prompt = tokensFrom( 0, 40 );
        const auto next = tokensFrom( 40, 1 );
        const dim_t position = static_cast<dim_t>( prompt.size() );

        auto drafted = buildTinyGemma( true );
        auto plain = buildTinyGemma( true );

        ( void )Measurement::hostLogits( *drafted, drafted->prefill( Measurement::deviceTokens( *drafted, prompt ) ) );
        ( void )Measurement::hostLogits( *plain, plain->prefill( Measurement::deviceTokens( *plain, prompt ) ) );

        std::vector<std::int32_t> slots( kDraftTokens + 1, 0 );
        slots[ 0 ] = next[ 0 ];

        auto first = Measurement::deviceTokens( *drafted, slots );
        auto second = Measurement::deviceTokens( *drafted, slots );
        drafted->draftTokens( first, position, 0 );
        drafted->draftTokens( second, position, 0 );
        drafted->synchronize();

        const auto first_draft = toHost<TensorDataType::INT32>( first );
        const auto second_draft = toHost<TensorDataType::INT32>( second );

        EXPECT_EQ( std::memcmp( first_draft.data(), second_draft.data(), sizeof( std::int32_t ) * slots.size() ), 0 );
        EXPECT_EQ( first_draft.data()[ 0 ], next[ 0 ] ) << "the draft overwrote the known token";

        for ( std::size_t i = 1; i < slots.size(); ++i )
        {
            EXPECT_GE( first_draft.data()[ i ], 0 );
            EXPECT_LT( first_draft.data()[ i ], Common::kTinyVocabulary );
        }

        const auto after_draft = Measurement::hostLogits( *drafted,
            drafted->decode( Measurement::deviceTokens( *drafted, next ), position ) );
        const auto without_draft = Measurement::hostLogits( *plain,
            plain->decode( Measurement::deviceTokens( *plain, next ), position ) );

        EXPECT_EQ( std::memcmp( after_draft.data(), without_draft.data(), after_draft.size() * sizeof( float ) ), 0 );
    }

    TEST_F( GemmaDraftModelMetalTests, ADraftNeedsADraftModel )
    {
        auto network = buildTinyGemma( false );
        ( void )Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, tokensFrom( 0, 16 ) ) ) );

        auto tokens = Measurement::deviceTokens( *network, tokensFrom( 16, 3 ) );

        EXPECT_THROW( network->draftTokens( tokens, 16, 0 ), std::logic_error );
    }

    // Gemma4Mtp.md 5.2 on Metal: greedy generation with the draft model gives the tokens greedy generation gives without it,
    // except where the 12B's top two logits are within the multi-token decode's rounding. The prompt is CUDA's test's.
    TEST_F( GemmaDraftModelMetalTests, Gemma4_12B_GreedyGenerationIsUnchanged )
    {
        const fs::path weights = gemmaData( "gemma4_12b_it_qat_q4_0.safetensors" );
        const fs::path drafter = gemmaData( "gemma4_12b_it_qat_drafter_bf16.bin" );
        const fs::path tokenizer_path = gemmaData( "gemma_tokenizer.bin" );

        if ( !fs::exists( weights ) || !fs::exists( drafter ) || !fs::exists( tokenizer_path ) )
        {
            GTEST_SKIP() << "Needs " << weights.string() << ", " << drafter.string() << " and the Gemma tokenizer";
        }

        auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( tokenizer_path );

        const std::vector<Mila::Dnn::Conversation::Turn> history{ { Mila::Dnn::Conversation::Role::User,
            "Explain how a sliding-window KV cache works in a transformer, why it saves memory at long context, and "
            "what it costs. Use a short example." } };
        const auto encoded = tokenizer->encode( Mila::Dnn::Gemma::formatPrompt( history ) );
        const std::vector<std::int32_t> prompt( encoded.begin(), encoded.end() );

        GenerateParams params;
        params.max_new_tokens = 128;
        params.sampling.temperature = 0.0f;

        const auto generate = [&]( const DeploymentRequest& request )
        {
            auto model = GemmaFp16::load( weights, request );
            std::vector<std::int32_t> tokens;

            const GenerateStatus status = model->generate( prompt, [&]( std::int32_t token ) { tokens.push_back( token ); }, params );
            EXPECT_EQ( status, GenerateStatus::MaxNewTokensReached );

            return tokens;
        };

        const DeploymentRequest plain_request = DeploymentRequest()
            .withWeightQuantization( WeightQuantization::Q4_0 )
            .withContextLength( 4096 );
        const DeploymentRequest drafted_request = DeploymentRequest( plain_request ).withSpeculativeDecode( drafter, kDraftTokens );

        // One 12B in memory at a time.
        const auto plain = generate( plain_request );
        const auto drafted = generate( drafted_request );

        ASSERT_EQ( plain.size(), 128u );

        std::size_t first_difference = 0;

        while ( first_difference < plain.size() && first_difference < drafted.size() && plain[ first_difference ] == drafted[ first_difference ] )
            ++first_difference;

        std::printf( "  greedy with the draft model equal to greedy without it for %zu of %zu tokens\n", first_difference, plain.size() );

        EXPECT_EQ( drafted, plain );
    }
}
