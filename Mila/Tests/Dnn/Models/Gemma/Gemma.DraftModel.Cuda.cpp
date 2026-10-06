/**
 * @file Gemma.DraftModel.Cuda.cpp
 * @brief A Gemma network built with a draft model beside it (Gemma4Mtp.md 4.1), and greedy generation through it.
 *
 * The seeded tiny Gemma carries a seeded tiny draft model, for what needs no weights: the plan prices exactly what the
 * build allocates, and a draft reads the caches without writing them. The 12B and Google's draft model, where they
 * are on disk, for what does: greedy generation with the draft model gives the tokens it gives without one.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

import Mila;

#include "Common/DecodeHarness.h"
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
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using TinyGemma = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaBf16::GemmaSlidingKvPolicy>;
        using CountedTinyGemma = Common::CountedDecodeNetwork<TinyGemma>;

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

        BuildContext tinyBuildContext()
        {
            return BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, true )
                .withPrefillSize( Common::kTinyPrefillRows );
        }

        /// The tiny Gemma, with the tiny draft model when asked, its parameters from the tiny networks' seed.
        template<typename TNetwork = TinyGemma>
        std::unique_ptr<TNetwork> buildTinyGemma( bool with_drafter )
        {
            auto& generator = Core::RandomGenerator::getInstance();
            const unsigned int previous = generator.getSeed();
            generator.setSeed( Common::kTinyDecodeSeed );

            auto network = std::make_unique<TNetwork>( "tiny", tinyConfig(), Device::Cuda( 0 ) );

            if ( with_drafter )
                network->addDrafter( tinyDrafterConfig( network->getConfig() ) );

            network->build( tinyBuildContext() );
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

    class GemmaDraftModelCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "Needs a CUDA device";
            }
        }
    };

    // Deployment.md 2.1: a selection is priced exactly, category by category, and a network without one allocates
    // nothing for it. One network alive at a time, as PlanEqualsBuild.Cuda.cpp prices.
    TEST_F( GemmaDraftModelCudaTests, PricedAsBuilt )
    {
        const BuildContext priced = BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
            .withPrefillSize( Common::kTinyPrefillRows );

        MemoryStats predicted_with;
        MemoryStats predicted_without;

        {
            TinyGemma predictor( "tiny", tinyConfig(), Device::Cuda( 0 ) );
            predicted_without = predictor.getRequiredMemory( priced );
        }

        {
            TinyGemma predictor( "tiny", tinyConfig(), Device::Cuda( 0 ) );
            predictor.addDrafter( tinyDrafterConfig( predictor.getConfig() ) );
            predicted_with = predictor.getRequiredMemory( priced );
        }

        MemoryStats built;

        {
            TinyGemma network( "tiny", tinyConfig(), Device::Cuda( 0 ) );
            network.addDrafter( tinyDrafterConfig( network.getConfig() ) );
            network.build( priced );
            built = network.getMemoryStats();
        }

        EXPECT_EQ( predicted_with.device_parameter_bytes, built.device_parameter_bytes );
        EXPECT_EQ( predicted_with.device_state_bytes, built.device_state_bytes );
        EXPECT_EQ( predicted_with.device_scratch_bytes, built.device_scratch_bytes );
        EXPECT_GT( predicted_with.device_parameter_bytes, predicted_without.device_parameter_bytes );
    }

    // Gemma4Mtp.md 4.3: a draft reads the caches and writes nothing, so the decode after it is the decode without it,
    // bit for bit. Its tokens are a function of the network's state: two drafts from one state agree.
    TEST_F( GemmaDraftModelCudaTests, ADraftLeavesTheCachesAsTheyWere )
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

    // DecodeGraph.md: a draft and its check replay as a decode step does, each with one recording for each number of
    // tokens. Two networks run the same rounds, one calling every pass and one replaying; every draft and every check
    // is equal between them bit for bit. Each round continues from another row of the last check, which the draft takes
    // before its replay, so a row frozen into the recording shows; a shorter round among the full ones is recorded on
    // its own and leaves theirs in place. The rounds cross the sliding ring's wrap (capacity 128 + 64 - 1 = 191), where
    // a check's writes come nearest the keys the next draft reads.
    TEST_F( GemmaDraftModelCudaTests, ReplayedRoundsEqualCalledRounds )
    {
        const dim_t prompt_length = 180;
        const auto prompt = tokensFrom( 0, static_cast<std::size_t>( prompt_length ) );
        const std::vector<dim_t> round_rows{ 5, 5, 5, 5, 3, 5, 5, 5 };

        auto called = buildTinyGemma<CountedTinyGemma>( true );
        auto replayed = buildTinyGemma<CountedTinyGemma>( true );
        replayed->setDecodeReplay( true );

        ( void )Measurement::hostLogits( *called, called->prefill( Measurement::deviceTokens( *called, prompt ) ) );
        ( void )Measurement::hostLogits( *replayed, replayed->prefill( Measurement::deviceTokens( *replayed, prompt ) ) );

        // One token tensor per network and round size, slot 0 rewritten each round: a recording holds its input's address.
        const std::vector<std::int32_t> full( kDraftTokens + 1, 0 );
        const std::vector<std::int32_t> shorter( 3, 0 );
        auto called_full = Measurement::deviceTokens( *called, full );
        auto called_shorter = Measurement::deviceTokens( *called, shorter );
        auto replayed_full = Measurement::deviceTokens( *replayed, full );
        auto replayed_shorter = Measurement::deviceTokens( *replayed, shorter );

        const auto setFirst = []( auto& network, auto& tokens, std::int32_t id )
        {
            Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), shape_t{ 1, 1 } );
            host.data()[ 0 ] = id;
            auto first = tokens.view( shape_t{ 1, 1 }, 0 );
            copy( host, first );
            network.synchronize();
        };

        const auto sameOnHost = []( const auto& a, const auto& b )
        {
            const auto host_a = toHost<TensorDataType::INT32>( a );
            const auto host_b = toHost<TensorDataType::INT32>( b );

            return std::memcmp( host_a.data(), host_b.data(), sizeof( std::int32_t ) * host_a.size() ) == 0;
        };

        dim_t position = prompt_length;
        dim_t hidden_row = 0;
        int differing_drafts = 0;
        int differing_checks = 0;

        for ( std::size_t round = 0; round < round_rows.size(); ++round )
        {
            const dim_t rows = round_rows[ round ];
            auto& called_tokens = rows == kDraftTokens + 1 ? called_full : called_shorter;
            auto& replayed_tokens = rows == kDraftTokens + 1 ? replayed_full : replayed_shorter;
            const std::int32_t first = tokensFrom( 1000 + round, 1 )[ 0 ];

            setFirst( *called, called_tokens, first );
            setFirst( *replayed, replayed_tokens, first );

            called->draftTokens( called_tokens, position, hidden_row );
            replayed->draftTokens( replayed_tokens, position, hidden_row );
            called->synchronize();
            replayed->synchronize();

            differing_drafts += !sameOnHost( called_tokens, replayed_tokens );

            const auto a = Measurement::hostLogits( *called, called->decodeTokens( called_tokens, position ) );
            const auto b = Measurement::hostLogits( *replayed, replayed->decodeTokens( replayed_tokens, position ) );

            differing_checks += std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) != 0;

            // Keep the token at the round's position and some of its drafts, as a round's acceptance would.
            const dim_t kept = static_cast<dim_t>( round ) % rows;
            position += kept + 1;
            hidden_row = kept;

            ASSERT_TRUE( called->rewindKvCache( position ) );
            ASSERT_TRUE( replayed->rewindKvCache( position ) );
        }

        EXPECT_TRUE( replayed->isDecodeReplayed() ) << "a self-check turned replay off";
        EXPECT_EQ( differing_drafts, 0 ) << "a replayed draft differs from the same draft called";
        EXPECT_EQ( differing_checks, 0 ) << "a replayed multi-token decode differs from the same one called";

        // Priming, recording and the self-check for each round size; every other full round replayed.
        EXPECT_EQ( replayed->calledDrafts(), 4 );
        EXPECT_EQ( replayed->calledDecodeTokens(), 4 );
        EXPECT_EQ( called->calledDrafts(), static_cast<int>( round_rows.size() ) );
    }

    TEST_F( GemmaDraftModelCudaTests, ADraftNeedsADraftModel )
    {
        auto network = buildTinyGemma( false );
        ( void )Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, tokensFrom( 0, 16 ) ) ) );

        auto tokens = Measurement::deviceTokens( *network, tokensFrom( 16, 3 ) );

        EXPECT_THROW( network->draftTokens( tokens, 16, 0 ), std::logic_error );
    }

    TEST_F( GemmaDraftModelCudaTests, FamiliesWithoutADraftModelRefuseTheRequest )
    {
        const auto request = DeploymentRequest().withContextLength( 1024 ).withSpeculativeDecode( "drafter.bin", 3 );

        using Llama = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using Qwen = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

        // Refused before the package is read.
        EXPECT_THROW( ( void )Llama::planDeployment( "absent.safetensors", request ), std::invalid_argument );
        EXPECT_THROW( ( void )Qwen::planDeployment( "absent.safetensors", request ), std::invalid_argument );
        EXPECT_THROW( ( void )DeploymentRequest().withSpeculativeDecode( "drafter.bin", 8 ), std::invalid_argument );
    }

    // Gemma4Mtp.md 5.2: greedy generation with the draft model gives the tokens greedy generation gives without it,
    // except where the 12B's top two logits are within the multi-token decode's rounding. Measured 2026-10-05
    // (Tools/Drafting speculate): this prompt's reply first parts at token 402 at K = 4, so 256 tokens are equal.
    // With the draft and its check replayed the tokens are those of the same passes called (DecodeGraph.md 4.7).
    TEST_F( GemmaDraftModelCudaTests, Gemma4_12B_GreedyGenerationIsUnchanged )
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
        params.max_new_tokens = 256;
        params.sampling.temperature = 0.0f;

        const auto generate = [&]( const DeploymentRequest& request, bool replay )
        {
            auto model = GemmaBf16::load( weights, request );
            model->setDecodeReplay( replay );
            std::vector<std::int32_t> tokens;

            const GenerateStatus status = model->generate( prompt, [&]( std::int32_t token ) { tokens.push_back( token ); }, params );
            EXPECT_EQ( status, GenerateStatus::MaxNewTokensReached );
            EXPECT_EQ( model->isDecodeReplayed(), replay ) << "a self-check turned replay off";

            return tokens;
        };

        const DeploymentRequest plain_request = DeploymentRequest()
            .withWeightQuantization( WeightQuantization::Q4_0 )
            .withContextLength( 4096 );
        const DeploymentRequest drafted_request = DeploymentRequest( plain_request ).withSpeculativeDecode( drafter, kDraftTokens );

        // One 12B on the card at a time.
        const auto plain = generate( plain_request, true );
        const auto drafted = generate( drafted_request, true );
        const auto drafted_called = generate( drafted_request, false );

        ASSERT_EQ( plain.size(), 256u );
        EXPECT_EQ( drafted, plain );
        EXPECT_EQ( drafted, drafted_called );
    }
}
