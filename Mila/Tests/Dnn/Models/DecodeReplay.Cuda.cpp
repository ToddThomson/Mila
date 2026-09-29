/**
 * @file DecodeReplay.Cuda.cpp
 * @brief A replayed decode step against the called one, on small seeded networks: DecodeGraph.md gates B2-B5.
 *
 * The long runs, and the published packages, are in DecodeReference.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string_view>
#include <unordered_set>
#include <vector>

import Mila;
import Compute.CudaExecutionContext;

#include "Common/DecodeHarness.h"
#include "Common/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        template<typename TNetwork>
        using Counted = Common::CountedDecodeNetwork<TNetwork>;

        using TinyLlamaBf16 = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16>>;
        using TinyLlamaBf16Fp8Kv = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, Quant::KvCache::PerTokenKvFp8<>>>;
        using TinyLlamaFp32 = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>>;
        using TinyGemmaBf16 = Counted<GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaBf16::GemmaSlidingKvPolicy>>;
        using TinyQwenBf16 = Counted<QwenTransformer<DeviceType::Cuda, TensorDataType::BF16>>;

        constexpr dim_t kContextLength = 512;

        // Past the tiny Gemma's 128-position window, and past a split-count change at 64.
        constexpr dim_t kSteps = 300;

        std::int32_t tokenAt( dim_t position )
        {
            return static_cast<std::int32_t>( ( 7 + 37 * position ) % Common::kTinyVocabulary );
        }

        struct DecodeRun
        {
            std::vector<std::vector<float>> logits;
            bool replayed = false;
            int called_steps = 0;
        };

        /// Teacher-forced: the prompt, then `steps` fixed tokens, so one step's difference cannot move the next.
        template<typename TNetwork>
        std::vector<std::vector<float>> decodeSteps( TNetwork& network, Common::DecodeInput<TNetwork>& input,
            dim_t first_position, dim_t steps )
        {
            std::vector<std::vector<float>> logits;

            for ( dim_t position = first_position; position < first_position + steps; ++position )
            {
                logits.push_back( Common::hostLogits( network, network.decode( input.set( tokenAt( position ) ), position ) ) );
            }

            return logits;
        }

        template<typename TNetwork, typename TConfig>
        DecodeRun runDecode( const TConfig& config, bool replay )
        {
            auto network = Common::buildTinyNetwork<TNetwork>( config, kContextLength );
            network->setDecodeReplay( replay );

            (void)Common::hostLogits( *network, network->prefill( Common::deviceTokens( *network, { tokenAt( 0 ) } ) ) );

            Common::DecodeInput<TNetwork> input( *network );

            DecodeRun run;
            run.logits = decodeSteps( *network, input, 1, kSteps );
            run.replayed = network->isDecodeReplayed();
            run.called_steps = network->calledDecodeSteps();

            return run;
        }

        std::size_t differingSteps( const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b )
        {
            std::size_t differing = 0;

            for ( std::size_t i = 0; i < a.size() && i < b.size(); ++i )
            {
                if ( a[ i ].size() != b[ i ].size()
                    || std::memcmp( a[ i ].data(), b[ i ].data(), a[ i ].size() * sizeof( float ) ) != 0 )
                {
                    ++differing;
                }
            }

            return differing + ( a.size() > b.size() ? a.size() - b.size() : b.size() - a.size() );
        }

        /// B2: replay on equals replay off at every step, bit for bit, and the replayed run really replayed.
        template<typename TNetwork, typename TConfig>
        void expectReplayEqualsCalled( const TConfig& config )
        {
            const DecodeRun called = runDecode<TNetwork>( config, false );
            const DecodeRun replayed = runDecode<TNetwork>( config, true );

            EXPECT_EQ( differingSteps( called.logits, replayed.logits ), 0u );
            EXPECT_TRUE( replayed.replayed ) << "the self-check turned replay off";
            EXPECT_EQ( replayed.called_steps, 3 ) << "priming, recording and the self-check are the only called steps";
            EXPECT_EQ( called.called_steps, static_cast<int>( kSteps ) );
        }

        /**
         * A decode step that breaks the rule replay depends on: it passes the position to the device as a launch
         * argument (a memset's value), which a recording freezes at the step it was recorded at.
         */
        class FrozenPositionLlama : public TinyLlamaBf16
        {
        public:
            using TinyLlamaBf16::TinyLlamaBf16;

        protected:
            TensorType& onDecode( const TokenIndexType& input, dim_t position ) override
            {
                TensorType& logits = TinyLlamaBf16::onDecode( input, position );

                auto* context = dynamic_cast<CudaExecutionContext*>( this->getExecutionContext() );
                cudaMemsetAsync( logits.rawData(), static_cast<int>( position & 0x3F ), 1, context->getStream() );

                return logits;
            }
        };
    }

    class DecodeReplayCudaTests : public ::testing::Test
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

    TEST_F( DecodeReplayCudaTests, OffByDefaultOnADirectlyBuiltNetwork )
    {
        auto network = Common::buildTinyNetwork<TinyLlamaBf16>( Common::tinyLlamaConfig( kContextLength ), kContextLength );

        EXPECT_FALSE( network->isDecodeReplayed() );
    }

    TEST_F( DecodeReplayCudaTests, ReplayEqualsCalled_LlamaBf16 )
    {
        expectReplayEqualsCalled<TinyLlamaBf16>( Common::tinyLlamaConfig( kContextLength ) );
    }

    TEST_F( DecodeReplayCudaTests, ReplayEqualsCalled_LlamaBf16Fp8KvCache )
    {
        expectReplayEqualsCalled<TinyLlamaBf16Fp8Kv>( Common::tinyLlamaConfig( kContextLength ) );
    }

    // FP32 decodes through the cuBLASLt fallback and its decode softmax rather than the fused kernel.
    TEST_F( DecodeReplayCudaTests, ReplayEqualsCalled_LlamaFp32 )
    {
        expectReplayEqualsCalled<TinyLlamaFp32>( Common::tinyLlamaConfig( kContextLength ) );
    }

    TEST_F( DecodeReplayCudaTests, ReplayEqualsCalled_GemmaPastItsWindow )
    {
        expectReplayEqualsCalled<TinyGemmaBf16>( Common::tinyGemmaConfig( kContextLength ) );
    }

    // The test below means something only if this fixture's logits depend on the DeltaNet state: the same step taken
    // again from the state it left behind must come out different.
    TEST_F( DecodeReplayCudaTests, QwenFixture_DecodeDependsOnTheRecurrentState )
    {
        auto network = Common::buildTinyNetwork<TinyQwenBf16>( Common::tinyQwenConfig( kContextLength ), kContextLength );

        std::vector<std::int32_t> prompt;

        for ( dim_t position = 0; position < 16; ++position )
            prompt.push_back( tokenAt( position ) );

        (void)Common::hostLogits( *network, network->prefill( Common::deviceTokens( *network, prompt ) ) );

        Common::DecodeInput<TinyQwenBf16> input( *network );

        const auto first = Common::hostLogits( *network, network->decode( input.set( tokenAt( 16 ) ), 16 ) );
        const auto again = Common::hostLogits( *network, network->decode( input.set( tokenAt( 16 ) ), 16 ) );

        EXPECT_NE( differingSteps( { first }, { again } ), 0u );
    }

    // The self-check runs a step twice; the DeltaNet layers' recurrent state is put back between the two.
    TEST_F( DecodeReplayCudaTests, ReplayEqualsCalled_QwenWithRecurrentState )
    {
        expectReplayEqualsCalled<TinyQwenBf16>( Common::tinyQwenConfig( kContextLength ) );
    }

    // B3: the self-check catches a value the recording froze, turns replay off, and the output stays correct.
    TEST_F( DecodeReplayCudaTests, SelfCheckTurnsReplayOffForAFrozenLaunchArgument )
    {
        const DecodeRun called = runDecode<FrozenPositionLlama>( Common::tinyLlamaConfig( kContextLength ), false );
        const DecodeRun replayed = runDecode<FrozenPositionLlama>( Common::tinyLlamaConfig( kContextLength ), true );

        EXPECT_FALSE( replayed.replayed );
        EXPECT_EQ( differingSteps( called.logits, replayed.logits ), 0u );
    }

    // B4: with an observer installed every decode step is called, so every decode publication is delivered.
    TEST_F( DecodeReplayCudaTests, ObservedStepsAreCalledAndPublish )
    {
        const auto publications = []( bool replay, int& called_steps ) {
            auto network = Common::buildTinyNetwork<TinyLlamaBf16>(
                Common::tinyLlamaConfig( kContextLength ), kContextLength );
            network->setDecodeReplay( replay );

            (void)Common::hostLogits( *network, network->prefill( Common::deviceTokens( *network, { tokenAt( 0 ) } ) ) );

            int count = 0;
            const std::size_t matched = network->observe( "*", ComputePassMask::inference(),
                [ & ]( std::string_view, ComputePass pass, std::string_view, const ITensor& ) {
                    count += pass == ComputePass::Decode ? 1 : 0;
                } );

            EXPECT_GT( matched, 0u );

            Common::DecodeInput<TinyLlamaBf16> input( *network );
            (void)decodeSteps( *network, input, 1, 12 );

            network->stopObserving();
            called_steps = network->calledDecodeSteps();

            return count;
        };

        int called_off = 0;
        int called_on = 0;

        const int off = publications( false, called_off );
        const int on = publications( true, called_on );

        EXPECT_GT( off, 0 );
        EXPECT_EQ( on, off );
        EXPECT_EQ( called_on, 12 );
    }

    // B5: a chat continuation after replayed steps rewinds and reuses the prompt prefix exactly as the called path does.
    TEST_F( DecodeReplayCudaTests, PrefixReuseAfterReplayedStepsMatchesTheCalledPath )
    {
        const auto session = []( bool replay ) {
            auto network = Common::buildTinyNetwork<TinyGemmaBf16>(
                Common::tinyGemmaConfig( kContextLength ), kContextLength );
            network->setDecodeReplay( replay );

            std::vector<std::int32_t> first_prompt;
            std::vector<std::int32_t> second_prompt;

            for ( dim_t position = 0; position < 40; ++position )
                first_prompt.push_back( tokenAt( position ) );

            // The second turn shares the first 30 tokens and diverges after them.
            for ( dim_t position = 0; position < 45; ++position )
                second_prompt.push_back( position < 30 ? tokenAt( position ) : tokenAt( position + 1000 ) );

            Common::DecodeInput<TinyGemmaBf16> input( *network );
            std::vector<std::vector<float>> logits;

            logits.push_back( Common::hostLogits( *network, network->prefill( Common::deviceTokens( *network, first_prompt ) ) ) );

            for ( auto& step : decodeSteps( *network, input, 40, 20 ) )
                logits.push_back( std::move( step ) );

            EXPECT_TRUE( network->rewindKvCache( 30 ) );

            logits.push_back( Common::hostLogits(
                *network, network->prefillFrom( Common::deviceTokens( *network, second_prompt ), 30 ) ) );

            for ( auto& step : decodeSteps( *network, input, 45, 20 ) )
                logits.push_back( std::move( step ) );

            EXPECT_EQ( network->isDecodeReplayed(), replay );

            return logits;
        };

        EXPECT_EQ( differingSteps( session( false ), session( true ) ), 0u );
    }
}
