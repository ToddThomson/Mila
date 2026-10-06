/**
 * @file DecodeTokens.Cuda.cpp
 * @brief A network's decode of several tokens in one call against the same tokens decoded one at a time
 *        (Gemma4Mtp.md 4.7, step 4).
 *
 * The tiny Gemma network has a sliding layer (head size 256, a ring of window 128) and a global one (head size 512).
 * Each token's logits in the one call equal its decode()'s within the rounding
 * of two arithmetics -- the multi-row Linear product and the union-band attention sum in other orders than the decode
 * matvecs and one-token attention -- and the caches the call leaves give the next decode() the same logits to the
 * same rounding. A structural error (a token attending past itself, a row read from another token) moves a logit by
 * the size of the logits. The rounding is judged against a prefill of the same tokens, a third arithmetic, since how far
 * a small seeded network carries a summation-order difference is a property of the network.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using TinyGemma = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::GemmaSlidingKvPolicy>;
        using TinyRoutedGemma = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::GemmaSlidingKvPolicy,
            GemmaFeedForward::Routed>;
        using TinyLlama = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16>;
        using TinyExperts = MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu>;
        using TinyComposite = CompositeComponent<DeviceType::Cuda, TensorDataType::BF16>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        constexpr dim_t kContextLength = 256;
        // Short, so each key carries weight: at 150 the diffuse attention of random weights hid a token attending four
        // keys past itself under the rounding budget. The ring past its window is CudaGqaOp.DecodeTokens.Cuda.cpp's.
        constexpr dim_t kPrompt = 12;
        constexpr dim_t kTokens = 5;

        std::vector<std::int32_t> tokensFrom( std::size_t first, std::size_t count )
        {
            std::vector<std::int32_t> tokens( count );

            for ( std::size_t i = 0; i < count; ++i )
                tokens[ i ] = static_cast<std::int32_t>( ( 7 + 31 * ( first + i ) ) % Common::kTinyVocabulary );

            return tokens;
        }

        struct Comparison
        {
            double largest_over_rms;
            int disagreeing_argmax;
        };

        /// Logits rows compared as the test's budget reads them: the largest difference over the rows' RMS, and how
        /// many rows' argmax differ where the reference's top two are not within that difference.
        Comparison compare( const std::vector<float>& produced, const std::vector<float>& expected, dim_t vocabulary )
        {
            double square_sum = 0.0;
            double largest = 0.0;

            for ( std::size_t i = 0; i < expected.size(); ++i )
            {
                square_sum += static_cast<double>( expected[ i ] ) * expected[ i ];
                largest = std::max( largest, std::abs( static_cast<double>( produced[ i ] ) - expected[ i ] ) );
            }

            int disagreeing = 0;

            for ( std::size_t row = 0; row * vocabulary < expected.size(); ++row )
            {
                const std::vector<float> a( produced.begin() + row * vocabulary, produced.begin() + ( row + 1 ) * vocabulary );
                std::vector<float> b( expected.begin() + row * vocabulary, expected.begin() + ( row + 1 ) * vocabulary );

                const std::int32_t argmax_a = Measurement::argMax( a );
                const std::int32_t argmax_b = Measurement::argMax( b );

                std::vector<float> sorted = b;
                std::partial_sort( sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>() );
                const bool near_tie = sorted[ 0 ] - sorted[ 1 ] <= 2.0 * largest;

                disagreeing += ( argmax_a != argmax_b && !near_tie );
            }

            return { largest / std::sqrt( square_sum / expected.size() ), disagreeing };
        }

        // An expert bank initializes to zeros, which would leave the routed branch out of every comparison.
        void fillExperts( const TinyComposite& root, IExecutionContext* context )
        {
            for ( const auto& child : root.getComponents() )
            {
                if ( auto experts = std::dynamic_pointer_cast<TinyExperts>( child ) )
                {
                    for ( ITensor* parameter : experts->getParameters() )
                    {
                        if ( auto* weights = dynamic_cast<DeviceBf16*>( parameter ) )
                            fill_uniform( *weights, -0.1f, 0.1f, context );
                    }
                }
                else if ( auto composite = std::dynamic_pointer_cast<TinyComposite>( child ) )
                {
                    fillExperts( *composite, context );
                }
            }
        }

        /// The tiny Gemma with a routed feed-forward, 4 of 16 experts a token, its banks filled from a fixed seed.
        std::unique_ptr<TinyRoutedGemma> buildTinyRoutedGemma( const GemmaConfig& dense )
        {
            auto network = Common::buildTinyNetwork<TinyRoutedGemma>(
                GemmaConfig( dense ).withMixtureOfExperts( 16, 4, 128 ), kContextLength );

            auto& generator = Core::RandomGenerator::getInstance();
            const unsigned int previous = generator.getSeed();
            generator.setSeed( Common::kTinyDecodeSeed + 1 );

            fillExperts( *network, network->getExecutionContext() );
            network->synchronize();

            generator.setSeed( previous );

            return network;
        }
    }

    class DecodeTokensCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "Needs a CUDA device";
            }
        }

        static GemmaConfig config( dim_t decode_tokens = kTokens )
        {
            return Common::tinyGemmaConfig( kContextLength ).withDecodeTokens( decode_tokens );
        }
    };

    /// `build` returns a fresh network, the same weights every call.
    template<typename TBuild>
    void expectOneCallMatchesOneTokenDecodes( TBuild&& build )
    {
        const auto prompt = tokensFrom( 0, kPrompt );
        const auto next = tokensFrom( kPrompt, kTokens + 1 );
        const dim_t vocabulary = Common::kTinyVocabulary;

        auto one_at_a_time = build();
        auto one_call = build();

        ( void )Measurement::hostLogits( *one_at_a_time, one_at_a_time->prefill( Measurement::deviceTokens( *one_at_a_time, prompt ) ) );
        ( void )Measurement::hostLogits( *one_call, one_call->prefill( Measurement::deviceTokens( *one_call, prompt ) ) );

        std::vector<float> expected;

        for ( dim_t t = 0; t < kTokens; ++t )
        {
            const auto row = Measurement::hostLogits( *one_at_a_time, one_at_a_time->decode(
                Measurement::deviceTokens( *one_at_a_time, { next[ t ] } ), kPrompt + t ) );
            expected.insert( expected.end(), row.begin(), row.end() );
        }

        const std::vector<std::int32_t> run( next.begin(), next.begin() + kTokens );
        auto& logits = one_call->decodeTokens( Measurement::deviceTokens( *one_call, run ), kPrompt );

        EXPECT_EQ( logits.shape(), ( shape_t{ 1, kTokens, vocabulary } ) );
        EXPECT_EQ( one_call->finalNormedHidden().shape()[ 1 ], kTokens );

        const auto produced = Measurement::hostLogits( *one_call, logits );
        ASSERT_EQ( produced.size(), expected.size() );

        const Comparison in_call = compare( produced, expected, vocabulary );

        // The yardstick: how far a third arithmetic -- each token's logits from a prefill of the prompt through it --
        // lies from the same one-token decodes. This small seeded network turns summation-order differences into a
        // few hundredths of its logits' RMS whichever arithmetic changes.
        auto prefilled = build();
        std::vector<float> by_prefill;

        for ( dim_t t = 0; t < kTokens; ++t )
        {
            std::vector<std::int32_t> through( prompt );
            through.insert( through.end(), next.begin(), next.begin() + t + 1 );

            const auto row = Measurement::hostLogits( *prefilled, prefilled->prefill( Measurement::deviceTokens( *prefilled, through ) ) );
            by_prefill.insert( by_prefill.end(), row.begin(), row.end() );
        }

        const double yardstick = compare( by_prefill, expected, vocabulary ).largest_over_rms;

        // The caches the call left: the next token decoded on both.
        const auto after_expected = Measurement::hostLogits( *one_at_a_time, one_at_a_time->decode(
            Measurement::deviceTokens( *one_at_a_time, { next[ kTokens ] } ), kPrompt + kTokens ) );
        const auto after_produced = Measurement::hostLogits( *one_call, one_call->decode(
            Measurement::deviceTokens( *one_call, { next[ kTokens ] } ), kPrompt + kTokens ) );

        const Comparison after = compare( after_produced, after_expected, vocabulary );

        std::printf( "  %lld tokens: largest logit difference %.2e of RMS in the call, %.2e on the next decode; "
            "prefill against decode %.2e\n", static_cast<long long>( kTokens ), in_call.largest_over_rms,
            after.largest_over_rms, yardstick );

        // Measured 2026-10-06 at this prompt: 0 in the call and on the next decode, against 6.3e-2 (dense) and 1.9e-1
        // (routed) for prefill -- a band this short is one split, and these widths' product rounds to the matvec's
        // BF16 bits. Twice the yardstick is the budget; a token attending another's keys, a row taken from another
        // token, or a router skipping its norm (mutation-checked 2026-10-06, 2.1) is the size of the RMS itself.
        EXPECT_LT( in_call.largest_over_rms, 2.0 * yardstick );
        EXPECT_LT( after.largest_over_rms, 2.0 * yardstick );
        EXPECT_EQ( in_call.disagreeing_argmax, 0 );
        EXPECT_EQ( after.disagreeing_argmax, 0 );
    }

    TEST_F( DecodeTokensCudaTests, Gemma_OneCallMatchesOneTokenDecodes )
    {
        expectOneCallMatchesOneTokenDecodes( [] { return Common::buildTinyNetwork<TinyGemma>( config(), kContextLength ); } );
    }

    // Gemma4Mtp.md 4.7, step 5: the routed feed-forward's router and expert bank at several tokens.
    TEST_F( DecodeTokensCudaTests, RoutedGemma_OneCallMatchesOneTokenDecodes )
    {
        expectOneCallMatchesOneTokenDecodes( [] { return buildTinyRoutedGemma( config() ); } );
    }

    // The scratch a multi-token decode requests is reserved at build, so a recorded decode step that captured the
    // shared buffer replays exactly after one: every logit of every step equal to a network that replays nothing.
    TEST_F( DecodeTokensCudaTests, Gemma_ADecodeReplayStaysExactAcrossAMultiTokenDecode )
    {
        const auto prompt = tokensFrom( 0, kPrompt );
        const auto next = tokensFrom( kPrompt, 4 + kTokens + 2 );

        auto called = Common::buildTinyNetwork<TinyGemma>( config(), kContextLength );
        auto replayed = Common::buildTinyNetwork<TinyGemma>( config(), kContextLength );
        replayed->setDecodeReplay( true );

        // One token tensor per network, rewritten in place: a recording holds its input's address.
        auto called_token = Measurement::deviceTokens( *called, { 0 } );
        auto replayed_token = Measurement::deviceTokens( *replayed, { 0 } );

        const auto setToken = [&]( auto& network, auto& device, std::int32_t id )
        {
            Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), shape_t{ 1, 1 } );
            host.data()[ 0 ] = id;
            copy( host, device );
            network.synchronize();
        };

        ( void )Measurement::hostLogits( *called, called->prefill( Measurement::deviceTokens( *called, prompt ) ) );
        ( void )Measurement::hostLogits( *replayed, replayed->prefill( Measurement::deviceTokens( *replayed, prompt ) ) );

        dim_t position = kPrompt;
        std::size_t differing = 0;
        std::size_t next_index = 0;

        const auto step = [&]()
        {
            const std::int32_t id = next[ next_index++ ];
            setToken( *called, called_token, id );
            setToken( *replayed, replayed_token, id );

            const auto a = Measurement::hostLogits( *called, called->decode( called_token, position ) );
            const auto b = Measurement::hostLogits( *replayed, replayed->decode( replayed_token, position ) );

            differing += std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) != 0;
            ++position;
        };

        // Called, recorded, checked, replayed.
        for ( int i = 0; i < 4; ++i )
            step();

        const std::vector<std::int32_t> run( next.begin() + next_index, next.begin() + next_index + kTokens );
        next_index += kTokens;

        const auto a = Measurement::hostLogits( *called, called->decodeTokens( Measurement::deviceTokens( *called, run ), position ) );
        const auto b = Measurement::hostLogits( *replayed, replayed->decodeTokens( Measurement::deviceTokens( *replayed, run ), position ) );
        differing += std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) != 0;
        position += kTokens;

        // Replayed again from the same recording.
        step();
        step();

        EXPECT_TRUE( replayed->isDecodeReplayed() ) << "the self-check turned replay off";
        EXPECT_EQ( differing, 0u ) << "a pass of the replaying network differs from the same pass called";
    }

    // Gemma4Mtp.md 5.2: a verify whose later tokens are rejected, then rewound past them, leaves nothing of them behind.
    // Two networks verify runs that agree on the tokens kept and differ on the rest. The kept tokens' logits are then
    // bit-identical between them, since no token reads a later one, and so is every pass after the rewind, since nothing
    // reads the rejected rows. At a prompt of 188 the verify's rows cross the sliding ring's wrap (capacity
    // 128 + 64 - 1 = 191). Mutation-checked 2026-10-05: a multi-token decode letting each token see the next token's key
    // fails the kept-row check at both prompts.
    TEST_F( DecodeTokensCudaTests, Gemma_RejectedTokensLeaveNoTraceAfterTheRewind )
    {
        for ( const dim_t prompt_length : { kPrompt, dim_t{ 188 } } )
        {
            for ( const dim_t kept : { dim_t{ 1 }, dim_t{ 3 } } )
            {
                const auto prompt = tokensFrom( 0, static_cast<std::size_t>( prompt_length ) );
                const auto agreed = tokensFrom( prompt_length, static_cast<std::size_t>( kTokens + 4 ) );

                // The same kept tokens, then other tokens in the rejected slots.
                std::vector<std::int32_t> first( agreed.begin(), agreed.begin() + kTokens );
                std::vector<std::int32_t> second( first );

                for ( dim_t t = kept; t < kTokens; ++t )
                    second[ t ] = static_cast<std::int32_t>( ( first[ t ] + 101 ) % Common::kTinyVocabulary );

                auto a = Common::buildTinyNetwork<TinyGemma>( config(), kContextLength );
                auto b = Common::buildTinyNetwork<TinyGemma>( config(), kContextLength );

                ( void )Measurement::hostLogits( *a, a->prefill( Measurement::deviceTokens( *a, prompt ) ) );
                ( void )Measurement::hostLogits( *b, b->prefill( Measurement::deviceTokens( *b, prompt ) ) );

                const auto verified_a = Measurement::hostLogits( *a, a->decodeTokens( Measurement::deviceTokens( *a, first ), prompt_length ) );
                const auto verified_b = Measurement::hostLogits( *b, b->decodeTokens( Measurement::deviceTokens( *b, second ), prompt_length ) );

                // The kept tokens' own logits: a token reading a later token's key shows here first.
                const std::size_t kept_values = static_cast<std::size_t>( kept * Common::kTinyVocabulary );
                const bool kept_rows_equal = std::memcmp( verified_a.data(), verified_b.data(), kept_values * sizeof( float ) ) == 0;

                ASSERT_TRUE( a->rewindKvCache( prompt_length + kept ) );
                ASSERT_TRUE( b->rewindKvCache( prompt_length + kept ) );

                std::size_t differing = 0;
                double largest = 0.0;

                const auto tally = [&]( const std::vector<float>& from_a, const std::vector<float>& from_b )
                {
                    differing += std::memcmp( from_a.data(), from_b.data(), from_a.size() * sizeof( float ) ) != 0;

                    for ( std::size_t i = 0; i < from_a.size(); ++i )
                        largest = std::max( largest, std::abs( static_cast<double>( from_a[ i ] ) - from_b[ i ] ) );
                };

                // What the next round would do: a verify from the first position past the kept tokens, then decodes.
                const std::vector<std::int32_t> next( agreed.begin() + kept, agreed.begin() + kept + 3 );
                const dim_t position = prompt_length + kept;

                tally( Measurement::hostLogits( *a, a->decodeTokens( Measurement::deviceTokens( *a, next ), position ) ),
                    Measurement::hostLogits( *b, b->decodeTokens( Measurement::deviceTokens( *b, next ), position ) ) );

                for ( dim_t t = 0; t < 2; ++t )
                {
                    const std::int32_t id = agreed[ kept + 3 + t ];
                    const dim_t at = position + 3 + t;

                    tally( Measurement::hostLogits( *a, a->decode( Measurement::deviceTokens( *a, { id } ), at ) ),
                        Measurement::hostLogits( *b, b->decode( Measurement::deviceTokens( *b, { id } ), at ) ) );
                }

                std::printf( "  prompt %lld, %lld of %lld kept: kept rows %s; after the rewind %zu of 3 passes differ, largest "
                    "logit difference %.3e\n", static_cast<long long>( prompt_length ), static_cast<long long>( kept ),
                    static_cast<long long>( kTokens ), kept_rows_equal ? "equal" : "DIFFER", differing, largest );

                EXPECT_TRUE( kept_rows_equal ) << "prompt " << prompt_length << ", " << kept << " of " << kTokens
                    << " tokens kept: a kept token's logits depend on a token after it";

                EXPECT_EQ( differing, 0u ) << "prompt " << prompt_length << ", " << kept << " of " << kTokens
                    << " tokens kept: a pass after the rewind differs between verifies that differed only in rejected tokens";
            }
        }
    }

    TEST_F( DecodeTokensCudaTests, Gemma_MoreTokensThanTheNetworkWasBuiltForAreRefused )
    {
        auto network = Common::buildTinyNetwork<TinyGemma>( config( 3 ), kContextLength );
        ( void )Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, tokensFrom( 0, 16 ) ) ) );

        EXPECT_THROW( ( void )network->decodeTokens( Measurement::deviceTokens( *network, tokensFrom( 16, 4 ) ), 16 ),
            std::invalid_argument );
        EXPECT_NO_THROW( ( void )network->decodeTokens( Measurement::deviceTokens( *network, tokensFrom( 16, 3 ) ), 16 ) );
    }

    TEST_F( DecodeTokensCudaTests, AFamilyWithoutAMultiTokenDecodeRefuses )
    {
        auto network = Common::buildTinyNetwork<TinyLlama>( Common::tinyLlamaConfig( kContextLength ), kContextLength );
        ( void )Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, tokensFrom( 0, 16 ) ) ) );

        EXPECT_THROW( ( void )network->decodeTokens( Measurement::deviceTokens( *network, tokensFrom( 16, 2 ) ), 16 ),
            std::logic_error );
    }
}
