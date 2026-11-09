/**
 * @file DecodeTokens.Metal.cpp
 * @brief A network's decode of several tokens in one call on Metal, against the same tokens decoded one at a time
 *        (MetalBackend.md section 4.6, Gemma4Mtp.md 4.7).
 *
 * The tiny Gemma network has a sliding layer (head size 256, a ring of window 128) and a global one (head size 512). On
 * Metal each token's attention sums run as its one-token decode's do; the Linears' several rows take another kernel than
 * one row's, so a token's logits equal its decode()'s within that rounding, judged against a prefill of the same tokens,
 * a third arithmetic, as on CUDA. A structural error moves a logit by the size of the logits.
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
        constexpr TensorDataType kPrecision = TensorDataType::FP16;

        using TinyGemma = GemmaTransformer<DeviceType::Metal, kPrecision,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Metal, kPrecision>::GemmaSlidingKvPolicy>;
        using TinyRoutedGemma = GemmaTransformer<DeviceType::Metal, kPrecision,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Metal, kPrecision>::GemmaSlidingKvPolicy,
            GemmaFeedForward::Routed>;
        using TinyExperts = MixtureOfExperts<DeviceType::Metal, kPrecision, ActivationType::Gelu>;
        using TinyComposite = CompositeComponent<DeviceType::Metal, kPrecision>;
        using DeviceFp16 = Tensor<kPrecision, MetalMemoryResource>;

        constexpr dim_t kContextLength = 256;
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

        /// The largest difference over the rows' RMS, and how many rows' argmax differ where the reference's top two are
        /// not within that difference.
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

                std::vector<float> sorted = b;
                std::partial_sort( sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>() );
                const bool near_tie = sorted[ 0 ] - sorted[ 1 ] <= 2.0 * largest;

                disagreeing += ( Measurement::argMax( a ) != Measurement::argMax( b ) && !near_tie );
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
                        if ( auto* weights = dynamic_cast<DeviceFp16*>( parameter ) )
                            fill_uniform( *weights, -0.1f, 0.1f, context );
                    }
                }
                else if ( auto composite = std::dynamic_pointer_cast<TinyComposite>( child ) )
                {
                    fillExperts( *composite, context );
                }
            }
        }

        template<typename TNetwork>
        std::unique_ptr<TNetwork> buildTiny( const GemmaConfig& config )
        {
            return Common::buildTinyNetwork<TNetwork>( config, kContextLength, Device::Metal( 0 ) );
        }

        /// The tiny Gemma with a routed feed-forward, 4 of 16 experts a token, its banks filled from a fixed seed.
        std::unique_ptr<TinyRoutedGemma> buildTinyRoutedGemma( const GemmaConfig& dense )
        {
            auto network = buildTiny<TinyRoutedGemma>( GemmaConfig( dense ).withMixtureOfExperts( 16, 4, 128 ) );

            auto& generator = Core::RandomGenerator::getInstance();
            const unsigned int previous = generator.getSeed();
            generator.setSeed( Common::kTinyDecodeSeed + 1 );

            fillExperts( *network, network->getExecutionContext() );
            network->synchronize();

            generator.setSeed( previous );

            return network;
        }
    }

    class DecodeTokensMetalTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Metal ) == 0 )
            {
                GTEST_SKIP() << "Needs a Metal device";
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

        // The yardstick: how far each token's logits from a prefill of the prompt through it lie from the same decodes.
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

        EXPECT_LT( in_call.largest_over_rms, 2.0 * yardstick );
        EXPECT_LT( after.largest_over_rms, 2.0 * yardstick );
        EXPECT_EQ( in_call.disagreeing_argmax, 0 );
        EXPECT_EQ( after.disagreeing_argmax, 0 );
    }

    TEST_F( DecodeTokensMetalTests, Gemma_OneCallMatchesOneTokenDecodes )
    {
        expectOneCallMatchesOneTokenDecodes( [] { return buildTiny<TinyGemma>( config() ); } );
    }

    TEST_F( DecodeTokensMetalTests, RoutedGemma_OneCallMatchesOneTokenDecodes )
    {
        expectOneCallMatchesOneTokenDecodes( [] { return buildTinyRoutedGemma( config() ); } );
    }

    // A verify whose later tokens are rejected, then rewound past them, leaves nothing of them behind: two networks verify runs
    // that agree on the tokens kept and differ on the rest, and every pass after the rewind is equal between them bit for bit. At
    // a prompt of 188 the verify's rows cross the sliding ring's wrap (capacity 128 + 64 - 1 = 191).
    TEST_F( DecodeTokensMetalTests, Gemma_RejectedTokensLeaveNoTraceAfterTheRewind )
    {
        for ( const dim_t prompt_length : { kPrompt, dim_t{ 188 } } )
        {
            for ( const dim_t kept : { dim_t{ 1 }, dim_t{ 3 } } )
            {
                const auto prompt = tokensFrom( 0, static_cast<std::size_t>( prompt_length ) );
                const auto agreed = tokensFrom( prompt_length, static_cast<std::size_t>( kTokens + 4 ) );

                std::vector<std::int32_t> first( agreed.begin(), agreed.begin() + kTokens );
                std::vector<std::int32_t> second( first );

                for ( dim_t t = kept; t < kTokens; ++t )
                    second[ t ] = static_cast<std::int32_t>( ( first[ t ] + 101 ) % Common::kTinyVocabulary );

                auto a = buildTiny<TinyGemma>( config() );
                auto b = buildTiny<TinyGemma>( config() );

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

                const auto tally = [&]( const std::vector<float>& from_a, const std::vector<float>& from_b )
                {
                    differing += std::memcmp( from_a.data(), from_b.data(), from_a.size() * sizeof( float ) ) != 0;
                };

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

                EXPECT_TRUE( kept_rows_equal ) << "prompt " << prompt_length << ", " << kept << " of " << kTokens
                    << " tokens kept: a kept token's logits depend on a token after it";

                EXPECT_EQ( differing, 0u ) << "prompt " << prompt_length << ", " << kept << " of " << kTokens
                    << " tokens kept: a pass after the rewind differs between verifies that differed only in rejected tokens";
            }
        }
    }

    TEST_F( DecodeTokensMetalTests, Gemma_MoreTokensThanTheNetworkWasBuiltForAreRefused )
    {
        auto network = buildTiny<TinyGemma>( config( 3 ) );
        ( void )Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, tokensFrom( 0, 16 ) ) ) );

        EXPECT_THROW( ( void )network->decodeTokens( Measurement::deviceTokens( *network, tokensFrom( 16, 4 ) ), 16 ),
            std::invalid_argument );
        EXPECT_NO_THROW( ( void )network->decodeTokens( Measurement::deviceTokens( *network, tokensFrom( 16, 3 ) ), 16 ) );
    }
}
