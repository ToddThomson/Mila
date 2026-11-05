/**
 * @file Sampling.Metal.cpp
 * @brief Token sampling and next-token log-probabilities on Metal, against a double-precision reference on the logits as stored.
 *
 * The reference follows the same steps in the host's own order. A uniform that falls within a hair of a token's edge is skipped, since
 * there the two may legitimately differ by a rounding of their running sums.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Dnn.Samplers.SamplingConfig;
import Serialization.Tensor;

#include "../Components/MetalReference.h"

namespace Mila::Tests::Dnn::Samplers
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        // About what 4099 single-precision additions can drift by.
        constexpr double kEdgeSkipped = 5e-5;

        template<TensorDataType TPrecision>
        using Op = typename OperationTraits<OperationType::SamplingOp, DeviceType::Metal, TPrecision>::type;

        using TokenTensor = Tensor<TensorDataType::INT32, MetalMemoryResource>;
        using Flags = Tensor<TensorDataType::UINT8, MetalMemoryResource>;

        struct Settings
        {
            float temperature{ 1.0f };
            int top_k{ 0 };
            float top_p{ 1.0f };
            float softcap{ 0.0f };
            float penalty{ 0.0f };
            std::vector<std::int32_t> suppressed{};
            std::vector<std::int32_t> produced{};
        };

        /// The probabilities the settings give each token, each survivor's in token order.
        std::vector<double> referenceDistribution( std::span<const double> logits, const Settings& settings )
        {
            const auto vocab = logits.size();

            std::vector<double> scaled( vocab );
            std::vector<bool> alive( vocab, true );

            for ( std::size_t i = 0; i < vocab; ++i )
            {
                if ( std::ranges::find( settings.suppressed, static_cast<std::int32_t>( i ) ) != settings.suppressed.end() )
                {
                    alive[i] = false;
                }

                double x = logits[i];

                if ( settings.softcap > 0.0f )
                {
                    x = static_cast<double>( settings.softcap ) * std::tanh( x / static_cast<double>( settings.softcap ) );
                }

                if ( std::ranges::find( settings.produced, static_cast<std::int32_t>( i ) ) != settings.produced.end() )
                {
                    x -= static_cast<double>( settings.penalty );
                }

                scaled[i] = x / static_cast<double>( settings.temperature );
            }

            std::vector<std::size_t> order;

            for ( std::size_t i = 0; i < vocab; ++i )
            {
                if ( alive[i] )
                {
                    order.push_back( i );
                }
            }

            std::ranges::stable_sort( order, [&]( std::size_t a, std::size_t b ) { return scaled[a] > scaled[b]; } );

            // Every token tied with the k-th survives.
            if ( settings.top_k > 0 && static_cast<std::size_t>( settings.top_k ) < order.size() )
            {
                const double kth = scaled[order[static_cast<std::size_t>( settings.top_k ) - 1]];

                for ( const std::size_t i : order )
                {
                    alive[i] = alive[i] && scaled[i] >= kth;
                }
            }

            const double largest = order.empty() ? 0.0 : scaled[order[0]];

            std::vector<double> mass( vocab, 0.0 );
            double total = 0.0;

            for ( std::size_t i = 0; i < vocab; ++i )
            {
                if ( alive[i] )
                {
                    mass[i] = std::exp( scaled[i] - largest );
                    total += mass[i];
                }
            }

            // The smallest set of the most probable tokens whose mass passes top_p of the total.
            if ( settings.top_p < 1.0f )
            {
                double running = 0.0;
                double kept_total = 0.0;
                bool done = false;

                for ( const std::size_t i : order )
                {
                    if ( !alive[i] )
                    {
                        continue;
                    }

                    if ( done )
                    {
                        mass[i] = 0.0;

                        continue;
                    }

                    running += mass[i];
                    kept_total += mass[i];
                    done = running > static_cast<double>( settings.top_p ) * total;
                }

                total = kept_total;
            }

            for ( double& value : mass )
            {
                value /= total;
            }

            return mass;
        }

        template<TensorDataType TPrecision>
        struct Fixture
        {
            std::unique_ptr<IExecutionContext> context{ createExecutionContext( Device::Metal( 0 ) ) };
            std::vector<float> values;
            std::vector<double> stored;
            Tensor<TPrecision, MetalMemoryResource> logits;
            TokenTensor token{ Device::Metal( 0 ), shape_t{ 1, 1 } };

            explicit Fixture( std::vector<float> logit_values )
                : values( std::move( logit_values ) ),
                  stored( asStored<TPrecision>( std::span<const float>( values ) ) ),
                  logits( metalTensor<TPrecision>( values, shape_t{ 1, 1, static_cast<dim_t>( values.size() ) } ) )
            {
            }

            [[nodiscard]] Op<TPrecision> op( const Settings& settings ) const
            {
                return Op<TPrecision>( context.get(), SamplingConfig{}
                    .withVocabularySize( static_cast<std::int64_t>( values.size() ) )
                    .withFinalLogitSoftcap( settings.softcap )
                    .withSuppressedTokens( settings.suppressed ) );
            }

            [[nodiscard]] static SamplingParams params( const Settings& settings )
            {
                SamplingParams params;
                params.temperature = settings.temperature;
                params.top_k = settings.top_k;
                params.top_p = settings.top_p;
                params.presence_penalty = settings.penalty;

                return params;
            }

            std::int32_t draw( const Op<TPrecision>& sampler, const Settings& settings, float r, ITensor* produced = nullptr )
            {
                sampler.forward( logits, token, params( settings ), r, produced );
                context->synchronize();

                Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), shape_t{ 1, 1 } );
                copy( token, host );

                return host.data()[0];
            }

            [[nodiscard]] Flags flagsFor( const std::vector<std::int32_t>& flagged ) const
            {
                Flags flags( Device::Metal( 0 ), shape_t{ static_cast<dim_t>( values.size() ) } );
                std::vector<std::int32_t> host( values.size(), 0 );

                for ( const std::int32_t token_id : flagged )
                {
                    host[static_cast<std::size_t>( token_id )] = 1;
                }

                fill( flags, std::span<const std::int32_t>( host ) );

                return flags;
            }

            /// Draw at many uniforms, each compared to the token the reference's CDF gives there.
            void expectDistribution( const Settings& settings )
            {
                const Op<TPrecision> sampler = op( settings );
                const bool penalized = settings.penalty != 0.0f;
                const std::vector<double> mass = referenceDistribution( stored, settings );

                constexpr int kDraws = 64;
                int compared = 0;

                for ( int d = 0; d < kDraws; ++d )
                {
                    const double r = ( static_cast<double>( d ) + 0.5 ) / kDraws;

                    double cumulative = 0.0;
                    std::int32_t expected = -1;
                    double nearest_edge = 1.0;

                    for ( std::size_t i = 0; i < mass.size(); ++i )
                    {
                        if ( mass[i] <= 0.0 )
                        {
                            continue;
                        }

                        cumulative += mass[i];
                        nearest_edge = std::min( nearest_edge, std::abs( cumulative - r ) );

                        if ( expected < 0 && cumulative >= r )
                        {
                            expected = static_cast<std::int32_t>( i );
                        }
                    }

                    if ( nearest_edge < kEdgeSkipped )
                    {
                        continue;
                    }

                    // A penalized draw marks its token, which is meant to change the next draw, so each starts afresh.
                    Flags produced = flagsFor( settings.produced );

                    EXPECT_EQ( draw( sampler, settings, static_cast<float>( r ), penalized ? &produced : nullptr ), expected ) << "uniform " << r;

                    ++compared;
                }

                EXPECT_GT( compared, kDraws / 2 ) << "Most draws should have been far enough from an edge to compare";
            }
        };

        std::vector<float> randomLogits( std::size_t vocab, std::uint32_t seed, float sigma = 2.0f )
        {
            return randomValues( vocab, seed, sigma );
        }

        template<typename TPrecisionConstant>
        class SamplingMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( SamplingMetal, Precisions, PrecisionNames );

        constexpr std::size_t kVocab = 4099;
    }

    TYPED_TEST( SamplingMetal, GreedyTakesTheLargestAndTheLowestOfATie )
    {
        Fixture<TypeParam::value> fixture( { 1.0f, 3.0f, 2.0f, 3.0f, -1.0f } );
        const Settings settings{ .temperature = 0.0f };
        const auto sampler = fixture.op( settings );

        EXPECT_EQ( fixture.draw( sampler, settings, 0.5f ), 1 );
    }

    TYPED_TEST( SamplingMetal, GreedyAtAWideVocabularyEvenWhenTheLargestIsLast )
    {
        std::vector<float> values = randomLogits( kVocab, 81 );
        values.back() = 100.0f;

        Fixture<TypeParam::value> fixture( std::move( values ) );
        const Settings settings{ .top_k = 1 };
        const auto sampler = fixture.op( settings );

        EXPECT_EQ( fixture.draw( sampler, settings, 0.5f ), static_cast<std::int32_t>( kVocab - 1 ) );
    }

    TYPED_TEST( SamplingMetal, TheDrawFollowsTheDistribution )
    {
        Fixture<TypeParam::value> fixture( randomLogits( kVocab, 82 ) );

        fixture.expectDistribution( Settings{} );
    }

    TYPED_TEST( SamplingMetal, TemperatureFlattensOrSharpens )
    {
        Fixture<TypeParam::value> fixture( randomLogits( kVocab, 83 ) );

        fixture.expectDistribution( Settings{ .temperature = 0.5f } );
        fixture.expectDistribution( Settings{ .temperature = 2.0f } );
    }

    TYPED_TEST( SamplingMetal, TopKLeavesOnlyTheLargest )
    {
        Fixture<TypeParam::value> fixture( randomLogits( kVocab, 84 ) );

        fixture.expectDistribution( Settings{ .top_k = 40 } );
    }

    TYPED_TEST( SamplingMetal, TopKKeepsEveryTokenTiedWithTheKth )
    {
        Fixture<TypeParam::value> fixture( { 1.0f, 4.0f, 2.0f, 2.0f, 2.0f, 0.5f, 3.0f, -1.0f } );

        fixture.expectDistribution( Settings{ .top_k = 3 } );
    }

    TYPED_TEST( SamplingMetal, TopPLeavesTheNucleus )
    {
        // A small vocabulary, where one token more or less changes the distribution.
        Fixture<TypeParam::value> fixture( randomLogits( 257, 85 ) );

        fixture.expectDistribution( Settings{ .top_p = 0.9f } );
        fixture.expectDistribution( Settings{ .top_p = 0.5f } );
    }

    TYPED_TEST( SamplingMetal, TopKThenTopPTogether )
    {
        Fixture<TypeParam::value> fixture( randomLogits( 257, 86 ) );

        fixture.expectDistribution( Settings{ .temperature = 0.7f, .top_k = 50, .top_p = 0.9f } );
    }

    TYPED_TEST( SamplingMetal, ASoftcapReshapesTheDistribution )
    {
        Fixture<TypeParam::value> fixture( randomLogits( 257, 87, 30.0f ) );

        fixture.expectDistribution( Settings{ .softcap = 30.0f } );
    }

    TYPED_TEST( SamplingMetal, ASuppressedTokenIsNeverDrawnEvenWhenItIsTheLargest )
    {
        std::vector<float> values = randomLogits( 257, 88 );
        values[17] = 50.0f;

        Fixture<TypeParam::value> fixture( std::move( values ) );

        const Settings settings{ .suppressed = { 17 } };

        fixture.expectDistribution( settings );

        const Settings greedy{ .temperature = 0.0f, .suppressed = { 17 } };
        const auto sampler = fixture.op( greedy );

        EXPECT_NE( fixture.draw( sampler, greedy, 0.5f ), 17 );
    }

    TYPED_TEST( SamplingMetal, APenaltyLowersAProducedToken )
    {
        std::vector<float> values = randomLogits( 257, 89 );
        values[9] = 20.0f;

        Fixture<TypeParam::value> fixture( std::move( values ) );

        fixture.expectDistribution( Settings{ .penalty = 25.0f, .produced = { 9 } } );

        const Settings greedy{ .temperature = 0.0f, .penalty = 25.0f, .produced = { 9 } };
        const auto sampler = fixture.op( greedy );
        Flags greedy_flags = fixture.flagsFor( greedy.produced );

        EXPECT_NE( fixture.draw( sampler, greedy, 0.5f, &greedy_flags ), 9 );
    }

    TYPED_TEST( SamplingMetal, APenalizedDrawMarksItsTokenAndClearingForgetsIt )
    {
        Fixture<TypeParam::value> fixture( { 0.0f, 9.0f, 0.0f, 0.0f } );

        const Settings settings{ .temperature = 0.0f, .penalty = 1.0f };
        const auto sampler = fixture.op( settings );
        Flags flags = fixture.flagsFor( {} );

        EXPECT_EQ( fixture.draw( sampler, settings, 0.5f, &flags ), 1 );

        Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), flags.shape() );
        copy( flags, host );
        EXPECT_EQ( host.data()[1], 1 );
        EXPECT_EQ( host.data()[0], 0 );

        sampler.enqueueClearProduced( flags );
        fixture.context->synchronize();

        copy( flags, host );
        EXPECT_EQ( host.data()[1], 0 );
    }

    TYPED_TEST( SamplingMetal, AwaitingATokenNeedsOneSampleOutstanding )
    {
        Fixture<TypeParam::value> fixture( { 0.0f, 9.0f, 0.0f, 0.0f } );

        const Settings settings{ .temperature = 0.0f };
        const auto sampler = fixture.op( settings );

        EXPECT_THROW( static_cast<void>( sampler.awaitToken() ), std::logic_error );

        sampler.enqueueForward( fixture.logits, fixture.token, Fixture<TypeParam::value>::params( settings ), 0.5f );

        EXPECT_EQ( sampler.awaitToken(), 1 );
        EXPECT_THROW( static_cast<void>( sampler.awaitToken() ), std::logic_error );
    }

    TYPED_TEST( SamplingMetal, RefusesAPenaltyItCannotApply )
    {
        Fixture<TypeParam::value> fixture( { 0.0f, 1.0f } );

        const Settings settings{ .penalty = 1.0f };
        const auto sampler = fixture.op( settings );

        EXPECT_THROW( sampler.forward( fixture.logits, fixture.token, Fixture<TypeParam::value>::params( settings ), 0.5f ), std::invalid_argument );

        Settings not_finite = settings;
        not_finite.penalty = std::numeric_limits<float>::infinity();
        Flags flags = fixture.flagsFor( {} );

        EXPECT_THROW( sampler.forward( fixture.logits, fixture.token, Fixture<TypeParam::value>::params( not_finite ), 0.5f, &flags ), std::invalid_argument );
    }

    TYPED_TEST( SamplingMetal, ADrawReadsTheLastRowOfItsLogits )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const std::array<float, 8> values{ 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f };
        auto logits = metalTensor<kPrecision>( values, shape_t{ 1, 2, 4 } );

        Fixture<kPrecision> fixture( { 0.0f, 0.0f, 0.0f, 0.0f } );
        const Settings settings{ .temperature = 0.0f };
        const auto sampler = fixture.op( settings );

        sampler.forward( logits, fixture.token, Fixture<kPrecision>::params( settings ), 0.5f );
        fixture.context->synchronize();

        Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), shape_t{ 1, 1 } );
        copy( fixture.token, host );

        EXPECT_EQ( host.data()[0], 2 );
    }

    // ====================================================================
    // Next-token log-probability
    // ====================================================================

    namespace
    {
        constexpr double kLogProbabilityBar = 1e-3;

        template<typename TPrecisionConstant>
        class NextTokenMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( NextTokenMetal, Precisions, PrecisionNames );

        template<TensorDataType TPrecision>
        void expectLogProbabilities( float softcap )
        {
            using LogProbabilityOp = typename OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, TPrecision>::type;

            constexpr dim_t kRows = 5;
            constexpr dim_t kVocabulary = 263;
            constexpr dim_t kFirst = 2;

            const auto context = createExecutionContext( Device::Metal( 0 ) );

            const std::vector<float> values = randomValues( kRows * kVocabulary, 91, 4.0f );
            auto logits = metalTensor<TPrecision>( values, shape_t{ 1, kRows, kVocabulary } );

            std::vector<std::int32_t> ids( kFirst + kRows + 1 );
            std::mt19937 engine( 92 );
            std::ranges::generate( ids, [&] { return static_cast<std::int32_t>( engine() % kVocabulary ); } );

            Tensor<TensorDataType::INT32, MetalMemoryResource> tokens( Device::Metal( 0 ), shape_t{ 1, static_cast<dim_t>( ids.size() ) } );
            fill( tokens, std::span<const std::int32_t>( ids ) );

            LogProbabilityOp op( context.get(), softcap );
            op.begin( kFirst + kRows );
            op.forward( logits, tokens, kFirst, kRows );
            context->synchronize();

            const std::span<const float> result = op.logProbabilities( kFirst + kRows );
            const std::vector<double> stored = asStored<TPrecision>( std::span<const float>( values ) );

            for ( dim_t row = 0; row < kRows; ++row )
            {
                std::vector<float> stored_row( static_cast<std::size_t>( kVocabulary ) );

                for ( dim_t v = 0; v < kVocabulary; ++v )
                {
                    stored_row[static_cast<std::size_t>( v )] = static_cast<float>( stored[static_cast<std::size_t>( row * kVocabulary + v )] );
                }

                const double expected = nextTokenLogProbability(
                    stored_row.data(), kVocabulary, ids[static_cast<std::size_t>( kFirst + row + 1 )], softcap );

                EXPECT_NEAR( result[static_cast<std::size_t>( kFirst + row )], expected, kLogProbabilityBar ) << "row " << row;
            }
        }
    }

    TYPED_TEST( NextTokenMetal, MatchesTheHostReference )
    {
        expectLogProbabilities<TypeParam::value>( 0.0f );
    }

    TYPED_TEST( NextTokenMetal, MatchesTheHostReferenceWithASoftcap )
    {
        expectLogProbabilities<TypeParam::value>( 30.0f );
    }

    TYPED_TEST( NextTokenMetal, ATargetOutsideTheVocabularyIsNotANumber )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using LogProbabilityOp = typename OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, kPrecision>::type;

        const auto context = createExecutionContext( Device::Metal( 0 ) );

        const std::array<float, 4> values{ 0.0f, 1.0f, 2.0f, 3.0f };
        auto logits = metalTensor<kPrecision>( values, shape_t{ 1, 1, 4 } );

        const std::array<std::int32_t, 2> ids{ 0, 9 };
        Tensor<TensorDataType::INT32, MetalMemoryResource> tokens( Device::Metal( 0 ), shape_t{ 1, 2 } );
        fill( tokens, std::span<const std::int32_t>( ids ) );

        LogProbabilityOp op( context.get(), 0.0f );
        op.begin( 1 );
        op.forward( logits, tokens, 0, 1 );
        context->synchronize();

        EXPECT_TRUE( std::isnan( op.logProbabilities( 1 )[0] ) );
    }

    TYPED_TEST( NextTokenMetal, RefusesAPositionPastWhatBeginSized )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using LogProbabilityOp = typename OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, kPrecision>::type;

        const auto context = createExecutionContext( Device::Metal( 0 ) );

        const std::array<float, 4> values{ 0.0f, 1.0f, 2.0f, 3.0f };
        auto logits = metalTensor<kPrecision>( values, shape_t{ 1, 1, 4 } );

        Tensor<TensorDataType::INT32, MetalMemoryResource> tokens( Device::Metal( 0 ), shape_t{ 1, 2 } );

        LogProbabilityOp op( context.get(), 0.0f );
        op.begin( 1 );

        EXPECT_THROW( op.forward( logits, tokens, 1, 1 ), std::out_of_range );
    }

    TEST( SamplingMetalRows, OnlyFp16AndBf16Exist )
    {
        EXPECT_FALSE( ( OperationSupported<OperationType::SamplingOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, TensorDataType::FP32> ) );
    }
}
