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
import Dnn.Samplers.SpeculativeSampler;
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

    // ====================================================================
    // Several rows at once, and a speculative round's acceptance (Gemma4Mtp.md 4.8)
    // ====================================================================

    namespace
    {
        constexpr dim_t kRows = 5;
        constexpr dim_t kRowsVocab = 4099;

        template<TensorDataType TPrecision>
        struct RowsFixture
        {
            std::unique_ptr<IExecutionContext> context{ createExecutionContext( Device::Metal( 0 ) ) };

            [[nodiscard]] Op<TPrecision> op( dim_t rows, float softcap = 0.0f, dim_t vocab = kRowsVocab, std::vector<std::int32_t> suppressed = {} ) const
            {
                return Op<TPrecision>( context.get(), SamplingConfig{}
                    .withVocabularySize( vocab )
                    .withFinalLogitSoftcap( softcap )
                    .withMaximumRows( rows )
                    .withSuppressedTokens( suppressed ) );
            }

            /// `rows` rows of random logits, each its own seed, laid out as a verify's [1, rows, vocab].
            [[nodiscard]] static Tensor<TPrecision, MetalMemoryResource> logitRows( dim_t rows, dim_t vocab, std::uint32_t seed )
            {
                std::vector<float> values;

                for ( dim_t row = 0; row < rows; ++row )
                {
                    const std::vector<float> row_values = randomLogits( static_cast<std::size_t>( vocab ), seed + static_cast<std::uint32_t>( row ) );
                    values.insert( values.end(), row_values.begin(), row_values.end() );
                }

                return metalTensor<TPrecision>( values, shape_t{ 1, rows, vocab } );
            }

            [[nodiscard]] std::vector<std::int32_t> read( const TokenTensor& tokens, dim_t count ) const
            {
                context->synchronize();

                Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), tokens.shape() );
                copy( tokens, host );

                return std::vector<std::int32_t>( host.data(), host.data() + count );
            }

            [[nodiscard]] static TokenTensor tokens( const std::vector<std::int32_t>& values )
            {
                TokenTensor tensor( Device::Metal( 0 ), shape_t{ static_cast<dim_t>( values.size() ) } );
                fill( tensor, std::span<const std::int32_t>( values ) );

                return tensor;
            }

            [[nodiscard]] static Flags flags( dim_t vocab, const std::vector<std::int32_t>& flagged )
            {
                Flags tensor( Device::Metal( 0 ), shape_t{ vocab } );
                std::vector<std::int32_t> host( static_cast<std::size_t>( vocab ), 0 );

                for ( const std::int32_t token_id : flagged )
                {
                    host[static_cast<std::size_t>( token_id )] = 1;
                }

                fill( tensor, std::span<const std::int32_t>( host ) );

                return tensor;
            }

            [[nodiscard]] std::vector<std::int32_t> flagged( const Flags& tensor ) const
            {
                context->synchronize();

                Tensor<TensorDataType::INT32, CpuMemoryResource> host( Device::Cpu(), tensor.shape() );
                copy( tensor, host );

                std::vector<std::int32_t> result;

                for ( dim_t i = 0; i < host.size(); ++i )
                {
                    if ( host.data()[i] != 0 )
                    {
                        result.push_back( static_cast<std::int32_t>( i ) );
                    }
                }

                return result;
            }

            /// Each row drawn alone through the one-row entry, on that row's view.
            [[nodiscard]] std::vector<std::int32_t> eachRowAlone( const Op<TPrecision>& sampler, Tensor<TPrecision, MetalMemoryResource>& logits,
                dim_t rows, dim_t vocab, const SamplingParams& params, const std::vector<float>& uniforms,
                const std::vector<std::vector<std::int32_t>>& produced_per_row = {} ) const
            {
                std::vector<std::int32_t> drawn;

                for ( dim_t row = 0; row < rows; ++row )
                {
                    auto view = logits.view( shape_t{ 1, 1, vocab }, row * vocab );
                    TokenTensor token( Device::Metal( 0 ), shape_t{ 1, 1 } );
                    std::optional<Flags> produced;

                    if ( !produced_per_row.empty() )
                    {
                        produced.emplace( flags( vocab, produced_per_row[static_cast<std::size_t>( row )] ) );
                    }

                    sampler.enqueueForwardOnDevice( view, token, params, uniforms[static_cast<std::size_t>( row )], produced ? &*produced : nullptr );
                    drawn.push_back( read( token, 1 )[0] );
                }

                return drawn;
            }
        };

        std::vector<float> uniformsFrom( std::uint32_t seed, dim_t count )
        {
            std::mt19937 engine( seed );
            std::uniform_real_distribution<float> distribution( 0.0f, 1.0f );
            std::vector<float> uniforms( static_cast<std::size_t>( count ) );

            std::ranges::generate( uniforms, [&] { return distribution( engine ); } );

            return uniforms;
        }

        SamplingParams paramsOf( float temperature, int top_k, float top_p, float penalty = 0.0f )
        {
            SamplingParams params;
            params.temperature = temperature;
            params.top_k = top_k;
            params.top_p = top_p;
            params.presence_penalty = penalty;

            return params;
        }

        template<typename TPrecisionConstant>
        class SamplingRowsMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( SamplingRowsMetal, Precisions, PrecisionNames );
    }

    // Each row of a rows call draws the token the one-row entry draws from it with the same uniform, at Google's sampling and the
    // truncations, with Gemma's softcap.
    TYPED_TEST( SamplingRowsMetal, EachRowIsTheOneRowDraw )
    {
        RowsFixture<TypeParam::value> fixture;
        const auto rows_op = fixture.op( kRows, 30.0f );
        const auto one_row_op = fixture.op( 1, 30.0f );
        auto logits = RowsFixture<TypeParam::value>::logitRows( kRows, kRowsVocab, 100 );

        for ( std::uint32_t trial = 0; trial < 4; ++trial )
        {
            const std::vector<float> uniforms = uniformsFrom( 7 + trial, kRows );

            for ( const SamplingParams& params : { paramsOf( 1.0f, 64, 0.95f ), paramsOf( 0.8f, 40, 1.0f ), paramsOf( 0.8f, 0, 0.9f ),
                                                   paramsOf( 1.0f, 0, 1.0f ), paramsOf( 0.0f, 0, 1.0f ) } )
            {
                TokenTensor tokens( Device::Metal( 0 ), shape_t{ kRows } );
                rows_op.enqueueRowsOnDevice( logits, tokens, kRows, params, uniforms );

                EXPECT_EQ( fixture.read( tokens, kRows ), fixture.eachRowAlone( one_row_op, logits, kRows, kRowsVocab, params, uniforms ) )
                    << "temperature " << params.temperature << ", top_k " << params.top_k << ", top_p " << params.top_p << ", trial " << trial;
            }
        }
    }

    TYPED_TEST( SamplingRowsMetal, GreedyRowsAreEachRowsLargest )
    {
        RowsFixture<TypeParam::value> fixture;
        const auto op = fixture.op( 3, 0.0f, 8 );

        const std::array<float, 24> values{ 1, 9, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 9 };
        const auto logits = metalTensor<TypeParam::value>( values, shape_t{ 1, 3, 8 } );

        TokenTensor tokens( Device::Metal( 0 ), shape_t{ 3 } );
        op.enqueueRowsOnDevice( logits, tokens, 3, paramsOf( 0.0f, 0, 1.0f ), std::vector<float>( 3, 0.5f ) );

        EXPECT_EQ( fixture.read( tokens, 3 ), ( std::vector<std::int32_t>{ 1, 0, 7 } ) );
    }

    TYPED_TEST( SamplingRowsMetal, RefusesMoreRowsThanBuiltOrTooFewUniforms )
    {
        RowsFixture<TypeParam::value> fixture;
        const auto op = fixture.op( 2, 0.0f, 8 );
        auto logits = RowsFixture<TypeParam::value>::logitRows( 3, 8, 1 );

        TokenTensor tokens( Device::Metal( 0 ), shape_t{ 3 } );

        EXPECT_THROW( op.enqueueRowsOnDevice( logits, tokens, 3, paramsOf( 1.0f, 0, 1.0f ), std::vector<float>( 3, 0.5f ) ), std::invalid_argument );
        EXPECT_THROW( op.enqueueRowsOnDevice( logits, tokens, 2, paramsOf( 1.0f, 0, 1.0f ), std::vector<float>( 1, 0.5f ) ), std::invalid_argument );
        EXPECT_THROW( static_cast<void>( fixture.op( 9 ) ), std::invalid_argument ) << "nine rows is past what one call draws";
    }

    // With a presence penalty, row r also penalizes the round's drafts 1..r: each row's draw is the one-row draw with those drafts
    // added to the produced tokens, and the rows mark nothing.
    TYPED_TEST( SamplingRowsMetal, APenaltyCountsTheDraftsBeforeEachRow )
    {
        constexpr float kPenalty = 2.0f;

        RowsFixture<TypeParam::value> fixture;
        const auto rows_op = fixture.op( kRows, 30.0f );
        const auto one_row_op = fixture.op( 1, 30.0f );
        auto logits = RowsFixture<TypeParam::value>::logitRows( kRows, kRowsVocab, 700 );

        // Draft r is row r's own unpenalized greedy choice, so counting it moves that row's greedy draw.
        const std::vector<std::int32_t> greedy_draws =
            fixture.eachRowAlone( one_row_op, logits, kRows, kRowsVocab, paramsOf( 0.0f, 0, 1.0f ), std::vector<float>( kRows, 0.5f ) );

        std::vector<std::int32_t> round = greedy_draws;
        round[0] = 1234;

        const TokenTensor round_tokens = RowsFixture<TypeParam::value>::tokens( round );
        const std::vector<std::int32_t> base = { 5, 77, 1234 };
        Flags produced = RowsFixture<TypeParam::value>::flags( kRowsVocab, base );

        for ( const SamplingParams& params : { paramsOf( 0.0f, 0, 1.0f, kPenalty ), paramsOf( 1.0f, 64, 0.95f, kPenalty ) } )
        {
            const std::vector<float> uniforms = uniformsFrom( 21, kRows );

            TokenTensor drawn( Device::Metal( 0 ), shape_t{ kRows } );
            rows_op.enqueueRowsOnDevice( logits, drawn, kRows, params, uniforms, &produced, &round_tokens );

            std::vector<std::vector<std::int32_t>> produced_per_row;

            for ( dim_t row = 0; row < kRows; ++row )
            {
                std::vector<std::int32_t> flagged = base;
                flagged.insert( flagged.end(), round.begin() + 1, round.begin() + 1 + row );
                produced_per_row.push_back( std::move( flagged ) );
            }

            const std::vector<std::int32_t> rows = fixture.read( drawn, kRows );

            EXPECT_EQ( rows, fixture.eachRowAlone( one_row_op, logits, kRows, kRowsVocab, params, uniforms, produced_per_row ) )
                << "temperature " << params.temperature;
            EXPECT_EQ( fixture.flagged( produced ), base ) << "rows mark nothing";

            if ( params.temperature == 0.0f )
            {
                for ( dim_t row = 1; row < kRows; ++row )
                {
                    EXPECT_NE( rows[static_cast<std::size_t>( row )], round[static_cast<std::size_t>( row )] ) << "row " << row << " drew the draft it counts";
                }
            }
        }

        // Without the penalty the round's drafts change nothing.
        const std::vector<float> uniforms = uniformsFrom( 21, kRows );
        TokenTensor with_round( Device::Metal( 0 ), shape_t{ kRows } );
        TokenTensor without_round( Device::Metal( 0 ), shape_t{ kRows } );
        rows_op.enqueueRowsOnDevice( logits, with_round, kRows, paramsOf( 1.0f, 64, 0.95f ), uniforms, &produced, &round_tokens );
        rows_op.enqueueRowsOnDevice( logits, without_round, kRows, paramsOf( 1.0f, 64, 0.95f ), uniforms );

        EXPECT_EQ( fixture.read( with_round, kRows ), fixture.read( without_round, kRows ) );
    }

    TYPED_TEST( SamplingRowsMetal, RowsNeverDrawASuppressedToken )
    {
        RowsFixture<TypeParam::value> fixture;
        auto logits = RowsFixture<TypeParam::value>::logitRows( kRows, kRowsVocab, 900 );
        const auto plain = fixture.op( kRows );

        TokenTensor greedy_tokens( Device::Metal( 0 ), shape_t{ kRows } );
        plain.enqueueRowsOnDevice( logits, greedy_tokens, kRows, paramsOf( 0.0f, 0, 1.0f ), std::vector<float>( kRows, 0.5f ) );
        const std::vector<std::int32_t> suppressed = fixture.read( greedy_tokens, kRows );

        const auto suppressing = fixture.op( kRows, 0.0f, kRowsVocab, suppressed );

        for ( const SamplingParams& params : { paramsOf( 0.0f, 0, 1.0f ), paramsOf( 1.0f, 2, 1.0f ) } )
        {
            TokenTensor drawn( Device::Metal( 0 ), shape_t{ kRows } );
            suppressing.enqueueRowsOnDevice( logits, drawn, kRows, params, uniformsFrom( 3, kRows ) );

            for ( const std::int32_t token : fixture.read( drawn, kRows ) )
            {
                EXPECT_EQ( std::ranges::find( suppressed, token ), suppressed.end() ) << "drew suppressed " << token;
            }
        }
    }

    // The walk keeps a draft while the row before it chose it: none, some and all kept, laid out as [m, d_1 .. d_m, next], with the
    // next token in slot 0.
    TYPED_TEST( SamplingRowsMetal, AcceptKeepsDraftsWhileTheRowBeforeChoseThem )
    {
        RowsFixture<TypeParam::value> fixture;
        const auto op = fixture.op( kRows, 0.0f, 8 );

        struct AcceptCase
        {
            std::vector<std::int32_t> tokens;
            std::vector<std::int32_t> chosen;
            std::vector<std::int32_t> expected;
        };

        const AcceptCase cases[] = {
            { { 50, 11, 12, 13, 14 }, { 99, 12, 13, 14, 15 }, { 0, 99 } },
            { { 50, 11, 12, 13, 14 }, { 11, 12, 77, 14, 15 }, { 2, 11, 12, 77 } },
            { { 50, 11, 12, 13, 14 }, { 11, 12, 13, 14, 15 }, { 4, 11, 12, 13, 14, 15 } },
            { { 50, 11, 12, 13, 14 }, { 11, 99, 13, 14, 15 }, { 1, 11, 99 } } };

        for ( const AcceptCase& accept : cases )
        {
            TokenTensor tokens = RowsFixture<TypeParam::value>::tokens( accept.tokens );
            const TokenTensor chosen = RowsFixture<TypeParam::value>::tokens( accept.chosen );
            TokenTensor result( Device::Metal( 0 ), shape_t{ kRows + 1 } );

            op.enqueueAcceptOnDevice( tokens, chosen, kRows, result );

            EXPECT_EQ( fixture.read( result, static_cast<dim_t>( accept.expected.size() ) ), accept.expected );
            EXPECT_EQ( fixture.read( tokens, 1 )[0], accept.expected.back() ) << "slot 0 holds the next token";
        }
    }

    TYPED_TEST( SamplingRowsMetal, AcceptMarksTheKeptTokensProduced )
    {
        RowsFixture<TypeParam::value> fixture;
        const auto op = fixture.op( kRows, 0.0f, 128 );

        TokenTensor tokens = RowsFixture<TypeParam::value>::tokens( { 50, 11, 12, 13, 14 } );
        const TokenTensor chosen = RowsFixture<TypeParam::value>::tokens( { 11, 12, 77, 14, 15 } );
        TokenTensor result( Device::Metal( 0 ), shape_t{ kRows + 1 } );
        Flags produced = RowsFixture<TypeParam::value>::flags( 128, { 50 } );

        op.enqueueAcceptOnDevice( tokens, chosen, kRows, result, &produced );

        EXPECT_EQ( fixture.read( result, 4 ), ( std::vector<std::int32_t>{ 2, 11, 12, 77 } ) );
        EXPECT_EQ( fixture.flagged( produced ), ( std::vector<std::int32_t>{ 11, 12, 50, 77 } ) );
    }

    // The speculative sampler whole, against the host's walk over one-row draws: with the first j drafts the tokens the rows draw and
    // draft j + 1 another, it keeps exactly j and returns row j's draw.
    TYPED_TEST( SamplingRowsMetal, TheSpeculativeSamplerIsTheHostWalkOverOneRowDraws )
    {
        RowsFixture<TypeParam::value> fixture;

        SpeculativeSampler<DeviceType::Metal, TypeParam::value> sampler( fixture.context.get(), SamplingConfig{}
            .withVocabularySize( kRowsVocab )
            .withFinalLogitSoftcap( 30.0f )
            .withMaximumRows( kRows ) );

        const auto one_row_op = fixture.op( 1, 30.0f );
        auto logits = RowsFixture<TypeParam::value>::logitRows( kRows, kRowsVocab, 300 );
        const SamplingParams params = paramsOf( 1.0f, 64, 0.95f );
        const std::vector<float> uniforms = uniformsFrom( 11, kRows );
        const std::vector<std::int32_t> draws = fixture.eachRowAlone( one_row_op, logits, kRows, kRowsVocab, params, uniforms );

        for ( dim_t kept_drafts = 0; kept_drafts < kRows; ++kept_drafts )
        {
            std::vector<std::int32_t> round( static_cast<std::size_t>( kRows ) );
            round[0] = 1234;

            for ( dim_t i = 1; i < kRows; ++i )
            {
                const std::int32_t draw = draws[static_cast<std::size_t>( i - 1 )];
                round[static_cast<std::size_t>( i )] = i <= kept_drafts ? draw : ( draw + 1 ) % static_cast<std::int32_t>( kRowsVocab );
            }

            TokenTensor tokens = RowsFixture<TypeParam::value>::tokens( round );

            sampler.enqueueRound( logits, tokens, kRows, params, uniforms );
            const auto kept = sampler.awaitRound();

            const std::vector<std::int32_t> expected( draws.begin(), draws.begin() + kept_drafts + 1 );

            EXPECT_EQ( std::vector<std::int32_t>( kept.begin(), kept.end() ), expected ) << "kept drafts " << kept_drafts;
            EXPECT_EQ( fixture.read( tokens, 1 )[0], expected.back() );
        }
    }
}
