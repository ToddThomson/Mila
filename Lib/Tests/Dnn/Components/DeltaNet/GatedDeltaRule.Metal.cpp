/**
 * @file GatedDeltaRule.Metal.cpp
 * @brief The gated delta rule on Metal: CUDA's oracle vectors, and a double-precision recurrence at Qwen 3.8's geometry.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        template<typename TPrecisionConstant>
        class GatedDeltaRuleMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( GatedDeltaRuleMetal, Precisions, PrecisionNames );

        constexpr dim_t kBatch = 1;
        constexpr dim_t kSteps = 5;
        constexpr dim_t kKeyHeads = 2;
        constexpr dim_t kValueHeads = 6;
        constexpr dim_t kHeadKeyDim = 4;
        constexpr dim_t kHeadValueDim = 4;
        constexpr dim_t kKeyWidth = kKeyHeads * kHeadKeyDim;
        constexpr dim_t kValueWidth = kValueHeads * kHeadValueDim;

        // CUDA's oracle (GatedDeltaRule.Cuda.cpp), from delta_rule_oracle.py, seed 20260819.
        static constexpr float kQ[] = { 1.05705464f, 1.44421279f, -0.28207618f, 1.34406805f, -0.24894632f, -0.35209745f, 0.56162727f, 1.43236530f, 0.14786828f, 1.46781456f, -1.72898018f, 0.38752419f, -0.36124966f, -0.03852576f, 0.42569640f, -0.58954930f, -0.90253323f, -0.83942890f, -1.24193430f, -0.79828852f, 1.18322778f, -1.98333228f, 0.45120141f, -1.16295266f, 1.11489928f, 2.23451161f, -0.16044247f, -0.85984623f, -0.44878489f, -0.08352322f, 0.18415673f, -0.29725680f, -2.39690495f, -1.02603257f, -0.08978408f, -1.08331931f, -2.41726565f, -0.75296527f, 1.54356372f, -0.38646945f };

        static constexpr float kK[] = { 0.51512873f, 0.52369690f, -0.24314360f, -0.89409006f, 0.02833949f, 0.66830873f, -0.20664166f, 0.45517504f, -1.40313911f, -1.49992824f, 1.65911007f, 0.29586226f, 0.02743289f, -0.47586161f, 0.32010028f, 0.19459344f, -1.59062684f, -0.47742298f, 0.01511013f, -0.37076524f, 0.84459299f, -1.10163069f, -1.18762004f, 1.43959069f, 0.76561141f, -0.16311520f, 1.95577550f, 2.26063395f, -1.01325047f, 0.01468214f, 1.08488727f, -0.99071157f, 0.44533166f, -1.17938018f, -0.36608893f, 0.39010081f, -0.52604550f, -1.77198040f, 0.30997291f, 1.70180941f };

        static constexpr float kV[] = { -0.93039393f, 1.23692751f, -1.16809618f, 0.20896040f, -0.07304917f, 1.19814527f, -0.14460102f, 0.26151353f, -1.66809571f, -0.71040863f, -0.73731244f, 0.64460617f, 0.30209839f, -1.55141127f, -1.08423817f, 0.82472891f, 1.13151109f, -1.00455356f, -1.43640816f, -1.52544117f, 0.18873377f, 0.31342587f, 0.62671959f, 0.97521418f, 0.80179834f, 0.28575599f, 0.22024269f, -1.16358805f, -1.24520075f, 0.80424827f, -0.02859749f, -0.12638266f, -1.83920085f, 0.28713235f, 0.63904852f, -1.73722959f, 1.57390118f, -1.98116386f, 1.65092659f, -0.45082769f, 1.04877841f, 0.26407269f, 0.51059759f, -1.94094002f, 1.63868940f, -0.15680431f, -1.03621089f, -0.84076917f, -0.67262393f, 0.45321041f, -0.14218473f, 0.52326626f, 0.42681995f, -0.76767612f, 0.30088511f, 0.62166840f, -0.85034794f, 1.19379771f, -0.31669697f, 0.02975057f, 1.48981619f, 0.79153877f, 0.27950165f, 0.71432036f, -0.39862317f, -1.61327684f, 0.09710249f, -1.66975486f, -1.17259824f, -1.06897247f, 1.15270126f, 0.05398748f, 0.04653400f, 0.21682912f, 0.54264069f, 0.37949678f, -1.10723448f, -0.47537854f, -0.14109719f, 0.97900337f, -0.03496336f, 0.64319092f, -0.81796724f, -0.36260056f, 0.80267429f, -1.11809242f, -0.49641821f, 0.28194425f, -0.98838836f, -2.18700051f, 0.48741943f, -0.93060321f, 0.46738577f, -0.00165446f, -0.50867623f, -0.34414527f, -1.60398948f, 1.76005507f, -0.66321051f, 1.68116021f, -0.63446242f, -1.14930725f, -0.61889094f, -0.13165191f, -0.66749513f, -0.46859357f, -1.13658631f, 1.39645147f, -0.87644088f, -2.75275087f, -1.71047997f, -0.17475589f, 0.92975849f, 0.34466249f, -0.50480819f, 0.68516016f, -1.05087674f, -0.13240187f, 0.79990798f, 0.17338556f };

        static constexpr float kA[] = { -0.13974093f, -1.46453726f, 0.41820452f, 0.38623208f, -0.00425688f, 0.49934173f, -1.00336421f, 2.20248270f, 0.38398138f, 0.12577532f, -0.59895486f, 1.99951172f, -0.42756030f, 0.95718545f, 0.33197996f, 0.86469173f, 1.85532188f, 1.02893162f, -0.49810359f, -0.48944792f, -2.10644603f, 0.42673030f, -0.99422312f, 0.54020089f, -1.20903742f, -2.66140461f, -1.59693897f, 0.29994959f, -1.18506372f, 0.89684218f };

        static constexpr float kB[] = { 0.26933718f, -1.72954690f, -0.29514074f, 0.34280843f, 0.54095882f, 0.50893855f, 1.81579924f, 2.70643544f, 0.93539447f, 0.88824219f, 0.20952514f, 0.37571675f, -0.11815063f, -0.70197475f, 0.08987319f, 0.78100348f, 0.04038204f, 1.16408288f, -1.51714659f, -0.02998285f, 0.92341101f, -0.37990007f, 0.22534779f, -0.18120773f, -2.19400597f, 0.40527984f, 1.39167345f, 0.73923546f, -0.96731287f, -0.55279768f };

        static constexpr float kALog[] = { -0.00810267f, -0.98629463f, 0.57363623f, 0.11149225f, -0.01736570f, 0.04299238f };

        static constexpr float kDtBias[] = { -1.18187070f, -0.38994655f, 0.58872199f, 1.44299650f, -0.82737941f, 0.43788269f };

        static constexpr float kExpectedOut[] = { -0.01658218f, 0.02204548f, -0.02081873f, 0.00372425f, -0.00034595f, 0.00567428f, -0.00068481f, 0.00123850f, -0.02237873f, -0.00953065f, -0.00989160f, 0.00864786f, 0.01943678f, -0.09981665f, -0.06975909f, 0.05306245f, 0.07867107f, -0.06984404f, -0.09986977f, -0.10606000f, 0.01296692f, 0.02153387f, 0.04305867f, 0.06700194f, -0.25142542f, -0.15567133f, -0.03062780f, 0.41294301f, 0.49246821f, -0.32684016f, 0.01239859f, 0.04793534f, 0.55959272f, -0.08564518f, -0.19210652f, 0.52463973f, 0.02706173f, -0.00753614f, 0.06012353f, -0.02839296f, -0.14398366f, 0.14789380f, 0.21370581f, 0.18526268f, 0.02717519f, -0.00674780f, -0.02651848f, -0.02735599f, -0.15911558f, 0.07290203f, -0.05139699f, 0.14440402f, 0.08180445f, -0.10351754f, 0.03278137f, 0.07064600f, -0.13179077f, 0.19937468f, -0.05775224f, 0.01649516f, 0.10100983f, 0.02097911f, 0.03833544f, 0.03181635f, -0.02967372f, -0.00601296f, 0.09454828f, -0.07086121f, -0.02754128f, -0.07009848f, 0.04036295f, -0.02212281f, -0.12408759f, -0.03062819f, -0.10545297f, 0.15279725f, 0.23442674f, -0.03790233f, -0.01476305f, -0.08318648f, 0.12385163f, -0.18972942f, 0.08998052f, 0.04918348f, 0.13086933f, -0.22088194f, -0.09628344f, 0.04302313f, -0.24255589f, -0.45634222f, 0.14161664f, -0.13278842f, 0.13931236f, 0.03457410f, -0.14659359f, -0.07559959f, -0.04807262f, 0.10688534f, -0.03848106f, -0.03899021f, 0.02318343f, 0.09349942f, 0.06242722f, -0.05112671f, -0.06820969f, 0.05657011f, 0.08681424f, -0.00924931f, -0.08001355f, -0.32376078f, -0.19641736f, -0.01402410f, -0.14274880f, -0.38780269f, 0.09231328f, -0.14018352f, -0.03672060f, -0.00284143f, 0.02102125f, -0.00396330f };

        // CUDA's BF16 bar on these vectors: the inputs' rounding carried through every step.
        constexpr double kOracleTolerance = 2e-2;

        // The FP32 state against a double recurrence on the same stored inputs: float summation over a head's 128 keys.
        constexpr double kStateBar = 1e-4;

        struct Geometry
        {
            dim_t batch;
            dim_t key_heads;
            dim_t value_heads;
            dim_t head_key_dim;
            dim_t head_value_dim;

            [[nodiscard]] dim_t keyWidth() const { return key_heads * head_key_dim; }
            [[nodiscard]] dim_t valueWidth() const { return value_heads * head_value_dim; }
        };

        /// A whole sequence's inputs and the rule's parameters, as floats the precision holds.
        struct Sequence
        {
            Geometry geometry;
            dim_t steps;
            std::vector<float> q, k, v, a, b, a_log, dt_bias;
        };

        /// The recurrence in double precision, the state held between calls: outputs [B, T, value_width], state [B, Hv, dk, dv].
        struct Recurrence
        {
            Geometry geometry;
            std::vector<double> state;

            explicit Recurrence( const Geometry& g )
                : geometry( g ), state( static_cast<std::size_t>( g.batch * g.value_heads * g.head_key_dim * g.head_value_dim ), 0.0 )
            {
            }

            std::vector<double> advance( const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v,
                const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& a_log, const std::vector<double>& dt_bias,
                dim_t steps )
            {
                const Geometry& g = geometry;
                const dim_t dk = g.head_key_dim;
                const dim_t dv = g.head_value_dim;
                const dim_t group = g.value_heads / g.key_heads;

                std::vector<double> out( static_cast<std::size_t>( g.batch * steps * g.valueWidth() ) );

                for ( dim_t batch = 0; batch < g.batch; ++batch )
                {
                    for ( dim_t h = 0; h < g.value_heads; ++h )
                    {
                        double* const s = state.data() + ( batch * g.value_heads + h ) * dk * dv;
                        const dim_t key_head = h / group;

                        for ( dim_t t = 0; t < steps; ++t )
                        {
                            const dim_t qk = ( batch * steps + t ) * g.keyWidth() + key_head * dk;
                            double q_squares = 0.0;
                            double k_squares = 0.0;

                            for ( dim_t d = 0; d < dk; ++d )
                            {
                                q_squares += q[static_cast<std::size_t>( qk + d )] * q[static_cast<std::size_t>( qk + d )];
                                k_squares += k[static_cast<std::size_t>( qk + d )] * k[static_cast<std::size_t>( qk + d )];
                            }

                            const double q_scale = 1.0 / std::sqrt( q_squares + 1e-6 ) / std::sqrt( static_cast<double>( dk ) );
                            const double k_scale = 1.0 / std::sqrt( k_squares + 1e-6 );

                            const std::size_t scalar = static_cast<std::size_t>( ( batch * steps + t ) * g.value_heads + h );
                            const double beta = 1.0 / ( 1.0 + std::exp( -b[scalar] ) );
                            const double x = a[scalar] + dt_bias[static_cast<std::size_t>( h )];
                            const double softplus = x > 20.0 ? x : std::log1p( std::exp( x ) );
                            const double decay = std::exp( -std::exp( a_log[static_cast<std::size_t>( h )] ) * softplus );

                            for ( dim_t j = 0; j < dv; ++j )
                            {
                                const std::size_t vo = static_cast<std::size_t>( ( batch * steps + t ) * g.valueWidth() + h * dv + j );
                                double memory = 0.0;

                                for ( dim_t d = 0; d < dk; ++d )
                                {
                                    s[d * dv + j] *= decay;
                                    memory += s[d * dv + j] * k[static_cast<std::size_t>( qk + d )] * k_scale;
                                }

                                const double delta = ( v[vo] - memory ) * beta;
                                double result = 0.0;

                                for ( dim_t d = 0; d < dk; ++d )
                                {
                                    s[d * dv + j] += k[static_cast<std::size_t>( qk + d )] * k_scale * delta;
                                    result += s[d * dv + j] * q[static_cast<std::size_t>( qk + d )] * q_scale;
                                }

                                out[vo] = result;
                            }
                        }
                    }
                }

                return out;
            }
        };

        template<TensorDataType TPrecision>
        using MetalTensorOf = Tensor<TPrecision, MetalMemoryResource>;

        template<TensorDataType TPrecision>
        std::unique_ptr<GatedDeltaRule<DeviceType::Metal, TPrecision>> builtRule( const Sequence& sequence, dim_t built_steps )
        {
            const Geometry& g = sequence.geometry;
            auto rule = std::make_unique<GatedDeltaRule<DeviceType::Metal, TPrecision>>( "delta_rule",
                GatedDeltaRuleConfig( g.key_heads, g.value_heads, g.head_key_dim, g.head_value_dim ), Device::Metal( 0 ) );
            rule->build( BuildContext( shape_t{ g.batch, built_steps, g.valueWidth() }, RuntimeMode::Inference, false ) );

            const std::vector<ITensor*> parameters = rule->getParameters();
            fill( *static_cast<MetalTensorOf<TPrecision>*>( parameters[0] ), std::span<const float>( sequence.a_log ) );
            fill( *static_cast<MetalTensorOf<TPrecision>*>( parameters[1] ), std::span<const float>( sequence.dt_bias ) );

            return rule;
        }

        /// Steps [first, first + count) of every batch item of a [B, T, width] array.
        std::vector<float> stepsOf( const std::vector<float>& values, dim_t batch, dim_t steps, dim_t width, dim_t first, dim_t count )
        {
            std::vector<float> slice;

            for ( dim_t b = 0; b < batch; ++b )
            {
                const auto begin = values.begin() + static_cast<std::ptrdiff_t>( ( b * steps + first ) * width );
                slice.insert( slice.end(), begin, begin + static_cast<std::ptrdiff_t>( count * width ) );
            }

            return slice;
        }

        /// Run steps [first, first + count) through the rule, as prefill at `first` or as a decode, returning the outputs as floats.
        template<TensorDataType TPrecision>
        std::vector<float> runSteps( GatedDeltaRule<DeviceType::Metal, TPrecision>& rule, const Sequence& sequence, dim_t first, dim_t count, bool decode )
        {
            const Geometry& g = sequence.geometry;
            const auto tensorOf = [&]( const std::vector<float>& values, dim_t width ) {
                return metalTensor<TPrecision>( stepsOf( values, g.batch, sequence.steps, width, first, count ), shape_t{ g.batch, count, width } );
            };

            auto q = tensorOf( sequence.q, g.keyWidth() );
            auto k = tensorOf( sequence.k, g.keyWidth() );
            auto v = tensorOf( sequence.v, g.valueWidth() );
            auto a = tensorOf( sequence.a, g.value_heads );
            auto b = tensorOf( sequence.b, g.value_heads );

            auto& output = decode ? rule.decode( q, k, v, a, b, first ) : rule.prefill( q, k, v, a, b, first );
            rule.synchronize();

            return readFloats<TPrecision>( output );
        }

        template<TensorDataType TPrecision>
        std::vector<double> stored( const std::vector<float>& values )
        {
            return asStored<TPrecision>( std::span<const float>( values ) );
        }

        Sequence randomSequence( const Geometry& g, dim_t steps, std::uint32_t seed )
        {
            const auto count = [&]( dim_t width ) { return static_cast<std::size_t>( g.batch * steps * width ); };

            return Sequence{ g, steps,
                randomValues( count( g.keyWidth() ), seed, 1.0f ), randomValues( count( g.keyWidth() ), seed + 1, 1.0f ),
                randomValues( count( g.valueWidth() ), seed + 2, 1.0f ), randomValues( count( g.value_heads ), seed + 3, 1.0f ),
                randomValues( count( g.value_heads ), seed + 4, 1.0f ), randomValues( static_cast<std::size_t>( g.value_heads ), seed + 5, 0.5f ),
                randomValues( static_cast<std::size_t>( g.value_heads ), seed + 6, 1.0f ) };
        }

        Sequence oracleSequence()
        {
            const auto vectorOf = []( const auto& array ) { return std::vector<float>( std::begin( array ), std::end( array ) ); };

            return Sequence{ Geometry{ kBatch, kKeyHeads, kValueHeads, kHeadKeyDim, kHeadValueDim }, kSteps,
                vectorOf( kQ ), vectorOf( kK ), vectorOf( kV ), vectorOf( kA ), vectorOf( kB ), vectorOf( kALog ), vectorOf( kDtBias ) };
        }

        /// A prefill of `prefill_steps`, then decodes to the sequence's end, each output and the state at the end against the recurrence.
        template<TensorDataType TPrecision>
        void expectPrefillThenDecodes( const Sequence& sequence, dim_t prefill_steps )
        {
            const Geometry& g = sequence.geometry;
            auto rule = builtRule<TPrecision>( sequence, prefill_steps );
            Recurrence recurrence( g );

            const auto storedSteps = [&]( const std::vector<float>& values, dim_t width, dim_t first, dim_t count ) {
                return stored<TPrecision>( stepsOf( values, g.batch, sequence.steps, width, first, count ) );
            };

            const auto expected = [&]( dim_t first, dim_t count ) {
                return recurrence.advance( storedSteps( sequence.q, g.keyWidth(), first, count ), storedSteps( sequence.k, g.keyWidth(), first, count ),
                    storedSteps( sequence.v, g.valueWidth(), first, count ), storedSteps( sequence.a, g.value_heads, first, count ),
                    storedSteps( sequence.b, g.value_heads, first, count ), stored<TPrecision>( sequence.a_log ), stored<TPrecision>( sequence.dt_bias ),
                    count );
            };

            const std::vector<float> prefill = runSteps<TPrecision>( *rule, sequence, 0, prefill_steps, false );
            EXPECT_LE( relativeError( prefill, expected( 0, prefill_steps ) ), kBar<TPrecision> ) << "prefill of " << prefill_steps;

            for ( dim_t t = prefill_steps; t < sequence.steps; ++t )
            {
                const std::vector<float> step = runSteps<TPrecision>( *rule, sequence, t, 1, true );
                EXPECT_LE( relativeError( step, expected( t, 1 ) ), kBar<TPrecision> ) << "decode at " << t;
            }

            auto snapshot = rule->makeStateSnapshot();
            rule->snapshotState( snapshot );

            EXPECT_LE( relativeError( std::span<const float>( snapshot.data(), static_cast<std::size_t>( snapshot.size() ) ), recurrence.state ), kStateBar );
        }
    }

    TYPED_TEST( GatedDeltaRuleMetal, MatchesCudasOracleWholeAndTokenByToken )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const Sequence sequence = oracleSequence();

        auto whole_rule = builtRule<kPrecision>( sequence, kSteps );
        const std::vector<float> whole = runSteps<kPrecision>( *whole_rule, sequence, 0, kSteps, false );

        auto token_rule = builtRule<kPrecision>( sequence, 1 );

        for ( dim_t i = 0; i < kSteps * kValueWidth; ++i )
        {
            EXPECT_NEAR( whole[static_cast<std::size_t>( i )], kExpectedOut[i], kOracleTolerance ) << "whole, index " << i;
        }

        for ( dim_t t = 0; t < kSteps; ++t )
        {
            const std::vector<float> step = runSteps<kPrecision>( *token_rule, sequence, t, 1, t > 0 );

            for ( dim_t w = 0; w < kValueWidth; ++w )
            {
                EXPECT_NEAR( step[static_cast<std::size_t>( w )], kExpectedOut[t * kValueWidth + w], kOracleTolerance ) << "step " << t << ", width " << w;
            }
        }
    }

    TYPED_TEST( GatedDeltaRuleMetal, MatchesTheRecurrenceAtSmallGeometries )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // Two batch items, a head of keys not a multiple of four, and more than one block of 32 value columns.
        expectPrefillThenDecodes<kPrecision>( randomSequence( Geometry{ 2, 2, 6, 6, 4 }, 12, 101 ), 9 );
        expectPrefillThenDecodes<kPrecision>( randomSequence( Geometry{ 1, 1, 2, 32, 40 }, 10, 103 ), 7 );
    }

    TYPED_TEST( GatedDeltaRuleMetal, MatchesTheRecurrenceAtQwensGeometry )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // Qwen 3.8: 16 key heads, 48 value heads, 128 by 128 per head. A prefill of 40, then decodes.
        expectPrefillThenDecodes<kPrecision>( randomSequence( Geometry{ 1, 16, 48, 128, 128 }, 48, 107 ), 40 );
    }

    TYPED_TEST( GatedDeltaRuleMetal, RefusesAHeadOfKeysPastTheKernelsHold )
    {
        const auto build = [] {
            GatedDeltaRule<DeviceType::Metal, TypeParam::value> rule( "delta_rule", GatedDeltaRuleConfig( 1, 1, 260, 4 ), Device::Metal( 0 ) );
            rule.build( BuildContext( shape_t{ 1, 1, 4 }, RuntimeMode::Inference, false ) );
        };

        EXPECT_THROW( build(), std::invalid_argument );
    }
}
