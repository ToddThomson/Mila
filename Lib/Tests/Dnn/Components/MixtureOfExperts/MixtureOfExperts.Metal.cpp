/**
 * @file MixtureOfExperts.Metal.cpp
 * @brief The mixture-of-experts bank on Metal, unquantized, per-group FP4 and Q4_0, in FP16 and BF16, against a double-precision
 *        reference on the weights as stored or as the library's own codec decodes them.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Dnn.Quantization.Weight.Int4Packing;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        constexpr dim_t kHidden = 128;
        constexpr dim_t kIntermediate = 64;
        constexpr dim_t kExperts = 16;
        constexpr dim_t kTopK = 4;
        constexpr dim_t kTokens = 48;

        using IndexTensor = Tensor<TensorDataType::INT32, MetalMemoryResource>;

        template<TensorDataType TPrecision, typename TPolicy = NoWeightQuant>
        using Bank = Mila::Dnn::MixtureOfExperts<DeviceType::Metal, TPrecision, ActivationType::Gelu, TPolicy>;

        double geluTanh( double x )
        {
            return 0.5 * x * ( 1.0 + std::tanh( std::sqrt( 2.0 / std::numbers::pi ) * ( x + 0.044715 * x * x * x ) ) );
        }

        /// Weights FP4 per 64 holds exactly: E2M1 values times a power-of-two scale, each group's largest 6 times it.
        std::vector<float> fp4ExactWeights( dim_t rows, dim_t columns, std::uint32_t seed )
        {
            constexpr std::array<float, 8> kMagnitudes{ 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

            std::mt19937 engine( seed );
            std::vector<float> values( static_cast<std::size_t>( rows * columns ) );

            for ( dim_t row = 0; row < rows; ++row )
            {
                for ( dim_t start = 0; start < columns; start += 64 )
                {
                    const float scale = std::ldexp( 1.0f, -static_cast<int>( 2 + engine() % 5 ) );

                    for ( dim_t column = start; column < start + 64; ++column )
                    {
                        values[static_cast<std::size_t>( row * columns + column )] = ( engine() & 1u ? -1.0f : 1.0f ) * kMagnitudes[engine() % 8] * scale;
                    }

                    values[static_cast<std::size_t>( row * columns + start + engine() % 64 )] = 6.0f * scale;
                }
            }

            return values;
        }

        /// One case's tokens, routing and weights, with the values the bank's weights hold once loaded.
        struct Case
        {
            std::vector<float> input;
            std::vector<float> combine;
            std::vector<std::int32_t> indices;
            std::vector<double> gate_up;
            std::vector<double> down;
        };

        std::vector<std::int32_t> routing( dim_t tokens )
        {
            std::vector<std::int32_t> indices( static_cast<std::size_t>( tokens * kTopK ) );

            for ( dim_t token = 0; token < tokens; ++token )
            {
                for ( dim_t slot = 0; slot < kTopK; ++slot )
                {
                    indices[static_cast<std::size_t>( token * kTopK + slot )] = static_cast<std::int32_t>( ( token * 7 + slot * 5 ) % kExperts );
                }
            }

            return indices;
        }

        template<TensorDataType TPrecision>
        Case caseFor( std::uint32_t seed )
        {
            Case result;
            result.input = randomValues( static_cast<std::size_t>( kTokens * kHidden ), seed, 1.0f );
            result.combine = randomValues( static_cast<std::size_t>( kTokens * kTopK ), seed + 1, 0.5f );
            result.indices = routing( kTokens );

            return result;
        }

        /// The bank's output in double on what the kernels read: the input and routing weights as stored, gated values in full.
        template<TensorDataType TPrecision>
        std::vector<double> reference( const Case& routed, dim_t tokens )
        {
            const std::vector<double> x = asStored<TPrecision>( std::span<const float>( routed.input ) );
            const std::vector<double> w = asStored<TPrecision>( std::span<const float>( routed.combine ) );

            std::vector<double> output( static_cast<std::size_t>( tokens * kHidden ), 0.0 );

            for ( dim_t token = 0; token < tokens; ++token )
            {
                for ( dim_t slot = 0; slot < kTopK; ++slot )
                {
                    const dim_t expert = routed.indices[static_cast<std::size_t>( token * kTopK + slot )];
                    std::vector<double> gated( static_cast<std::size_t>( kIntermediate ) );

                    for ( dim_t unit = 0; unit < kIntermediate; ++unit )
                    {
                        double gate = 0.0;
                        double up = 0.0;

                        for ( dim_t column = 0; column < kHidden; ++column )
                        {
                            const double value = x[static_cast<std::size_t>( token * kHidden + column )];

                            gate += routed.gate_up[static_cast<std::size_t>( ( expert * 2 * kIntermediate + unit ) * kHidden + column )] * value;
                            up += routed.gate_up[static_cast<std::size_t>( ( expert * 2 * kIntermediate + kIntermediate + unit ) * kHidden + column )] * value;
                        }

                        gated[static_cast<std::size_t>( unit )] = geluTanh( gate ) * up;
                    }

                    for ( dim_t column = 0; column < kHidden; ++column )
                    {
                        double projected = 0.0;

                        for ( dim_t unit = 0; unit < kIntermediate; ++unit )
                        {
                            projected += routed.down[static_cast<std::size_t>( ( expert * kHidden + column ) * kIntermediate + unit )] * gated[static_cast<std::size_t>( unit )];
                        }

                        output[static_cast<std::size_t>( token * kHidden + column )] += w[static_cast<std::size_t>( token * kTopK + slot )] * projected;
                    }
                }
            }

            return output;
        }

        TensorBlobView bytesBlob( std::span<const std::uint8_t> bytes, const shape_t& shape )
        {
            TensorMetadata metadata;
            metadata.dtype = TensorDataType::UINT8;
            metadata.shape = shape;
            metadata.total_bytes = bytes.size_bytes();

            return TensorBlobView( std::move( metadata ), bytes.data(), bytes.size_bytes() );
        }

        std::vector<std::uint16_t> bf16Of( std::span<const float> values )
        {
            std::vector<std::uint16_t> bits( values.size() );
            std::ranges::transform( values, bits.begin(), toBf16Bits );

            return bits;
        }

        /// A bank built for `built_tokens`, its two projections loaded as BF16 weights a model hands it; the case gains the values they hold.
        template<TensorDataType TPrecision, typename TPolicy>
        std::unique_ptr<Bank<TPrecision, TPolicy>> bf16Bank( Case& routed, const std::vector<float>& gate_up, const std::vector<float>& down,
            dim_t built_tokens = kTokens )
        {
            auto bank = std::make_unique<Bank<TPrecision, TPolicy>>(
                "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Metal( 0 ) );
            bank->build( BuildContext( shape_t{ built_tokens, kHidden }, RuntimeMode::Inference, false ) );

            const std::vector<std::uint16_t> gate_up_bits = bf16Of( gate_up );
            const std::vector<std::uint16_t> down_bits = bf16Of( down );

            bank->loadParameter( "gate_up_proj", blobOver( gate_up_bits, shape_t{ kExperts, 2 * kIntermediate, kHidden } ) );
            bank->loadParameter( "down_proj", blobOver( down_bits, shape_t{ kExperts, kHidden, kIntermediate } ) );
            bank->synchronize();

            const auto stored = [&]( const std::vector<std::uint16_t>& bits ) {
                std::vector<double> values( bits.size() );
                std::ranges::transform( bits, values.begin(), []( std::uint16_t b ) { return asStored<TPrecision>( fromBf16Bits( b ) ); } );

                return values;
            };

            // FP4's exact weights are exact at either precision, and so are their stored values.
            routed.gate_up = stored( gate_up_bits );
            routed.down = stored( down_bits );

            return bank;
        }

        /// A Q4_0 bank loaded with codes and scales as a package stores them; the case gains the values the codec decodes.
        template<TensorDataType TPrecision>
        std::unique_ptr<Bank<TPrecision, PerGroupInt4<32>>> q4Bank( Case& routed, dim_t built_tokens = kTokens )
        {
            auto bank = std::make_unique<Bank<TPrecision, PerGroupInt4<32>>>(
                "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Metal( 0 ) );
            bank->build( BuildContext( shape_t{ built_tokens, kHidden }, RuntimeMode::Inference, false ) );

            const auto load = [&]( const char* name, dim_t rows, dim_t columns, std::uint32_t seed, std::vector<double>& decoded ) {
                const std::vector<float> weights = randomValues( static_cast<std::size_t>( kExperts * rows * columns ), seed, 0.05f );
                std::vector<std::uint8_t> codes( static_cast<std::size_t>( kExperts * rows * columns / 2 ) );
                std::vector<std::uint16_t> scales( static_cast<std::size_t>( kExperts * rows * columns / 32 ) );
                std::vector<float> values( weights.size() );

                quantizeInt4( weights.data(), kExperts * rows, columns, 32, codes.data(), scales.data() );
                dequantizeInt4( codes.data(), scales.data(), kExperts * rows, columns, 32, values.data() );

                bank->loadParameter( name, bytesBlob( codes, shape_t{ kExperts, rows, columns / 2 } ) );
                bank->loadParameter( std::string( name ) + "_scale", blobOver( scales, shape_t{ kExperts, rows, columns / 32 }, TensorDataType::FP16 ) );

                decoded.assign( values.begin(), values.end() );
            };

            load( "gate_up_proj", 2 * kIntermediate, kHidden, 31, routed.gate_up );
            load( "down_proj", kHidden, kIntermediate, 32, routed.down );
            bank->synchronize();

            return bank;
        }

        template<typename TTensor>
        struct PrecisionOf;

        template<TensorDataType TPrecision, typename TMemoryResource>
        struct PrecisionOf<Tensor<TPrecision, TMemoryResource>>
        {
            static constexpr TensorDataType value = TPrecision;
        };

        /// The bank over the case's first `tokens` tokens, starting at `first`, through forward() or decodeTokens().
        template<typename TBank>
        std::vector<float> run( TBank& bank, const Case& routed, dim_t first, dim_t tokens, bool decode_tokens = false )
        {
            constexpr TensorDataType kPrecision = PrecisionOf<typename TBank::TensorType>::value;

            auto input = metalTensor<kPrecision>(
                std::span<const float>( routed.input ).subspan( static_cast<std::size_t>( first * kHidden ), static_cast<std::size_t>( tokens * kHidden ) ),
                shape_t{ 1, tokens, kHidden } );
            auto weights = metalTensor<kPrecision>(
                std::span<const float>( routed.combine ).subspan( static_cast<std::size_t>( first * kTopK ), static_cast<std::size_t>( tokens * kTopK ) ),
                shape_t{ 1, tokens, kTopK } );

            IndexTensor indices( Device::Metal( 0 ), shape_t{ 1, tokens, kTopK } );
            fill( indices, std::span<const std::int32_t>( routed.indices ).subspan( static_cast<std::size_t>( first * kTopK ), static_cast<std::size_t>( tokens * kTopK ) ) );

            auto& output = decode_tokens ? bank.decodeTokens( input, weights, indices ) : bank.forward( input, weights, indices );
            bank.synchronize();

            return readFloats<kPrecision>( output );
        }

        std::size_t bitsDiffering( const std::vector<float>& actual, const std::vector<float>& expected )
        {
            std::size_t differing = actual.size() == expected.size() ? 0 : std::max( actual.size(), expected.size() );

            for ( std::size_t i = 0; i < std::min( actual.size(), expected.size() ); ++i )
            {
                differing += std::bit_cast<std::uint32_t>( actual[i] ) != std::bit_cast<std::uint32_t>( expected[i] );
            }

            return differing;
        }

        /// Every token run alone, then every run of 1 to 8 tokens through decodeTokens(): bits differing from the full forward().
        template<typename TBank>
        std::size_t bitsDifferingFromForward( TBank& bank, const Case& routed )
        {
            const std::vector<float> all = run( bank, routed, 0, kTokens );
            std::size_t differing = 0;

            for ( dim_t token = 0; token < kTokens; ++token )
            {
                const std::vector<float> alone = run( bank, routed, token, 1 );

                differing += bitsDiffering( alone, std::vector<float>( all.begin() + token * kHidden, all.begin() + ( token + 1 ) * kHidden ) );
            }

            for ( dim_t rows = 1; rows <= 8; ++rows )
            {
                for ( dim_t first = 0; first + rows <= kTokens; first += rows )
                {
                    const std::vector<float> decoded = run( bank, routed, first, rows, true );

                    differing += bitsDiffering( decoded, std::vector<float>( all.begin() + first * kHidden, all.begin() + ( first + rows ) * kHidden ) );
                }
            }

            return differing;
        }

        template<typename TPrecisionConstant>
        class MixtureOfExpertsMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( MixtureOfExpertsMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( MixtureOfExpertsMetal, AnUnquantizedBankMatchesTheReference )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Case routed = caseFor<kPrecision>( 41 );
        auto bank = bf16Bank<kPrecision, NoWeightQuant>( routed,
            randomValues( static_cast<std::size_t>( kExperts * 2 * kIntermediate * kHidden ), 42, 0.1f ),
            randomValues( static_cast<std::size_t>( kExperts * kHidden * kIntermediate ), 43, 0.1f ) );

        EXPECT_LE( relativeError( run( *bank, routed, 0, kTokens ), reference<kPrecision>( routed, kTokens ) ), kBar<kPrecision> );
    }

    TYPED_TEST( MixtureOfExpertsMetal, AnFp4BankQuantizedAtLoadMatchesTheReference )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Case routed = caseFor<kPrecision>( 51 );
        auto bank = bf16Bank<kPrecision, PerGroupFp4<64>>( routed,
            fp4ExactWeights( kExperts * 2 * kIntermediate, kHidden, 52 ), fp4ExactWeights( kExperts * kHidden, kIntermediate, 53 ) );

        EXPECT_LE( relativeError( run( *bank, routed, 0, kTokens ), reference<kPrecision>( routed, kTokens ) ), kBar<kPrecision> );
    }

    TYPED_TEST( MixtureOfExpertsMetal, AQ4_0BankAsStoredMatchesTheReference )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Case routed = caseFor<kPrecision>( 61 );
        auto bank = q4Bank<kPrecision>( routed );

        EXPECT_LE( relativeError( run( *bank, routed, 0, kTokens ), reference<kPrecision>( routed, kTokens ) ), kBar<kPrecision> );
    }

    // Each token's output is the one it gets alone, through forward() and through a multi-token decode of up to 8, for every format.
    TYPED_TEST( MixtureOfExpertsMetal, EachTokenHasTheBitsItHasAlone )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        {
            Case routed = caseFor<kPrecision>( 71 );
            auto bank = bf16Bank<kPrecision, NoWeightQuant>( routed,
                randomValues( static_cast<std::size_t>( kExperts * 2 * kIntermediate * kHidden ), 72, 0.1f ),
                randomValues( static_cast<std::size_t>( kExperts * kHidden * kIntermediate ), 73, 0.1f ) );

            EXPECT_EQ( bitsDifferingFromForward( *bank, routed ), 0u ) << "unquantized";
        }

        {
            Case routed = caseFor<kPrecision>( 74 );
            auto bank = bf16Bank<kPrecision, PerGroupFp4<64>>( routed,
                fp4ExactWeights( kExperts * 2 * kIntermediate, kHidden, 75 ), fp4ExactWeights( kExperts * kHidden, kIntermediate, 76 ) );

            EXPECT_EQ( bitsDifferingFromForward( *bank, routed ), 0u ) << "FP4";
        }

        {
            Case routed = caseFor<kPrecision>( 77 );
            auto bank = q4Bank<kPrecision>( routed );

            EXPECT_EQ( bitsDifferingFromForward( *bank, routed ), 0u ) << "Q4_0";
        }
    }

    TYPED_TEST( MixtureOfExpertsMetal, AMultiTokenDecodeOfMoreThanEightIsRefused )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Case routed = caseFor<kPrecision>( 81 );
        auto bank = q4Bank<kPrecision>( routed );

        EXPECT_THROW( static_cast<void>( run( *bank, routed, 0, 9, true ) ), std::invalid_argument );
    }

    // A kernel cannot throw: an expert index out of range poisons its token and leaves the others finite.
    TYPED_TEST( MixtureOfExpertsMetal, AnExpertOutOfRangePoisonsOnlyItsToken )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Case routed = caseFor<kPrecision>( 91 );
        auto bank = q4Bank<kPrecision>( routed );

        routed.indices[static_cast<std::size_t>( 1 * kTopK + 2 )] = static_cast<std::int32_t>( kExperts );

        const std::vector<float> output = run( *bank, routed, 0, 2 );

        for ( dim_t column = 0; column < kHidden; ++column )
        {
            EXPECT_TRUE( std::isfinite( output[static_cast<std::size_t>( column )] ) ) << "valid token, column " << column;
            EXPECT_TRUE( std::isnan( output[static_cast<std::size_t>( kHidden + column )] ) ) << "poisoned token, column " << column;
        }
    }

    TYPED_TEST( MixtureOfExpertsMetal, AQ4_0BankRefusesFullPrecisionWeights )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Bank<kPrecision, PerGroupInt4<32>> bank( "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Metal( 0 ) );
        bank.build( BuildContext( shape_t{ 4, kHidden }, RuntimeMode::Inference, false ) );

        const std::vector<std::uint16_t> bits = randomBf16( static_cast<std::size_t>( kExperts * kHidden * kIntermediate ), 3, 0.05f );

        EXPECT_THROW( bank.loadParameter( "down_proj", blobOver( bits, shape_t{ kExperts, kHidden, kIntermediate } ) ), std::invalid_argument );
    }

    TYPED_TEST( MixtureOfExpertsMetal, APackedWidthNotAWholeGroupIsRefused )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Bank<kPrecision, PerGroupFp4<64>> bank( "experts", MixtureOfExpertsConfig( kHidden, 96, kExperts, kTopK ), Device::Metal( 0 ) );

        EXPECT_THROW( bank.build( BuildContext( shape_t{ 4, kHidden }, RuntimeMode::Inference, false ) ), std::invalid_argument );
    }

    TEST( MixtureOfExpertsMetalRows, OnlyFp16AndBf16Exist )
    {
        EXPECT_FALSE( ( OperationSupported<OperationType::MoeOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::MoeOp, DeviceType::Metal, TensorDataType::FP16> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::MoeOp, DeviceType::Metal, TensorDataType::BF16> ) );
    }
}
