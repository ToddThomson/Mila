/**
 * @file Linear.Codebook.Metal.cpp
 * @brief Linear on Metal on Qwen's two- and three-bit codebook weights, in FP16 and BF16, against a double-precision reference
 *        on the codes, scales and table as the library's own codec packs and decodes them.
 *
 * The gate is Notebooks/MetalQwen.md section 1.1.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Dnn.Components.Linear;
import Dnn.Quantization.Weight.CodebookPacking;
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
        template<TensorDataType TPrecision, typename TPolicy>
        struct Case
        {
            static constexpr TensorDataType kPrecision = TPrecision;
            using Policy = TPolicy;

            static constexpr int kEntries = TPolicy::kCodebookEntries;
            static constexpr dim_t kGroup = TPolicy::kQuantizationGroupSize;
        };

        using Cases = ::testing::Types<
            Case<TensorDataType::FP16, PerGroupCodebook2<32>>,
            Case<TensorDataType::BF16, PerGroupCodebook2<32>>,
            Case<TensorDataType::FP16, PerGroupCodebook2<64>>,
            Case<TensorDataType::BF16, PerGroupCodebook2<64>>,
            Case<TensorDataType::FP16, PerGroupCodebook3<64>>,
            Case<TensorDataType::BF16, PerGroupCodebook3<64>>,
            Case<TensorDataType::FP16, PerGroupCodebook3<128>>,
            Case<TensorDataType::BF16, PerGroupCodebook3<128>>>;

        struct CaseNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                return std::format( "{}Codebook{}Group{}", T::kPrecision == TensorDataType::FP16 ? "Fp16" : "Bf16",
                    T::kEntries == 8 ? 3 : 2, T::kGroup );
            }
        };

        TensorBlobView bytesBlob( std::span<const std::uint8_t> bytes, const shape_t& shape )
        {
            TensorMetadata metadata;
            metadata.dtype = TensorDataType::UINT8;
            metadata.shape = shape;
            metadata.total_bytes = bytes.size_bytes();

            return TensorBlobView( std::move( metadata ), bytes.data(), bytes.size_bytes() );
        }

        TensorBlobView floatsBlob( std::span<const float> values, const shape_t& shape )
        {
            TensorMetadata metadata;
            metadata.dtype = TensorDataType::FP32;
            metadata.shape = shape;
            metadata.total_bytes = values.size_bytes();

            return TensorBlobView( std::move( metadata ), values.data(), values.size_bytes() );
        }

        /// An ascending, asymmetric table, as the converter writes one.
        template<int TEntries>
        std::vector<float> tableOf()
        {
            if constexpr ( TEntries == 4 )
            {
                return { -1.0f, -0.28125f, 0.34375f, 0.9375f };
            } else {
                return { -1.0f, -0.6171875f, -0.34375f, -0.109375f, 0.125f, 0.3828125f, 0.6640625f, 0.96875f };
            }
        }

        template<typename TCase>
        struct Fixture
        {
            static constexpr TensorDataType kPrecision = TCase::kPrecision;
            static constexpr dim_t kGroup = TCase::kGroup;
            static constexpr bool kThreeBits = TCase::kEntries == 8;

            using LinearType = Mila::Dnn::Linear<DeviceType::Metal, kPrecision, typename TCase::Policy>;
            using TensorType = Tensor<kPrecision, MetalMemoryResource>;

            dim_t inputs;
            dim_t outputs;
            std::vector<std::uint8_t> low_plane;
            std::vector<std::uint8_t> high_plane;
            std::vector<std::uint16_t> scales;
            std::vector<float> table;
            std::vector<float> weight_values;
            std::unique_ptr<LinearType> linear;

            Fixture( dim_t input_features, dim_t output_features, dim_t built_rows )
                : inputs( input_features ), outputs( output_features ),
                  low_plane( static_cast<std::size_t>( output_features * packedRowBytesForTwoBitCodes( input_features ) ) ),
                  high_plane( static_cast<std::size_t>( output_features * packedRowBytesForOneBitPlane( input_features ) ) ),
                  scales( static_cast<std::size_t>( output_features * groupsPerRow( input_features, kGroup ) ) ),
                  table( tableOf<TCase::kEntries>() ),
                  weight_values( static_cast<std::size_t>( output_features * input_features ) )
            {
                std::mt19937 engine( 41 );
                std::uniform_int_distribution<int> code( 0, TCase::kEntries - 1 );
                std::uniform_real_distribution<float> magnitude( 0.005f, 0.08f );

                std::vector<std::uint8_t> codes( weight_values.size() );
                std::ranges::generate( codes, [&] { return static_cast<std::uint8_t>( code( engine ) ); } );
                std::ranges::generate( scales, [&] { return floatToHalfBits( magnitude( engine ) ); } );

                // A group of zero scale, which a group of zero weights packs to.
                scales[1] = floatToHalfBits( 0.0f );

                if constexpr ( kThreeBits )
                {
                    packThreeBitCodes( codes.data(), outputs, inputs, low_plane.data(), high_plane.data() );
                } else {
                    packTwoBitCodes( codes.data(), outputs, inputs, low_plane.data() );
                }

                dequantizeCodes( codes.data(), scales.data(), table.data(), outputs, inputs, kGroup, weight_values.data() );

                LinearConfig config( input_features, output_features );
                config.withBias( false );

                linear = std::make_unique<LinearType>( "linear", config, Device::Metal( 0 ) );
                linear->build( BuildContext( shape_t{ built_rows, input_features }, RuntimeMode::Inference, false ) );

                linear->loadParameter( "weight", bytesBlob( low_plane, shape_t{ outputs, inputs / 4 } ) );
                linear->loadParameter( "weight_scale", blobOver( scales, shape_t{ outputs, inputs / kGroup }, TensorDataType::FP16 ) );
                linear->loadParameter( "weight_codebook", floatsBlob( table, shape_t{ static_cast<dim_t>( table.size() ) } ) );

                if constexpr ( kThreeBits )
                {
                    linear->loadParameter( "weight_high_plane", bytesBlob( high_plane, shape_t{ outputs, inputs / 8 } ) );
                }
            }

            /// Run `rows` rows, and return the error against the reference at every `stride`th output.
            double errorAt( dim_t rows, dim_t stride = 1 )
            {
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                TensorType input = metalTensor<kPrecision>( input_values, shape_t{ rows, inputs } );

                const TensorType& output = linear->forward( input );
                linear->synchronize();

                const std::vector<float> all = readFloats<kPrecision>( output );
                const std::vector<double> stored_input = asStored<kPrecision>( std::span<const float>( input_values ) );

                std::vector<float> actual;
                std::vector<double> reference;

                for ( dim_t r = 0; r < rows; ++r )
                {
                    for ( dim_t c = 0; c < outputs; c += stride )
                    {
                        double sum = 0.0;

                        for ( dim_t k = 0; k < inputs; ++k )
                        {
                            sum += stored_input[static_cast<std::size_t>( r * inputs + k )] * weight_values[static_cast<std::size_t>( c * inputs + k )];
                        }

                        actual.push_back( all[static_cast<std::size_t>( r * outputs + c )] );
                        reference.push_back( sum );
                    }
                }

                return relativeError( actual, reference );
            }
        };

        template<typename TCase>
        class LinearCodebookMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LinearCodebookMetal, Cases, CaseNames );

        // Own lanes for one row and for several, expanded weights from 16 rows, tiles of 128 and of 8 with rows left over.
        constexpr dim_t kEveryRowCount[] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 63, 64, 65, 127, 128, 129, 130, 135, 136, 143, 256, 1023, 1024, 1031 };
    }

    TYPED_TEST( LinearCodebookMetal, MatchesTheReferenceAtEveryRowCount )
    {
        Fixture<TypeParam> fixture( 256, 48, 1031 );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearCodebookMetal, MatchesTheReferenceWhenOutputsAreOffATileOrABlock )
    {
        for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 17 }, dim_t{ 1030 } } )
        {
            Fixture<TypeParam> fixture( 256, outputs, 1024 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 8 }, dim_t{ 19 }, dim_t{ 130 }, dim_t{ 1024 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << outputs << " outputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearCodebookMetal, MatchesTheReferenceOnQwensFeedForwardShapes )
    {
        // Qwen 3.8's gate and up, 5120 by 17408, and its down, 17408 by 5120, every 31st output checked.
        for ( const auto [inputs, outputs] : { std::pair{ dim_t{ 5120 }, dim_t{ 17408 } }, std::pair{ dim_t{ 17408 }, dim_t{ 5120 } } } )
        {
            Fixture<TypeParam> fixture( inputs, outputs, 64 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 8 }, dim_t{ 64 } } )
            {
                EXPECT_LE( fixture.errorAt( rows, 31 ), kBar<TypeParam::kPrecision> ) << inputs << " by " << outputs << ", " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearCodebookMetal, RefusesFullPrecisionWeights )
    {
        Fixture<TypeParam> fixture( 256, 8, 8 );
        const std::vector<std::uint16_t> bits = randomBf16( 8 * 256, 51, 0.02f );

        EXPECT_THROW( fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 8, 256 } ) ), std::invalid_argument );
    }

    TEST( LinearCodebookMetalRows, FormatsExistAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, PerGroupCodebook2<32>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, PerGroupCodebook3<64>> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, PerGroupCodebook2<32>> ) );
    }
}
