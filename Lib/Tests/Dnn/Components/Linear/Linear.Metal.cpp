/**
 * @file Linear.Metal.cpp
 * @brief Linear on Metal, in FP16 and BF16, against a double-precision reference on the weights as stored.
 *
 * Weights arrive as a BF16 blob through loadParameter, as a model hands them: no published model stores FP16. The bars and the
 * shapes are the gate's, in Notebooks/MetalLlamaOperations.md.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.MetalExecutionContext;
import Compute.MetalMatrixProduct;
import Dnn.Components.Linear;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Metal;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        template<TensorDataType TPrecision>
        struct Fixture
        {
            using LinearType = Mila::Dnn::Linear<DeviceType::Metal, TPrecision>;
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            dim_t inputs;
            dim_t outputs;
            std::vector<std::uint16_t> weight_bits;
            std::vector<std::uint16_t> bias_bits;
            std::unique_ptr<LinearType> linear;

            Fixture( dim_t input_features, dim_t output_features, dim_t built_rows, bool with_bias = false )
                : inputs( input_features ), outputs( output_features ),
                  weight_bits( randomBf16( static_cast<std::size_t>( input_features * output_features ), 11, 0.02f ) )
            {
                LinearConfig config( input_features, output_features );
                config.withBias( with_bias );

                linear = std::make_unique<LinearType>( "linear", config, Device::Metal( 0 ) );
                linear->build( BuildContext( shape_t{ built_rows, input_features }, RuntimeMode::Inference, false ) );

                linear->loadParameter( "weight", blobOver( weight_bits, shape_t{ output_features, input_features } ) );

                if ( with_bias )
                {
                    bias_bits = randomBf16( static_cast<std::size_t>( output_features ), 12, 0.5f );
                    linear->loadParameter( "bias", blobOver( bias_bits, shape_t{ output_features } ) );
                }
            }

            /// Run `rows` rows, and return the error against the reference on what the tensors hold.
            double errorAt( dim_t rows )
            {
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                TensorType input = metalTensor<TPrecision>( input_values, shape_t{ rows, inputs } );

                const TensorType& output = linear->forward( input );
                linear->synchronize();

                const std::vector<float> actual = readFloats<TPrecision>( output );
                const std::vector<double> stored_input = asStored<TPrecision>( std::span<const float>( input_values ) );

                std::vector<double> reference( static_cast<std::size_t>( rows * outputs ) );

                for ( dim_t r = 0; r < rows; ++r )
                {
                    for ( dim_t c = 0; c < outputs; ++c )
                    {
                        double sum = bias_bits.empty() ? 0.0
                            : asStored<TPrecision>( fromBf16Bits( bias_bits[static_cast<std::size_t>( c )] ) );

                        for ( dim_t k = 0; k < inputs; ++k )
                        {
                            sum += stored_input[static_cast<std::size_t>( r * inputs + k )]
                                * asStored<TPrecision>( fromBf16Bits( weight_bits[static_cast<std::size_t>( c * inputs + k )] ) );
                        }

                        reference[static_cast<std::size_t>( r * outputs + c )] = sum;
                    }
                }

                return relativeError( actual, reference );
            }
        };

        template<typename TPrecisionConstant>
        class LinearMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LinearMetal, Precisions, PrecisionNames );

        // 136 and 143 are where a tile of 8 follows tiles of 128, at an offset into every matrix. 2 to 7 pad a tile, and 10, 15, 130 and
        // 135 pad one after whole tiles.
        constexpr dim_t kEveryRowCount[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 127, 128, 129, 130, 135, 136, 143, 256 };

        /// A product made directly, with room around its output to see what it wrote.
        template<TensorDataType TPrecision>
        struct DirectProduct
        {
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            static constexpr float kUntouched = -7.0f;

            int rows;
            int inputs;
            int outputs;
            int stride;
            std::vector<float> input_values;
            std::vector<float> weight_values;
            TensorType input;
            TensorType weight;
            TensorType output;

            DirectProduct( int row_count, int input_count, int output_count, int output_stride )
                : rows( row_count ), inputs( input_count ), outputs( output_count ), stride( output_stride ),
                  input_values( randomValues( static_cast<std::size_t>( row_count * input_count ), 21, 1.0f ) ),
                  weight_values( randomValues( static_cast<std::size_t>( output_count * input_count ), 22, 0.02f ) ),
                  input( metalTensor<TPrecision>( input_values, shape_t{ row_count, input_count } ) ),
                  weight( metalTensor<TPrecision>( weight_values, shape_t{ output_count, input_count } ) ),
                  output( metalTensor<TPrecision>(
                      std::vector<float>( static_cast<std::size_t>( ( row_count + 8 ) * output_stride ), kUntouched ),
                      shape_t{ row_count + 8, output_stride } ) )
            {
            }

            [[nodiscard]] MatrixProduct product( bool pad ) const
            {
                return MatrixProduct{
                    static_cast<const ITensor&>( input ).rawData(), static_cast<const ITensor&>( weight ).rawData(),
                    static_cast<const ITensor&>( output ).rawData(), rows, inputs, outputs, stride, pad ? 0 : kNeverPad };
            }

            [[nodiscard]] std::vector<double> reference() const
            {
                const std::vector<double> stored_input = asStored<TPrecision>( std::span<const float>( input_values ) );
                const std::vector<double> stored_weight = asStored<TPrecision>( std::span<const float>( weight_values ) );

                std::vector<double> values( static_cast<std::size_t>( rows * outputs ), 0.0 );

                for ( int r = 0; r < rows; ++r )
                {
                    for ( int c = 0; c < outputs; ++c )
                    {
                        for ( int k = 0; k < inputs; ++k )
                        {
                            values[static_cast<std::size_t>( r * outputs + c )] +=
                                stored_input[static_cast<std::size_t>( r * inputs + k )] * stored_weight[static_cast<std::size_t>( c * inputs + k )];
                        }
                    }
                }

                return values;
            }
        };
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceAtEveryRowCount )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 256 );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, FewerRowsThanWereBuiltGiveThoseRowsTheSameValues )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 128 );

        // 128 rows go through a tile of 128, and their first 9 alone through a tile of 8 and the own kernel. The values match bit for bit
        // only if each path reads the same rows and rounds the same way, which is itself what this pins.
        const std::vector<float> values = randomValues( 128 * 64, 14, 1.0f );
        auto input = metalTensor<kPrecision>( values, shape_t{ 128, 64 } );

        const std::vector<float> all = readFloats<kPrecision>( [&]() -> const auto& {
            const auto& output = fixture.linear->forward( input );
            fixture.linear->synchronize();

            return output;
        }() );

        const std::vector<float> few_values( values.begin(), values.begin() + 9 * 64 );
        auto few = metalTensor<kPrecision>( few_values, shape_t{ 9, 64 } );

        const auto& viewed = fixture.linear->forward( few );
        fixture.linear->synchronize();

        EXPECT_EQ( readFloats<kPrecision>( viewed ), std::vector<float>( all.begin(), all.begin() + 9 * 48 ) );
    }

    TYPED_TEST( LinearMetal, RowsThatPadATileGiveTheValuesAFullTileGivesThem )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        MetalExecutionContext context( Device::Metal( 0 ) );

        DirectProduct<kPrecision> full( 128, 64, 48, 48 );

        encodeMatrixProduct<kPrecision>( context, full.product( false ) );
        context.synchronize();

        const std::vector<float> all = readFloats<kPrecision>( full.output );

        // A seed gives the same values in the same order, so these rows are the first rows of the full product.
        for ( const int rows : { 7, 2, 3, 4, 5, 6, 10, 15, 127 } )
        {
            DirectProduct<kPrecision> few( rows, 64, 48, 48 );

            encodeMatrixProduct<kPrecision>( context, few.product( true ) );
            context.synchronize();

            const std::vector<float> read = readFloats<kPrecision>( few.output );

            EXPECT_EQ( std::vector<float>( read.begin(), read.begin() + rows * 48 ), std::vector<float>( all.begin(), all.begin() + rows * 48 ) ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, PaddedRowsAreTheTilesAndNotAlsoTheOwnKernels )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        MetalExecutionContext context( Device::Metal( 0 ) );

        const auto firstRows = [&]( DirectProduct<kPrecision>& direct, int rows, bool pad )
        {
            encodeMatrixProduct<kPrecision>( context, direct.product( pad ) );
            context.synchronize();

            const std::vector<float> read = readFloats<kPrecision>( direct.output );

            return std::vector<float>( read.begin(), read.begin() + rows * direct.outputs );
        };

        // At this size the two paths round differently, which small shapes do not show: the own kernel running these rows after the tile
        // would leave its values, with every value still right.
        DirectProduct<kPrecision> tile( 8, 4096, 4096, 4096 );
        DirectProduct<kPrecision> padded( 3, 4096, 4096, 4096 );
        DirectProduct<kPrecision> own( 3, 4096, 4096, 4096 );

        const std::vector<float> by_tile = firstRows( tile, 3, false );

        EXPECT_NE( firstRows( own, 3, false ), by_tile );
        EXPECT_EQ( firstRows( padded, 3, true ), by_tile );
    }

    TYPED_TEST( LinearMetal, APaddedProductWritesItsRowsAndNothingElse )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // 35 outputs leave 3 to the own kernel, on every row. A stride past the outputs is what attention's scores have.
        for ( const auto [rows, outputs, stride] : { std::tuple{ 3, 48, 48 }, std::tuple{ 3, 48, 64 }, std::tuple{ 7, 35, 40 }, std::tuple{ 11, 48, 64 }, std::tuple{ 135, 35, 40 } } )
        {
            DirectProduct<kPrecision> direct( rows, 64, outputs, stride );
            MetalExecutionContext context( Device::Metal( 0 ) );

            const std::vector<float> input_before = readFloats<kPrecision>( direct.input );

            encodeMatrixProduct<kPrecision>( context, direct.product( true ) );
            context.synchronize();

            const std::vector<float> written = readFloats<kPrecision>( direct.output );
            const std::vector<double> reference = direct.reference();

            std::vector<float> in_rows( reference.size() );

            for ( int r = 0; r < rows + 8; ++r )
            {
                for ( int c = 0; c < stride; ++c )
                {
                    const float value = written[static_cast<std::size_t>( r * stride + c )];

                    if ( r < rows && c < outputs )
                    {
                        in_rows[static_cast<std::size_t>( r * outputs + c )] = value;
                    } else {
                        ASSERT_EQ( value, DirectProduct<kPrecision>::kUntouched ) << "row " << r << ", column " << c << " of " << rows << " rows";
                    }
                }
            }

            EXPECT_LE( relativeError( in_rows, reference ), kBar<kPrecision> ) << rows << " rows, " << outputs << " outputs";
            EXPECT_EQ( readFloats<kPrecision>( direct.input ), input_before ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, TheRowsThatPadATileAreZero )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        MetalExecutionContext context( Device::Metal( 0 ) );

        DirectProduct<kPrecision> earlier( 7, 64, 48, 48 );
        DirectProduct<kPrecision> later( 3, 64, 48, 48 );

        encodeMatrixProduct<kPrecision>( context, earlier.product( true ) );
        encodeMatrixProduct<kPrecision>( context, later.product( true ) );
        context.synchronize();

        // The tile's input is the start of scratch: 8 rows of 64.
        const auto* const scratch = static_cast<const std::uint16_t*>( context.getDeviceScratchBuffer( 1 ) );
        const auto* const input = static_cast<const std::uint16_t*>( static_cast<const ITensor&>( later.input ).rawData() );

        EXPECT_TRUE( std::equal( scratch, scratch + 3 * 64, input ) );
        EXPECT_TRUE( std::all_of( scratch + 3 * 64, scratch + 8 * 64, []( std::uint16_t bits ) { return bits == 0; } ) );
    }

    TYPED_TEST( LinearMetal, OnlyAProductThatPadsAsksForScratch )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const auto runsWithNoScratch = []( int rows, int outputs, bool pad )
        {
            DirectProduct<kPrecision> direct( rows, 64, outputs, outputs );
            MetalExecutionContext context( Device::Metal( 0 ) );

            context.reserveScratch( 0 );

            try
            {
                encodeMatrixProduct<kPrecision>( context, direct.product( pad ) );
                context.synchronize();
            }
            catch ( const std::logic_error& )
            {
                return false;
            }

            return true;
        };

        EXPECT_TRUE( runsWithNoScratch( 3, 48, false ) );
        EXPECT_FALSE( runsWithNoScratch( 2, 48, true ) );
        EXPECT_FALSE( runsWithNoScratch( 7, 48, true ) );
        EXPECT_TRUE( runsWithNoScratch( 1, 48, true ) );
        EXPECT_TRUE( runsWithNoScratch( 8, 48, true ) );
        EXPECT_TRUE( runsWithNoScratch( 9, 48, true ) );
        EXPECT_TRUE( runsWithNoScratch( 3, 15, true ) );
    }

    TYPED_TEST( LinearMetal, ALinearPadsOnlyRowsLargeEnoughToPay )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // 4096 x 4096: 2 rows are 33.5 million multiplies and go through the own kernel, 3 are 50.3 million and pad.
        Fixture<kPrecision> fixture( 4096, 4096, 16 );
        fixture.linear->getExecutionContext()->reserveScratch( 0 );

        const auto asksForScratch = [&]( dim_t rows )
        {
            const std::vector<float> values( static_cast<std::size_t>( rows * 4096 ), 1.0f );
            auto input = metalTensor<kPrecision>( values, shape_t{ rows, 4096 } );

            try
            {
                fixture.linear->forward( input );
                fixture.linear->synchronize();
            }
            catch ( const std::logic_error& )
            {
                return true;
            }

            return false;
        };

        EXPECT_FALSE( asksForScratch( 1 ) );
        EXPECT_FALSE( asksForScratch( 2 ) );
        EXPECT_TRUE( asksForScratch( 3 ) );
        EXPECT_TRUE( asksForScratch( 7 ) );
        EXPECT_FALSE( asksForScratch( 8 ) );
        EXPECT_FALSE( asksForScratch( 10 ) );
        EXPECT_TRUE( asksForScratch( 11 ) );
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWhenItPads )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // 8195 outputs leave 3 to the own kernel on every row. Each row count is past the point where a Linear pads.
        Fixture<kPrecision> fixture( 2048, 8195, 16, true );

        for ( const dim_t rows : { dim_t{ 3 }, dim_t{ 7 }, dim_t{ 11 } } )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, TheFootprintsScratchIsWhatPaddingAsksFor )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const auto scratchAt = []( dim_t rows, dim_t outputs )
        {
            Fixture<kPrecision> fixture( 4096, outputs, rows );

            const BuildContext context = BuildContext( shape_t{ rows, 4096 }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

            LinearConfig config( 4096, outputs );
            config.withBias( false );

            typename Fixture<kPrecision>::LinearType predictor( "predictor", config, Device::Metal( 0 ) );

            const std::size_t built = fixture.linear->getMemoryStats().device_scratch_bytes;

            EXPECT_EQ( predictor.getRequiredMemory( context ).device_scratch_bytes, built ) << rows << " rows, " << outputs << " outputs";

            // A request past the reservation throws, so a run at every padded count holds the footprint to what is asked.
            fixture.linear->getExecutionContext()->reserveScratch( built );

            for ( dim_t row_count = 1; row_count <= std::min<dim_t>( rows, 7 ); ++row_count )
            {
                const std::vector<float> values( static_cast<std::size_t>( row_count * 4096 ), 1.0f );
                auto input = metalTensor<kPrecision>( values, shape_t{ row_count, 4096 } );

                EXPECT_NO_THROW( fixture.linear->forward( input ) ) << row_count << " rows, " << outputs << " outputs";
                fixture.linear->synchronize();
            }

            return built;
        };

        // One row, as a generating model's output layer holds, asks for none, and neither do rows too few to pay. 8 rows of 1600 outputs
        // would pay at 52 million multiplies, but a tile takes 8, so at most 7 are ever left over, at 46 million.
        EXPECT_EQ( scratchAt( 1, 8192 ), std::size_t{ 0 } );
        EXPECT_EQ( scratchAt( 2, 4096 ), std::size_t{ 0 } );
        EXPECT_EQ( scratchAt( 8, 1600 ), std::size_t{ 0 } );

        // These pad: 8 rows in and out, with the outputs of whole tiles only.
        const std::size_t granularity = allocationGranularity( Device::Metal( 0 ) );

        EXPECT_EQ( scratchAt( 3, 4096 ), occupiedDeviceBytes( 2 * 8 * ( 4096 + 4096 ), granularity ) );
        EXPECT_EQ( scratchAt( 128, 4111 ), occupiedDeviceBytes( 2 * 8 * ( 4096 + 4096 ), granularity ) );
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWhenOutputsAreOffTheTile )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 5 }, dim_t{ 15 }, dim_t{ 17 }, dim_t{ 35 } } )
        {
            Fixture<kPrecision> fixture( 64, outputs, 128 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 5 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 19 }, dim_t{ 128 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << outputs << " outputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWhenInputsAreOffAVector )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        for ( const dim_t inputs : { dim_t{ 3 }, dim_t{ 13 }, dim_t{ 65 } } )
        {
            Fixture<kPrecision> fixture( inputs, 32, 32 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 3 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 19 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << inputs << " inputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWithABias )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 256, true );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, AnEarlierCallLeavesNoTraceInALaterOne )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 128 );

        EXPECT_LE( fixture.errorAt( 128 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 1 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 7 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 3 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 128 ), kBar<kPrecision> );
    }

    TYPED_TEST( LinearMetal, RefusesABlobOfAnotherType )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 8, 8, 8 );
        const std::vector<std::uint16_t> bits( 64, 0 );

        EXPECT_THROW( fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 8, 8 }, TensorDataType::INT16 ) ), std::invalid_argument );
    }

    TEST( LinearMetalFp16, HoldsBf16WeightsExactlyInItsNormalRange )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        // 3 at the first input, so that the output is the weight read back, times 3.
        const std::array<float, 4> values{ 3.0f, 0.0f, 0.0f, 0.0f };
        const std::vector<std::uint16_t> bits{ toBf16Bits( 1.0f / 256.0f ), 0, 0, 0 };

        fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) );

        auto input = metalTensor<TensorDataType::FP16>( values, shape_t{ 1, 4 } );
        const auto& output = fixture.linear->forward( input );
        fixture.linear->synchronize();

        EXPECT_EQ( readFloats<TensorDataType::FP16>( output )[0], 3.0f / 256.0f );
    }

    TEST( LinearMetalFp16, RefusesAWeightFp16CannotHoldAndNamesIt )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        for ( const float value : { 70144.0f, -70144.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() } )
        {
            const std::vector<std::uint16_t> bits{ 0, toBf16Bits( value ), 0, 0 };

            try
            {
                fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) );
                FAIL() << "A weight of " << value << " was accepted";
            }
            catch ( const std::invalid_argument& error )
            {
                EXPECT_NE( std::string( error.what() ).find( "linear.weight" ), std::string::npos ) << error.what();
            }
        }
    }

    TEST( LinearMetalFp16, AcceptsTheLargestWeightBf16AndFp16BothHold )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        // BF16's largest value below 65504: the next one up is 65536.
        const std::vector<std::uint16_t> bits{ toBf16Bits( 65280.0f ), 0, 0, 0 };

        EXPECT_NO_THROW( fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) ) );
    }

    TEST( LinearMetalRows, OnlyFp16AndBf16Exist )
    {
        using namespace Mila::Dnn::Quant::Weight;

        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, NoWeightQuant> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, NoWeightQuant> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, NoWeightQuant> ) );
    }
}
