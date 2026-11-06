/**
 * @file Linear.Quantized.Metal.cpp
 * @brief Linear on Metal on FP8 and FP4 weights, in FP16 and BF16, against a double-precision reference on the codes and scales
 *        as stored, and its quantize at load against CUDA's arithmetic.
 *
 * The gate is Notebooks/MetalQuantizedWeights.md section 1.1.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.MetalExecutionContext;
import Compute.MetalLinearOp;
import Dnn.Components.Linear;
import Serialization.Tensor;

#include "../MetalReference.h"
#include "../QuantizedReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Tests::Dnn::MetalReference;
    using namespace Mila::Tests::Dnn::QuantizedReference;

    namespace
    {
        template<TensorDataType TPrecision, typename TPolicy>
        struct Case
        {
            static constexpr TensorDataType kPrecision = TPrecision;
            using Policy = TPolicy;

            static constexpr bool kIsFp4 = HasFp4E2M1Codes<TPolicy>;
            static constexpr std::size_t kGroup = [] {
                if constexpr ( HasFp4E2M1Codes<TPolicy> )
                {
                    return static_cast<std::size_t>( TPolicy::kQuantizationGroupSize );
                } else {
                    return std::size_t{ 0 };
                }
            }();

            static QuantizedWeights quantize( std::span<const std::uint16_t> bf16, std::size_t outputs, std::size_t inputs )
            {
                return kIsFp4 ? quantizeFp4( bf16, outputs, inputs, kGroup ) : quantizeFp8( bf16, outputs, inputs );
            }

            static std::vector<double> decode( const QuantizedWeights& weights, std::size_t outputs, std::size_t inputs )
            {
                return kIsFp4 ? decodeFp4( weights, outputs, inputs, kGroup ) : decodeFp8( weights, outputs, inputs );
            }

            static shape_t codeShape( dim_t outputs, dim_t inputs )
            {
                return kIsFp4 ? shape_t{ outputs, inputs / 2 } : shape_t{ outputs, inputs };
            }

            static shape_t scaleShape( dim_t outputs, dim_t inputs )
            {
                return kIsFp4 ? shape_t{ outputs, inputs / static_cast<dim_t>( kGroup ) } : shape_t{ outputs };
            }

            static TensorDataType codeType()
            {
                return kIsFp4 ? TensorDataType::UINT8 : TensorDataType::FP8_E4M3;
            }
        };

        using Cases = ::testing::Types<
            Case<TensorDataType::FP16, PerChannelFp8<>>,
            Case<TensorDataType::BF16, PerChannelFp8<>>,
            Case<TensorDataType::FP16, PerGroupFp4<128>>,
            Case<TensorDataType::BF16, PerGroupFp4<128>>,
            Case<TensorDataType::FP16, PerGroupFp4<64>>,
            Case<TensorDataType::BF16, PerGroupFp4<64>>>;

        struct CaseNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                const std::string precision = T::kPrecision == TensorDataType::FP16 ? "Fp16" : "Bf16";

                return precision + ( T::kIsFp4 ? "Fp4Group" + std::to_string( T::kGroup ) : std::string( "Fp8" ) );
            }
        };

        TensorBlobView bytesBlob( std::span<const std::uint8_t> bytes, const shape_t& shape, TensorDataType data_type )
        {
            TensorMetadata metadata;
            metadata.dtype = data_type;
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

        /// Normal weights, a row of zeros, and a row with one large weight, so that every other weight in it rounds near zero.
        std::vector<std::uint16_t> testWeights( dim_t outputs, dim_t inputs, std::uint32_t seed )
        {
            std::vector<std::uint16_t> bits = randomBf16( static_cast<std::size_t>( outputs * inputs ), seed, 0.02f );

            if ( outputs > 2 )
            {
                std::fill_n( bits.begin() + inputs, inputs, std::uint16_t{ 0 } );
                bits[static_cast<std::size_t>( 2 * inputs + 3 )] = toBf16Bits( 7.5f );
            }

            return bits;
        }

        /// How many elements differ, and the first that does, so that a failure says where.
        template<typename T>
        std::string differences( std::span<const T> actual, std::span<const T> expected )
        {
            std::size_t count = 0;
            std::size_t first = 0;

            for ( std::size_t i = 0; i < expected.size(); ++i )
            {
                if ( std::memcmp( &actual[i], &expected[i], sizeof( T ) ) != 0 )
                {
                    first = count == 0 ? i : first;
                    ++count;
                }
            }

            if ( count == 0 )
            {
                return {};
            }

            if constexpr ( std::is_same_v<T, float> )
            {
                return std::format( "{} of {} differ, first at {}: {:a} against {:a}", count, expected.size(), first, actual[first], expected[first] );
            } else {
                return std::format( "{} of {} differ, first at {}: {:#04x} against {:#04x}", count, expected.size(), first, actual[first], expected[first] );
            }
        }

        enum class Load { StoredQuantized, QuantizedAtLoad };

        template<typename TCase>
        struct Fixture
        {
            static constexpr TensorDataType kPrecision = TCase::kPrecision;

            using LinearType = Mila::Dnn::Linear<DeviceType::Metal, kPrecision, typename TCase::Policy>;
            using TensorType = Tensor<kPrecision, MetalMemoryResource>;

            dim_t inputs;
            dim_t outputs;
            std::vector<std::uint16_t> weight_bits;
            QuantizedWeights quantized;
            std::vector<double> weight_values;
            std::vector<std::uint16_t> bias_bits;
            std::unique_ptr<LinearType> linear;

            Fixture( dim_t input_features, dim_t output_features, dim_t built_rows, Load load, bool with_bias = false )
                : inputs( input_features ), outputs( output_features ),
                  weight_bits( testWeights( output_features, input_features, 31 ) ),
                  quantized( TCase::quantize( weight_bits, static_cast<std::size_t>( output_features ), static_cast<std::size_t>( input_features ) ) ),
                  weight_values( TCase::decode( quantized, static_cast<std::size_t>( output_features ), static_cast<std::size_t>( input_features ) ) )
            {
                LinearConfig config( input_features, output_features );
                config.withBias( with_bias );

                linear = std::make_unique<LinearType>( "linear", config, Device::Metal( 0 ) );
                linear->build( BuildContext( shape_t{ built_rows, input_features }, RuntimeMode::Inference, false ) );

                if ( load == Load::StoredQuantized )
                {
                    linear->loadParameter( "weight", bytesBlob( quantized.codes, TCase::codeShape( outputs, inputs ), TCase::codeType() ) );
                    linear->loadParameter( "weight_scale", floatsBlob( quantized.scales, TCase::scaleShape( outputs, inputs ) ) );
                } else {
                    linear->loadParameter( "weight", blobOver( weight_bits, shape_t{ outputs, inputs } ) );
                }

                if ( with_bias )
                {
                    bias_bits = randomBf16( static_cast<std::size_t>( outputs ), 12, 0.5f );
                    linear->loadParameter( "bias", blobOver( bias_bits, shape_t{ outputs } ) );
                }
            }

            std::vector<float> run( dim_t rows )
            {
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                TensorType input = metalTensor<kPrecision>( input_values, shape_t{ rows, inputs } );

                const TensorType& output = linear->forward( input );
                linear->synchronize();

                const std::vector<float> all = readFloats<kPrecision>( output );

                return std::vector<float>( all.begin(), all.begin() + rows * outputs );
            }

            /// Run `rows` rows, and return the error against the reference on the codes and scales as stored.
            double errorAt( dim_t rows )
            {
                const std::vector<float> actual = run( rows );
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                const std::vector<double> stored_input = asStored<kPrecision>( std::span<const float>( input_values ) );

                std::vector<double> reference( static_cast<std::size_t>( rows * outputs ) );

                for ( dim_t r = 0; r < rows; ++r )
                {
                    for ( dim_t c = 0; c < outputs; ++c )
                    {
                        double sum = bias_bits.empty() ? 0.0 : asStored<kPrecision>( fromBf16Bits( bias_bits[static_cast<std::size_t>( c )] ) );

                        for ( dim_t k = 0; k < inputs; ++k )
                        {
                            sum += stored_input[static_cast<std::size_t>( r * inputs + k )] * weight_values[static_cast<std::size_t>( c * inputs + k )];
                        }

                        reference[static_cast<std::size_t>( r * outputs + c )] = sum;
                    }
                }

                return relativeError( actual, reference );
            }
        };

        template<typename TCase>
        class LinearQuantizedMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LinearQuantizedMetal, Cases, CaseNames );

        // Every path and every seam between them: own lanes for one row and for several, FP8's bare codes in tiles of 128 and of 8
        // with rows left over, and expanded weights from 64 rows for FP4 and 1024 for FP8.
        constexpr dim_t kEveryRowCount[] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 63, 64, 65, 127, 128, 129, 130, 135, 136, 143, 256, 1023, 1024, 1031 };
    }

    TYPED_TEST( LinearQuantizedMetal, MatchesTheReferenceAtEveryRowCount )
    {
        Fixture<TypeParam> fixture( 256, 48, 1031, Load::StoredQuantized );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearQuantizedMetal, MatchesTheReferenceWhenOutputsAreOffATileOrABlock )
    {
        // 1030 outputs are a block of 1024 expanded and 6 more, and leave 6 to the own lanes after FP8's tiles.
        for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 17 }, dim_t{ 35 }, dim_t{ 1030 } } )
        {
            Fixture<TypeParam> fixture( 256, outputs, 1024, Load::StoredQuantized );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 8 }, dim_t{ 19 }, dim_t{ 64 }, dim_t{ 130 }, dim_t{ 1024 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << outputs << " outputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearQuantizedMetal, MatchesTheReferenceWithABias )
    {
        Fixture<TypeParam> fixture( 256, 48, 1031, Load::StoredQuantized, true );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearQuantizedMetal, AnEarlierCallLeavesNoTraceInALaterOne )
    {
        Fixture<TypeParam> fixture( 256, 48, 1024, Load::StoredQuantized );

        for ( const dim_t rows : { dim_t{ 1024 }, dim_t{ 1 }, dim_t{ 136 }, dim_t{ 7 }, dim_t{ 64 }, dim_t{ 1 }, dim_t{ 1024 } } )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearQuantizedMetal, QuantizedAtLoadGivesWhatStoredQuantizedGives )
    {
        Fixture<TypeParam> stored( 512, 96, 1024, Load::StoredQuantized );
        Fixture<TypeParam> at_load( 512, 96, 1024, Load::QuantizedAtLoad );

        for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 9 }, dim_t{ 130 }, dim_t{ 1024 } } )
        {
            EXPECT_EQ( at_load.run( rows ), stored.run( rows ) ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearQuantizedMetal, QuantizeWritesTheBytesCudaWrites )
    {
        using OpType = Mila::Dnn::Compute::Metal::Linear::MetalLinearOp<TypeParam::kPrecision, typename TypeParam::Policy>;

        // 640 is two blocks of 256 and part of a third for FP8's threads, and ten groups of 64 or five of 128. The large weight in row 2
        // puts the rest of its row or group in E4M3's smallest codes, and a weight at max |w| lands past 448 by one rounding.
        const dim_t outputs = 6;
        const dim_t inputs = 640;
        const std::vector<std::uint16_t> bits = testWeights( outputs, inputs, 41 );
        const QuantizedWeights expected = TypeParam::quantize( bits, outputs, inputs );

        MetalExecutionContext context( Device::Metal( 0 ) );
        LinearConfig config( inputs, outputs );
        config.withBias( false );
        OpType op( &context, config );

        Tensor<TypeParam::kIsFp4 ? TensorDataType::UINT8 : TensorDataType::FP8_E4M3, MetalMemoryResource> codes(
            Device::Metal( 0 ), TypeParam::codeShape( outputs, inputs ), "linear.weight" );
        Tensor<TensorDataType::FP32, MetalMemoryResource> scales( Device::Metal( 0 ), TypeParam::scaleShape( outputs, inputs ), "linear.weight.scales" );

        op.quantize( blobOver( bits, shape_t{ outputs, inputs } ), codes, scales, shape_t{ outputs, inputs } );

        const auto* const code_bytes = static_cast<const std::uint8_t*>( static_cast<const ITensor&>( codes ).rawData() );
        const auto* const scale_values = static_cast<const float*>( static_cast<const ITensor&>( scales ).rawData() );

        EXPECT_EQ( differences<std::uint8_t>( std::span( code_bytes, expected.codes.size() ), expected.codes ), "" ) << "codes";
        EXPECT_EQ( differences<float>( std::span( scale_values, expected.scales.size() ), expected.scales ), "" ) << "scales";
    }

    TYPED_TEST( LinearQuantizedMetal, RefusesWeightsToQuantizeThatAreNotFiniteAndNamesThem )
    {
        for ( const float value : { std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() } )
        {
            Fixture<TypeParam> fixture( 256, 8, 8, Load::StoredQuantized );
            std::vector<std::uint16_t> bits = testWeights( 8, 256, 51 );
            bits[300] = toBf16Bits( value );

            try
            {
                fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 8, 256 } ) );
                FAIL() << "A weight of " << value << " was quantized";
            }
            catch ( const std::invalid_argument& error )
            {
                EXPECT_NE( std::string( error.what() ).find( "linear.weight" ), std::string::npos ) << error.what();
            }
        }
    }

    TYPED_TEST( LinearQuantizedMetal, RefusesInputsItCannotReadInChunks )
    {
        using LinearType = typename Fixture<TypeParam>::LinearType;

        const dim_t inputs = TypeParam::kIsFp4 ? 192 + 32 : 252;

        LinearConfig config( inputs, 16 );
        config.withBias( false );
        LinearType linear( "linear", config, Device::Metal( 0 ) );

        EXPECT_THROW( linear.build( BuildContext( shape_t{ 8, inputs }, RuntimeMode::Inference, false ) ), std::exception );
    }

    TYPED_TEST( LinearQuantizedMetal, TheFootprintsScratchIsWhatItAsksFor )
    {
        constexpr TensorDataType kPrecision = TypeParam::kPrecision;

        for ( const dim_t built : { dim_t{ 1 }, dim_t{ 7 }, dim_t{ 64 }, dim_t{ 200 }, dim_t{ 1100 } } )
        {
            Fixture<TypeParam> fixture( 512, 2056, built, Load::StoredQuantized );

            LinearConfig config( 512, 2056 );
            config.withBias( false );
            typename Fixture<TypeParam>::LinearType predictor( "predictor", config, Device::Metal( 0 ) );

            const BuildContext context = BuildContext( shape_t{ built, 512 }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );
            const std::size_t asked = fixture.linear->getMemoryStats().device_scratch_bytes;

            EXPECT_EQ( predictor.getRequiredMemory( context ).device_scratch_bytes, asked ) << built << " rows";

            // A request past the reservation throws, so a run at each path holds the footprint to what is asked.
            fixture.linear->getExecutionContext()->reserveScratch( asked );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 63 }, dim_t{ 64 }, dim_t{ 200 }, dim_t{ 1023 }, dim_t{ 1024 }, dim_t{ 1100 } } )
            {
                if ( rows > built )
                {
                    continue;
                }

                const std::vector<float> values( static_cast<std::size_t>( rows * 512 ), 1.0f );
                auto input = metalTensor<kPrecision>( values, shape_t{ rows, 512 } );

                EXPECT_NO_THROW( fixture.linear->forward( input ) ) << rows << " of " << built << " rows";
                fixture.linear->synchronize();
            }
        }
    }

    /// A tensor past the load's 256 MiB staging limit is quantized in blocks of rows, each landing where its rows belong.
    TEST( LinearQuantizedMetalStaging, ATensorPastTheLimitIsQuantizedInBlocks )
    {
        const auto quantizesLikeCuda = []<typename TPolicy>()
        {
            using TCase = Case<TensorDataType::FP16, TPolicy>;

            // 34,000 rows of 4,096 are 278 MB of BF16, so the second block starts at row 32,768.
            const dim_t outputs = 34000;
            const dim_t inputs = 4096;

            std::vector<std::uint16_t> bits( static_cast<std::size_t>( outputs * inputs ) );

            for ( std::size_t i = 0; i < bits.size(); ++i )
            {
                // Each row its own spread, so that a scale or code written to another row's place shows.
                const float row_scale = 0.01f * static_cast<float>( 1 + ( i / inputs ) % 97 );
                bits[i] = toBf16Bits( row_scale * static_cast<float>( static_cast<int>( ( i * 2654435761u ) % 2001 ) - 1000 ) / 1000.0f );
            }

            const QuantizedWeights expected = TCase::quantize( bits, outputs, inputs );

            MetalExecutionContext context( Device::Metal( 0 ) );
            LinearConfig config( inputs, outputs );
            config.withBias( false );
            Mila::Dnn::Compute::Metal::Linear::MetalLinearOp<TensorDataType::FP16, TPolicy> op( &context, config );

            Tensor<TCase::kIsFp4 ? TensorDataType::UINT8 : TensorDataType::FP8_E4M3, MetalMemoryResource> codes(
                Device::Metal( 0 ), TCase::codeShape( outputs, inputs ), "linear.weight" );
            Tensor<TensorDataType::FP32, MetalMemoryResource> scales( Device::Metal( 0 ), TCase::scaleShape( outputs, inputs ), "linear.weight.scales" );

            op.quantize( blobOver( bits, shape_t{ outputs, inputs } ), codes, scales, shape_t{ outputs, inputs } );

            const auto* const code_bytes = static_cast<const std::uint8_t*>( static_cast<const ITensor&>( codes ).rawData() );
            const auto* const scale_values = static_cast<const float*>( static_cast<const ITensor&>( scales ).rawData() );

            EXPECT_EQ( differences<std::uint8_t>( std::span( code_bytes, expected.codes.size() ), expected.codes ), "" ) << "codes";
            EXPECT_EQ( differences<float>( std::span( scale_values, expected.scales.size() ), expected.scales ), "" ) << "scales";
        };

        quantizesLikeCuda.template operator()<PerChannelFp8<>>();
        quantizesLikeCuda.template operator()<PerGroupFp4<128>>();
    }

    TEST( LinearQuantizedMetalRows, FormatsExistAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, PerChannelFp8<>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, PerChannelFp8<>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, PerGroupFp4<128>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, PerGroupFp4<64>> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, PerChannelFp8<>> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, PerGroupFp4<128>> ) );
    }

    /// Llama 3.2 3B's own weights: Metal's quantize of the BF16 weights against what `ExportArtifact` wrote on CUDA.
    TEST( LinearQuantizedMetalLlama32, QuantizesAsExportArtifactDid )
    {
        namespace fs = std::filesystem;

        const fs::path folder = fs::path( TEST_DATA_DIR ) / "models" / "llama";
        const fs::path bf16 = folder / "llama32_3b_instruct_bf16.bin";

        if ( !fs::exists( bf16 ) )
        {
            GTEST_SKIP() << "Llama 3.2 3B's BF16 weights are not present at " << folder.string();
        }

        Serialization::WeightsReader source( bf16 );
        int compared = 0;

        const auto compareFile = [&]<typename TPolicy>( const fs::path& path )
        {
            if ( !fs::exists( path ) )
            {
                return;
            }

            Serialization::WeightsReader exported( path );

            for ( const std::string name : { "tf_layer_0.fc_qkv_proj", "tf_layer_0.fc_down", "tf_layer_27.fc_gate_up" } )
            {
                auto weights = source.readTensorBlob<CpuMemoryResource>( name + ".weight" );
                auto codes = exported.readTensorBlob<CpuMemoryResource>( name + ".weight" );
                auto scales = exported.readTensorBlob<CpuMemoryResource>( name + ".weight_scale" );

                const shape_t shape = weights.getMetadata().shape;

                MetalExecutionContext context( Device::Metal( 0 ) );
                LinearConfig config( shape[1], shape[0] );
                config.withBias( false );
                Mila::Dnn::Compute::Metal::Linear::MetalLinearOp<TensorDataType::FP16, TPolicy> op( &context, config );

                Tensor<TPolicy::kStorageDtype, MetalMemoryResource> metal_codes( Device::Metal( 0 ), codes.getMetadata().shape, name + ".weight" );
                Tensor<TensorDataType::FP32, MetalMemoryResource> metal_scales( Device::Metal( 0 ), scales.getMetadata().shape, name + ".scales" );

                op.quantize( weights, metal_codes, metal_scales, shape );

                const auto* const code_bytes = static_cast<const std::uint8_t*>( static_cast<const ITensor&>( metal_codes ).rawData() );
                const auto* const scale_values = static_cast<const float*>( static_cast<const ITensor&>( metal_scales ).rawData() );

                EXPECT_EQ( differences<std::uint8_t>( std::span( code_bytes, codes.sizeBytes() ),
                    std::span( static_cast<const std::uint8_t*>( codes.data() ), codes.sizeBytes() ) ), "" )
                    << path.filename().string() << " " << name << " codes";
                EXPECT_EQ( differences<float>( std::span( scale_values, scales.sizeBytes() / 4 ),
                    std::span( static_cast<const float*>( scales.data() ), scales.sizeBytes() / 4 ) ), "" )
                    << path.filename().string() << " " << name << " scales";

                ++compared;
            }
        };

        compareFile.template operator()<PerGroupFp4<128>>( folder / "llama32_3b_instruct_fp4.safetensors" );
        compareFile.template operator()<PerChannelFp8<>>( folder / "llama32_3b_instruct_fp8.safetensors" );

        if ( compared == 0 )
        {
            GTEST_SKIP() << "No exported FP8 or FP4 weights of Llama 3.2 3B are present at " << folder.string();
        }
    }
}
