/**
 * @file CudaLinearOp.Int6.Cuda.cpp
 * @brief The CUDA INT6 quantizer produces exactly the codes and scale bits of the normative codec (Int6Packing.ixx),
 * and the decode matvec and the batched GEMM both compute with the exact decoded weights.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

import Mila;
import Serialization.Tensor;
import Dnn.Quantization.Weight.CodebookPacking;
import Dnn.Quantization.Weight.Int6Packing;
import Dnn.Quantization.Weight.Policies;
import Compute.CudaLinearOp;

namespace Mila::Tests::Dnn::Quantization
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Cuda::Linear;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        using Int6Op = CudaLinearOp<TensorDataType::BF16, PerGroupInt6<32>>;
        using Int6Linear = Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupInt6<32>>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        constexpr dim_t kGroup = 32;

        std::unique_ptr<IExecutionContext> makeContextOrSkip()
        {
            try {
                return createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& ) {
                return nullptr;
            }
        }

        std::uint16_t toBf16Bits( float value )
        {
            std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
            bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

            return static_cast<std::uint16_t>( bits >> 16 );
        }

        float fromBf16Bits( std::uint16_t bits )
        {
            return std::bit_cast<float>( static_cast<std::uint32_t>( bits ) << 16 );
        }

        float roundThroughBf16( float value )
        {
            return fromBf16Bits( toBf16Bits( value ) );
        }

        void setGroup( std::vector<std::uint16_t>& weights, dim_t columns, dim_t row, dim_t group,
            std::initializer_list<float> leading, float fill )
        {
            std::uint16_t* first = weights.data() + row * columns + group * kGroup;

            for ( dim_t index = 0; index < kGroup; ++index )
                first[ index ] = toBf16Bits( fill );

            dim_t index = 0;

            for ( const float value : leading )
                first[ index++ ] = toBf16Bits( value );
        }
    }

    TEST( CudaLinearOpInt6, QuantizeMatchesTheCodecBitForBit )
    {
        auto context = makeContextOrSkip();

        if ( !context )
            GTEST_SKIP() << "no CUDA device";

        // 122 groups per row: not a multiple of the kernel's eight groups per block, so the last block of
        // every row runs part-empty.
        const dim_t rows = 97;
        const dim_t columns = 122 * kGroup;

        std::mt19937 generator( 20261002u );
        std::normal_distribution<float> distribution( 0.0f, 0.02f );
        std::vector<std::uint16_t> weights( static_cast<std::size_t>( rows * columns ) );

        for ( auto& weight : weights )
            weight = toBf16Bits( distribution( generator ) );

        // Groups whose code or scale bits turn on a detail of the rule.
        setGroup( weights, columns, 0, 0, {}, 0.0f );                      // all zero
        setGroup( weights, columns, 0, 1, {}, -0.0f );                     // all negative zero: still d = -0
        setGroup( weights, columns, 0, 2, { 0.5f, -0.5f, 0.25f }, 0.0f ); // magnitude tie, positive first
        setGroup( weights, columns, 0, 3, { -0.5f, 0.5f, 0.25f }, 0.0f ); // magnitude tie, negative first
        setGroup( weights, columns, 1, 5, { 1.0e-6f, -3.0e-7f }, 0.0f );   // scale below FP16's normal range
        setGroup( weights, columns, rows - 1, 121, { 3.0f, -3.0f }, 1.0f ); // last group of the last row

        std::vector<float> reference_input( weights.size() );

        for ( std::size_t index = 0; index < weights.size(); ++index )
            reference_input[ index ] = fromBf16Bits( weights[ index ] );

        const dim_t row_bytes = packedRowBytesForSixBitCodes( columns );
        std::vector<std::uint8_t> expected_codes( static_cast<std::size_t>( rows * row_bytes ) );
        std::vector<std::uint16_t> expected_scales( static_cast<std::size_t>( rows * columns / kGroup ) );
        quantizeInt6( reference_input.data(), rows, columns, kGroup, expected_codes.data(), expected_scales.data() );

        LinearConfig config( columns, rows );
        config.withBias( false );
        Int6Op op( context.get(), config );

        Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> packed( Device::Cuda( 0 ), shape_t{ rows, row_bytes } );
        Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> scales( Device::Cuda( 0 ), shape_t{ rows, columns / kGroup } );

        const std::size_t weight_bytes = weights.size() * sizeof( std::uint16_t );
        Serialization::TensorMetadata meta{ TensorDataType::BF16, shape_t{ rows, columns }, weight_bytes };
        Serialization::TensorBlobView blob( meta, weights.data(), weight_bytes );

        op.quantize( blob, packed, scales, shape_t{ rows, columns } );
        context->synchronize();

        std::vector<std::uint8_t> device_codes( expected_codes.size() );
        std::vector<std::uint16_t> device_scales( expected_scales.size() );
        ASSERT_EQ( cudaMemcpy( device_codes.data(), packed.rawData(), device_codes.size(), cudaMemcpyDeviceToHost ),
            cudaSuccess );
        ASSERT_EQ( cudaMemcpy( device_scales.data(), scales.rawData(),
            device_scales.size() * sizeof( std::uint16_t ), cudaMemcpyDeviceToHost ), cudaSuccess );

        // The special groups first, so a failure names the rule it breaks.
        EXPECT_EQ( device_scales[ 0 ], 0x8000 );
        EXPECT_EQ( device_scales[ 1 ], 0x8000 );
        EXPECT_EQ( halfBitsToFloat( device_scales[ 2 ] ), -0.015625f );
        EXPECT_EQ( halfBitsToFloat( device_scales[ 3 ] ), 0.015625f );

        std::size_t differing_codes = 0;
        std::size_t differing_scales = 0;

        for ( std::size_t index = 0; index < expected_codes.size(); ++index )
            differing_codes += device_codes[ index ] != expected_codes[ index ];

        for ( std::size_t index = 0; index < expected_scales.size(); ++index )
            differing_scales += device_scales[ index ] != expected_scales[ index ];

        EXPECT_EQ( differing_codes, 0u );
        EXPECT_EQ( differing_scales, 0u );
    }

    namespace
    {
        struct ForwardCase
        {
            dim_t rows;
            dim_t columns;
            std::vector<std::uint16_t> weightBits;
            std::vector<float> exactWeights;
        };

        ForwardCase makeForwardCase( dim_t rows, dim_t columns, unsigned seed )
        {
            ForwardCase forward_case{ rows, columns };
            std::mt19937 generator( seed );
            std::normal_distribution<float> distribution( 0.0f, 0.02f );

            forward_case.weightBits.resize( static_cast<std::size_t>( rows * columns ) );

            for ( auto& bits : forward_case.weightBits )
                bits = toBf16Bits( distribution( generator ) );

            std::vector<float> source( forward_case.weightBits.size() );

            for ( std::size_t index = 0; index < source.size(); ++index )
                source[ index ] = fromBf16Bits( forward_case.weightBits[ index ] );

            std::vector<std::uint8_t> codes( static_cast<std::size_t>( rows * packedRowBytesForSixBitCodes( columns ) ) );
            std::vector<std::uint16_t> scale_bits( static_cast<std::size_t>( rows * columns / kGroup ) );
            quantizeInt6( source.data(), rows, columns, kGroup, codes.data(), scale_bits.data() );

            forward_case.exactWeights.resize( source.size() );
            dequantizeInt6( codes.data(), scale_bits.data(), rows, columns, kGroup, forward_case.exactWeights.data() );

            return forward_case;
        }

        std::unique_ptr<Int6Linear> loadLinear( const ForwardCase& forward_case, dim_t built_rows )
        {
            LinearConfig config( forward_case.columns, forward_case.rows );
            config.withBias( false );

            auto linear = std::make_unique<Int6Linear>( "int6", config, Device::Cuda( 0 ) );
            linear->build( BuildContext( shape_t{ built_rows, forward_case.columns }, RuntimeMode::Inference, false ) );

            const std::size_t bytes = forward_case.weightBits.size() * sizeof( std::uint16_t );
            Serialization::TensorMetadata meta{ TensorDataType::BF16, shape_t{ forward_case.rows, forward_case.columns }, bytes };
            Serialization::TensorBlobView blob( meta, forward_case.weightBits.data(), bytes );

            linear->loadParameter( "weight", blob );
            linear->synchronize();

            return linear;
        }

        std::vector<float> runForward( Int6Linear& linear, IExecutionContext* context, const std::vector<float>& input,
            dim_t batch, dim_t columns )
        {
            HostFp32 host_input( Device::Cpu(), shape_t{ batch, columns } );
            std::copy( input.begin(), input.end(), host_input.data() );

            DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ batch, columns } );
            copy( host_input, device_input, context );
            context->synchronize();

            const auto& device_output = linear.forward( device_input );
            linear.synchronize();

            auto host_output = toHost<TensorDataType::FP32>( device_output, context );
            context->synchronize();

            return std::vector<float>( host_output.data(), host_output.data() + host_output.size() );
        }

        struct Reference
        {
            double value;
            double magnitude;
        };

        Reference dot( const float* x, const float* w, dim_t columns )
        {
            Reference reference{ 0.0, 0.0 };

            for ( dim_t column = 0; column < columns; ++column )
            {
                const double product = static_cast<double>( x[ column ] ) * static_cast<double>( w[ column ] );
                reference.value += product;
                reference.magnitude += std::abs( product );
            }

            return reference;
        }

        // The BF16 output store, plus FP32 accumulation in any order.
        double tolerance( const Reference& reference )
        {
            return std::abs( reference.value ) * 0x1p-8 + reference.magnitude * 1e-5 + 1e-7;
        }

        // One row takes the decode matvec, more rows the batched GEMM; both must apply the exact weights to the
        // exact BF16 activations, so one reference serves both.
        void runAgainstExactWeights( dim_t batch, dim_t rows, dim_t columns, unsigned seed )
        {
            auto context = makeContextOrSkip();

            if ( !context )
                GTEST_SKIP() << "no CUDA device";

            const ForwardCase forward_case = makeForwardCase( rows, columns, seed );
            auto linear = loadLinear( forward_case, batch );

            std::mt19937 generator( seed + 1 );
            std::normal_distribution<float> distribution( 0.0f, 1.0f );
            std::vector<float> input( static_cast<std::size_t>( batch * columns ) );

            for ( auto& value : input )
                value = roundThroughBf16( distribution( generator ) );

            const std::vector<float> output = runForward( *linear, context.get(), input, batch, columns );

            for ( dim_t token = 0; token < batch; ++token )
            {
                for ( dim_t row = 0; row < rows; ++row )
                {
                    const Reference reference = dot(
                        input.data() + token * columns, forward_case.exactWeights.data() + row * columns, columns );

                    ASSERT_NEAR( output[ token * rows + row ], reference.value, tolerance( reference ) )
                        << "token " << token << " row " << row;
                }
            }
        }
    }

    TEST( CudaLinearOpInt6, DecodeMatchesExactWeightsOnTheNarrowLoadPath )
    {
        runAgainstExactWeights( 1, 200, 2816, 11u );
    }

    TEST( CudaLinearOpInt6, DecodeMatchesExactWeightsOnTheWideLoadPath )
    {
        runAgainstExactWeights( 1, 200, 15360, 12u );
    }

    // 37 rows run one partial row tile; 200 output channels a partial column tile.
    TEST( CudaLinearOpInt6, BatchedMatchesExactWeights )
    {
        runAgainstExactWeights( 37, 200, 3840, 13u );
    }

    // 300 rows span three row tiles, the last partial, at the Gemma 4 26B-A4B's model width.
    TEST( CudaLinearOpInt6, BatchedMatchesExactWeightsAcrossRowTiles )
    {
        runAgainstExactWeights( 300, 256, 2816, 14u );
    }

    // 2080 is whole groups but no whole 64-element tile, and the layer refuses it at build rather than at its first
    // forward.
    TEST( CudaLinearOpInt6, AWidthNoTileDividesIsRefusedAtBuild )
    {
        if ( !makeContextOrSkip() )
            GTEST_SKIP() << "no CUDA device";

        LinearConfig config( 2080, 64 );
        config.withBias( false );

        Int6Linear linear( "int6", config, Device::Cuda( 0 ) );

        EXPECT_THROW( linear.build( BuildContext( shape_t{ 4, 2080 }, RuntimeMode::Inference, false ) ), std::invalid_argument );
    }

    // The packed tensors the quantizer wrote load back as stored weights and compute the same outputs: the round trip
    // an exported package takes.
    TEST( CudaLinearOpInt6, StoredCodesAndScalesLoadAsWritten )
    {
        auto context = makeContextOrSkip();

        if ( !context )
            GTEST_SKIP() << "no CUDA device";

        const dim_t rows = 96;
        const dim_t columns = 2816;
        const dim_t batch = 5;
        const ForwardCase forward_case = makeForwardCase( rows, columns, 21u );
        auto quantized_on_load = loadLinear( forward_case, batch );

        std::vector<float> source( forward_case.weightBits.size() );

        for ( std::size_t index = 0; index < source.size(); ++index )
            source[ index ] = fromBf16Bits( forward_case.weightBits[ index ] );

        const dim_t row_bytes = packedRowBytesForSixBitCodes( columns );
        std::vector<std::uint8_t> codes( static_cast<std::size_t>( rows * row_bytes ) );
        std::vector<std::uint16_t> scale_bits( static_cast<std::size_t>( rows * columns / kGroup ) );
        quantizeInt6( source.data(), rows, columns, kGroup, codes.data(), scale_bits.data() );

        LinearConfig config( columns, rows );
        config.withBias( false );
        Int6Linear stored( "int6_stored", config, Device::Cuda( 0 ) );
        stored.build( BuildContext( shape_t{ batch, columns }, RuntimeMode::Inference, false ) );

        Serialization::TensorMetadata code_meta{ TensorDataType::UINT8, shape_t{ rows, row_bytes }, codes.size() };
        Serialization::TensorBlobView code_blob( code_meta, codes.data(), codes.size() );
        stored.loadParameter( "weight", code_blob );

        const std::size_t scale_bytes = scale_bits.size() * sizeof( std::uint16_t );
        Serialization::TensorMetadata scale_meta{ TensorDataType::FP16, shape_t{ rows, columns / kGroup }, scale_bytes };
        Serialization::TensorBlobView scale_blob( scale_meta, scale_bits.data(), scale_bytes );
        stored.loadParameter( "weight_scale", scale_blob );
        stored.synchronize();

        std::mt19937 generator( 22u );
        std::normal_distribution<float> distribution( 0.0f, 1.0f );
        std::vector<float> input( static_cast<std::size_t>( batch * columns ) );

        for ( auto& value : input )
            value = roundThroughBf16( distribution( generator ) );

        EXPECT_EQ( runForward( stored, context.get(), input, batch, columns ),
            runForward( *quantized_on_load, context.get(), input, batch, columns ) );
    }
}
