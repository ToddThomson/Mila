/**
 * @file Linear.Decode.Cuda.cpp
 * @brief Linear::decode over a few rows equals forward() one row at a time, for every weight format.
 *
 * forward() over one row runs the format's decode matvec, which the per-format tests prove against exact weights.
 * decode() over R rows runs one tensor-core product instead (Gemma4Mtp.md 4.7), summing in another order, so each row
 * is held to the one-row decode within the BF16 output store and an FP32 floor -- far inside what a structural error
 * (a wrong column map, scale or channel) moves an output by, which is the size of the output itself.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <chrono>
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
import Dnn.Quantization.Weight.Policies;

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        template<typename TPolicy>
        using CudaLinear = Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, TPolicy>;

        constexpr dim_t kBuiltRows = 8;

        bool deviceAvailable()
        {
            int count = 0;

            return cudaGetDeviceCount( &count ) == cudaSuccess && count > 0;
        }

        std::uint16_t toBf16Bits( float value )
        {
            std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
            bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

            return static_cast<std::uint16_t>( bits >> 16 );
        }

        void loadBlob( auto& linear, const char* name, TensorDataType dtype, const shape_t& shape, const void* bytes,
            std::size_t byte_count )
        {
            Serialization::TensorMetadata meta{ dtype, shape, byte_count };
            Serialization::TensorBlobView blob( meta, bytes, byte_count );
            linear.loadParameter( name, blob );
        }

        /// BF16 weights from N(0, 0.02), quantized by the layer's own load path for every quantized policy.
        template<typename TPolicy>
        std::unique_ptr<CudaLinear<TPolicy>> loadQuantizedOnLoad( dim_t out_features, dim_t in_features, bool has_bias,
            unsigned seed )
        {
            LinearConfig config( in_features, out_features );
            config.withBias( has_bias );

            auto linear = std::make_unique<CudaLinear<TPolicy>>( "decode", config, Device::Cuda( 0 ) );
            linear->build( BuildContext( shape_t{ kBuiltRows, in_features }, RuntimeMode::Inference, false ) );

            std::mt19937 generator( seed );
            std::normal_distribution<float> distribution( 0.0f, 0.02f );
            std::vector<std::uint16_t> weights( static_cast<std::size_t>( out_features * in_features ) );

            for ( auto& bits : weights )
                bits = toBf16Bits( distribution( generator ) );

            loadBlob( *linear, "weight", TensorDataType::BF16, shape_t{ out_features, in_features }, weights.data(),
                weights.size() * sizeof( std::uint16_t ) );

            if ( has_bias )
            {
                std::vector<std::uint16_t> bias( static_cast<std::size_t>( out_features ) );

                for ( auto& bits : bias )
                    bits = toBf16Bits( 5.0f * distribution( generator ) );

                loadBlob( *linear, "bias", TensorDataType::BF16, shape_t{ out_features }, bias.data(),
                    bias.size() * sizeof( std::uint16_t ) );
            }

            linear->synchronize();

            return linear;
        }

        /// Random codes, scales and table, packed by the normative codec and loaded as an exported model is.
        template<typename TPolicy>
        std::unique_ptr<CudaLinear<TPolicy>> loadCodebook( dim_t out_features, dim_t in_features, unsigned seed )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;
            constexpr int kEntries = TPolicy::kCodebookEntries;

            LinearConfig config( in_features, out_features );
            config.withBias( false );

            auto linear = std::make_unique<CudaLinear<TPolicy>>( "decode", config, Device::Cuda( 0 ) );
            linear->build( BuildContext( shape_t{ kBuiltRows, in_features }, RuntimeMode::Inference, false ) );

            std::mt19937 generator( seed );
            std::uniform_int_distribution<int> code_distribution( 0, kEntries - 1 );
            std::uniform_real_distribution<float> scale_distribution( 0.001f, 0.03f );
            std::uniform_real_distribution<float> entry_distribution( -1.0f, 1.0f );

            std::vector<std::uint8_t> codes( static_cast<std::size_t>( out_features * in_features ) );

            for ( auto& code : codes )
                code = static_cast<std::uint8_t>( code_distribution( generator ) );

            std::vector<std::uint16_t> scales( static_cast<std::size_t>( out_features * groupsPerRow( in_features, kGroupSize ) ) );

            for ( auto& scale : scales )
                scale = floatToHalfBits( scale_distribution( generator ) );

            // Entries a BF16 value cannot carry exactly, so the kernel's two-part table is what is under test.
            std::vector<float> codebook( static_cast<std::size_t>( kEntries ) );

            for ( auto& entry : codebook )
                entry = entry_distribution( generator );

            std::vector<std::uint8_t> plane_two( static_cast<std::size_t>( out_features * packedRowBytesForTwoBitCodes( in_features ) ) );
            std::vector<std::uint8_t> plane_one;

            if constexpr ( TPolicy::kHasHighBitPlane )
            {
                plane_one.resize( static_cast<std::size_t>( out_features * packedRowBytesForOneBitPlane( in_features ) ) );
                packThreeBitCodes( codes.data(), out_features, in_features, plane_two.data(), plane_one.data() );
            }
            else
            {
                packTwoBitCodes( codes.data(), out_features, in_features, plane_two.data() );
            }

            loadBlob( *linear, "weight", TensorDataType::UINT8, shape_t{ out_features, in_features / 4 }, plane_two.data(),
                plane_two.size() );

            if constexpr ( TPolicy::kHasHighBitPlane )
            {
                loadBlob( *linear, "weight_high_plane", TensorDataType::UINT8, shape_t{ out_features, in_features / 8 },
                    plane_one.data(), plane_one.size() );
            }

            loadBlob( *linear, "weight_scale", TensorDataType::FP16,
                shape_t{ out_features, groupsPerRow( in_features, kGroupSize ) }, scales.data(),
                scales.size() * sizeof( std::uint16_t ) );
            loadBlob( *linear, "weight_codebook", TensorDataType::FP32, shape_t{ static_cast<dim_t>( kEntries ) },
                codebook.data(), codebook.size() * sizeof( float ) );

            linear->synchronize();

            return linear;
        }

        template<typename TLinear>
        std::vector<float> toHostVector( TLinear& linear, const DeviceBf16& device, IExecutionContext* context )
        {
            linear.synchronize();
            auto host = toHost<TensorDataType::FP32>( device, context );
            context->synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        /**
         * For each row count, decode() over R random rows against forward() of each row alone. The budget per output is
         * one BF16 step of the one-row result (the two FP32 sums may round to neighbouring BF16 values) plus 0.2% of the
         * outputs' RMS for FP32 summation order and the codebook's two-part entries.
         */
        template<typename TLinear>
        void expectDecodeMatchesForward( TLinear& linear, dim_t in_features, dim_t out_features, unsigned seed )
        {
            auto context = createExecutionContext( Device::Cuda( 0 ) );

            std::mt19937 generator( seed );
            std::normal_distribution<float> distribution( 0.0f, 1.0f );

            for ( const dim_t rows : { dim_t( 2 ), dim_t( 3 ), dim_t( 5 ), dim_t( 8 ) } )
            {
                HostFp32 host_input( Device::Cpu(), shape_t{ rows, in_features } );

                for ( dim_t index = 0; index < rows * in_features; ++index )
                    host_input.data()[ index ] = distribution( generator );

                DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ rows, in_features } );
                copy( host_input, device_input, context.get() );
                context->synchronize();

                const std::vector<float> decoded = toHostVector( linear, linear.decode( device_input ), context.get() );
                ASSERT_EQ( decoded.size(), static_cast<std::size_t>( rows * out_features ) );

                std::vector<float> single( decoded.size() );

                for ( dim_t row = 0; row < rows; ++row )
                {
                    HostFp32 host_row( Device::Cpu(), shape_t{ 1, in_features } );
                    std::copy_n( host_input.data() + row * in_features, in_features, host_row.data() );

                    DeviceBf16 device_row( Device::Cuda( 0 ), shape_t{ 1, in_features } );
                    copy( host_row, device_row, context.get() );
                    context->synchronize();

                    const std::vector<float> output = toHostVector( linear, linear.forward( device_row ), context.get() );
                    std::copy( output.begin(), output.end(), single.begin() + row * out_features );
                }

                double square_sum = 0.0;

                for ( const float value : single )
                    square_sum += static_cast<double>( value ) * value;

                const double rms = std::sqrt( square_sum / single.size() );
                double largest = 0.0;
                int failures = 0;

                for ( std::size_t index = 0; index < single.size(); ++index )
                {
                    const double difference = std::abs( static_cast<double>( decoded[ index ] ) - single[ index ] );
                    largest = std::max( largest, difference );

                    if ( difference > std::abs( single[ index ] ) * 0x1p-7 + 2e-3 * rms && ++failures <= 8 )
                    {
                        ADD_FAILURE() << rows << " rows: row " << index / out_features << ", output "
                            << index % out_features << ": decode " << decoded[ index ] << ", forward " << single[ index ];
                    }
                }

                EXPECT_EQ( failures, 0 ) << rows << " rows";
                std::printf( "  %lld rows: largest difference %.2e of the outputs' RMS %.3f\n",
                    static_cast<long long>( rows ), largest / rms, rms );
            }
        }
    }

    // The Gemma 4 12B's tied head and projections are INT4 and INT6; 200 output channels leave a part-filled tile of 16.
    TEST( LinearDecodeCuda, Int4RowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerGroupInt4<32>>( 200, 3840, false, 21u );
        expectDecodeMatchesForward( *linear, 3840, 200, 22u );
    }

    TEST( LinearDecodeCuda, Int4RowsMatchOneRowForwardsAtTheFeedForwardWidth )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerGroupInt4<32>>( 64, 15360, false, 23u );
        expectDecodeMatchesForward( *linear, 15360, 64, 24u );
    }

    TEST( LinearDecodeCuda, Int6RowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerGroupInt6<32>>( 200, 3840, false, 25u );
        expectDecodeMatchesForward( *linear, 3840, 200, 26u );
    }

    TEST( LinearDecodeCuda, Fp4RowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerGroupFp4<128>>( 200, 3072, false, 27u );
        expectDecodeMatchesForward( *linear, 3072, 200, 28u );
    }

    TEST( LinearDecodeCuda, Fp8RowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerChannelFp8<>>( 200, 3072, false, 29u );
        expectDecodeMatchesForward( *linear, 3072, 200, 30u );
    }

    // 3872 columns are 121 blocks of 32: the last chunk of two blocks is one block short.
    TEST( LinearDecodeCuda, Bf16RowsWithBiasMatchOneRowForwardsOverAnOddBlockCount )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<NoWeightQuant>( 200, 3872, true, 31u );
        expectDecodeMatchesForward( *linear, 3872, 200, 32u );
    }

    // 544 columns are 17 blocks of 32, an odd count again, on the format whose entries BF16 cannot carry.
    TEST( LinearDecodeCuda, TwoBitCodebookRowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadCodebook<PerGroupCodebook2<32>>( 200, 544, 33u );
        expectDecodeMatchesForward( *linear, 544, 200, 34u );
    }

    TEST( LinearDecodeCuda, ThreeBitCodebookRowsMatchOneRowForwards )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadCodebook<PerGroupCodebook3<64>>( 200, 3072, 35u );
        expectDecodeMatchesForward( *linear, 3072, 200, 36u );
    }

    TEST( LinearDecodeCuda, OneRowIsForwardsOwnDecode )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        auto linear = loadQuantizedOnLoad<PerGroupInt4<32>>( 200, 3840, false, 37u );
        auto context = createExecutionContext( Device::Cuda( 0 ) );

        HostFp32 host_input( Device::Cpu(), shape_t{ 1, 3840 } );
        std::mt19937 generator( 38u );
        std::normal_distribution<float> distribution( 0.0f, 1.0f );

        for ( dim_t index = 0; index < 3840; ++index )
            host_input.data()[ index ] = distribution( generator );

        DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ 1, 3840 } );
        copy( host_input, device_input, context.get() );
        context->synchronize();

        const std::vector<float> decoded = toHostVector( *linear, linear->decode( device_input ), context.get() );
        const std::vector<float> forwarded = toHostVector( *linear, linear->forward( device_input ), context.get() );

        ASSERT_EQ( decoded.size(), forwarded.size() );

        for ( std::size_t index = 0; index < decoded.size(); ++index )
            EXPECT_EQ( std::bit_cast<std::uint32_t>( decoded[ index ] ), std::bit_cast<std::uint32_t>( forwarded[ index ] ) )
                << "output " << index;
    }

    // Not part of the correctness gate. The Gemma 4 12B's gate and up projection, 63 MB at INT4 and so read from DRAM
    // rather than L2: decode() over R rows against one row's forward(), as Profiling/Microbenchmarks/VerifyRows.cu
    // measured the same product outside the library. Invoke with
    // --gtest_also_run_disabled_tests --gtest_filter=LinearDecodeCuda.DISABLED_RowsCostAgainstOneRow
    TEST( LinearDecodeCuda, DISABLED_RowsCostAgainstOneRow )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        constexpr dim_t kIn = 3840;
        constexpr dim_t kOut = 30720;
        constexpr int kIterations = 200;

        auto linear = loadQuantizedOnLoad<PerGroupInt4<32>>( kOut, kIn, false, 39u );

        const auto time = [&]( dim_t rows )
        {
            DeviceBf16 input( Device::Cuda( 0 ), shape_t{ rows, kIn } );
            zero( input );

            const auto run = [&] { return rows == 1 ? &linear->forward( input ) : &linear->decode( input ); };

            for ( int i = 0; i < 20; ++i )
                run();

            linear->synchronize();
            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < kIterations; ++i )
                run();

            linear->synchronize();

            return std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count() / kIterations;
        };

        const double one = time( 1 );
        std::printf( "  one row %.1f us\n", one );

        for ( dim_t rows = 2; rows <= 8; ++rows )
        {
            const double microseconds = time( rows );
            std::printf( "  %lld rows %.1f us = %.2f x one row\n", static_cast<long long>( rows ), microseconds, microseconds / one );
        }
    }

    TEST( LinearDecodeCuda, MoreThanEightRowsAreRefused )
    {
        if ( !deviceAvailable() )
            GTEST_SKIP() << "no CUDA device";

        LinearConfig config( 256, 64 );
        config.withBias( false );

        CudaLinear<NoWeightQuant> linear( "decode", config, Device::Cuda( 0 ) );
        linear.build( BuildContext( shape_t{ 16, 256 }, RuntimeMode::Inference, false ) );

        DeviceBf16 input( Device::Cuda( 0 ), shape_t{ 9, 256 } );

        EXPECT_THROW( linear.decode( input ), std::invalid_argument );
    }
}
