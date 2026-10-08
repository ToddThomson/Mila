/**
 * @file CausalConv1d.Cpu.cpp
 * @brief Concrete-component tests for CausalConv1d<DeviceType::Cpu, FP32>, undilated and dilated.
 *
 * The CPU operation is the reference the CUDA kernels are checked against, so it carries the same two proofs
 * the CUDA suite does -- a sequence fed in chunks, or a token at a time, equals the whole sequence -- at
 * dilation 1 and at Qwen 4's dilation 3, and the dilated output is pinned to torch.nn.Conv1d.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Convolutions
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using ConvCpu = Mila::Dnn::CausalConv1d<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr dim_t kChannels = 3;
        constexpr dim_t kKernelWidth = 4;

        // torch.nn.Conv1d( 3, 3, 4, groups = 3, dilation = 3, bias = False ) over 9 zero rows of left padding,
        // weight 0.25 * ( k + 1 ) - 0.1 * c, input x[ t, c ] = sin( 0.37 * ( t * 3 + c ) + 0.1 ), T = 13.
        constexpr dim_t kReferenceLength = 13;

        constexpr float kTorchDilatedReference[ kReferenceLength * kChannels ] = {
            9.98334140e-02f, 4.07597631e-01f, 5.95714509e-01f, 9.35616016e-01f, 8.99961829e-01f, 7.43167818e-01f,
            7.32231438e-01f, 3.92759144e-01f, 6.52017221e-02f, -2.09550649e-01f, -2.56296009e-01f, -2.75628686e-01f,
            -2.83465743e-01f, -2.32512072e-01f, -1.63622350e-01f, -4.25419807e-02f, 4.95176576e-02f, 1.30115539e-01f,
            2.95548916e-01f, 4.57703799e-01f, 5.02730012e-01f, 7.28796422e-01f, 5.96407056e-01f, 3.96993279e-01f,
            3.52586508e-01f, 7.26947337e-02f, -1.49674758e-01f, -3.90274763e-01f, -4.63825017e-01f, -4.92870390e-01f,
            -4.87958789e-01f, -3.95605683e-01f, -2.75309622e-01f, -4.36782837e-02f, 1.12003766e-01f, 2.48031154e-01f,
            4.49114710e-01f, 4.95213211e-01f, 4.95889425e-01f };

        float weightValue( dim_t c, dim_t k )
        {
            return 0.25f * static_cast<float>( k + 1 ) - 0.1f * static_cast<float>( c );
        }

        CausalConv1dConfig config( dim_t dilation, bool has_bias = false )
        {
            return CausalConv1dConfig( kChannels, kKernelWidth ).withDilation( dilation ).withBias( has_bias );
        }

        std::unique_ptr<ConvCpu> builtConv( dim_t dilation, dim_t batch, dim_t length, bool has_bias = false )
        {
            auto conv = std::make_unique<ConvCpu>( "conv", config( dilation, has_bias ), Device::Cpu() );
            conv->build( BuildContext( shape_t{ batch, length, kChannels }, RuntimeMode::Inference, false ) );

            auto parameters = conv->getParameters();
            float* weight = static_cast<float*>( parameters[ 0 ]->rawData() );

            for ( dim_t c = 0; c < kChannels; ++c )
            {
                for ( dim_t k = 0; k < kKernelWidth; ++k )
                {
                    weight[ c * kKernelWidth + k ] = weightValue( c, k );
                }
            }

            if ( has_bias )
            {
                float* bias = static_cast<float*>( parameters[ 1 ]->rawData() );

                for ( dim_t c = 0; c < kChannels; ++c )
                {
                    bias[ c ] = 0.5f + static_cast<float>( c );
                }
            }

            return conv;
        }

        TensorFp32 sineInput( dim_t batch, dim_t length )
        {
            TensorFp32 input( Device::Cpu(), shape_t{ batch, length, kChannels } );

            for ( dim_t i = 0; i < input.size(); ++i )
            {
                input.data()[ i ] = static_cast<float>( std::sin( 0.37 * static_cast<double>( i ) + 0.1 ) );
            }

            return input;
        }

        /// Rows [ first, first + count ) of every batch entry of @p source.
        TensorFp32 rows( const TensorFp32& source, dim_t first, dim_t count )
        {
            const dim_t batch = source.shape()[ 0 ];
            const dim_t length = source.shape()[ 1 ];
            TensorFp32 slice( Device::Cpu(), shape_t{ batch, count, kChannels } );

            for ( dim_t b = 0; b < batch; ++b )
            {
                for ( dim_t t = 0; t < count; ++t )
                {
                    for ( dim_t c = 0; c < kChannels; ++c )
                    {
                        slice.data()[ ( b * count + t ) * kChannels + c ] =
                            source.data()[ ( b * length + first + t ) * kChannels + c ];
                    }
                }
            }

            return slice;
        }

        /// Whole-sequence causal convolution with zero left padding, in double.
        std::vector<float> referenceConv( const TensorFp32& x, dim_t dilation, bool has_bias )
        {
            const dim_t batch = x.shape()[ 0 ];
            const dim_t length = x.shape()[ 1 ];
            std::vector<float> out( static_cast<size_t>( x.size() ), 0.0f );

            for ( dim_t b = 0; b < batch; ++b )
            {
                for ( dim_t t = 0; t < length; ++t )
                {
                    for ( dim_t c = 0; c < kChannels; ++c )
                    {
                        double accumulator = has_bias ? 0.5 + static_cast<double>( c ) : 0.0;

                        for ( dim_t k = 0; k < kKernelWidth; ++k )
                        {
                            const dim_t source_t = t - ( kKernelWidth - 1 - k ) * dilation;

                            if ( source_t >= 0 )
                            {
                                accumulator += static_cast<double>( weightValue( c, k ) )
                                    * x.data()[ ( b * length + source_t ) * kChannels + c ];
                            }
                        }

                        out[ static_cast<size_t>( ( b * length + t ) * kChannels + c ) ] = static_cast<float>( accumulator );
                    }
                }
            }

            return out;
        }
    }

    class CausalConv1dCpuTests : public ::testing::TestWithParam<dim_t>
    {
    };

    INSTANTIATE_TEST_SUITE_P( Dilations, CausalConv1dCpuTests, ::testing::Values( dim_t{ 1 }, dim_t{ 3 } ),
        []( const ::testing::TestParamInfo<dim_t>& info ) { return "Dilation" + std::to_string( info.param ); } );

    // ====================================================================
    // A. Forward against a host reference
    // ====================================================================

    TEST_P( CausalConv1dCpuTests, Prefill_MatchesHostReference )
    {
        const dim_t dilation = GetParam();
        auto conv = builtConv( dilation, 2, 11, /*has_bias*/ true );
        auto input = sineInput( 2, 11 );

        const auto& output = conv->prefill( input, 0 );
        const auto expected = referenceConv( input, dilation, true );

        for ( size_t i = 0; i < expected.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-6f ) << "at index " << i;
        }
    }

    TEST_P( CausalConv1dCpuTests, StateDepthIsKernelMinusOneTimesDilation )
    {
        const dim_t dilation = GetParam();
        auto conv = builtConv( dilation, 1, 4 );

        EXPECT_EQ( conv->makeStateSnapshot().shape(), ( shape_t{ 1, ( kKernelWidth - 1 ) * dilation, kChannels } ) );
    }

    // ====================================================================
    // B. The state equivalences
    // ====================================================================

    TEST_P( CausalConv1dCpuTests, ChunkedPrefillEqualsWholeSequence )
    {
        const dim_t dilation = GetParam();
        constexpr dim_t kLength = 15;
        constexpr dim_t kChunk = 5;

        auto input = sineInput( 2, kLength );
        auto whole = builtConv( dilation, 2, kLength );
        const auto& whole_out = whole->prefill( input, 0 );

        // Chunks of 5 are shorter than the dilated window of 9, so a chunk's left context reaches past
        // the chunk before it into older state rows.
        auto chunked = builtConv( dilation, 2, kChunk );

        for ( dim_t first = 0; first < kLength; first += kChunk )
        {
            auto chunk = rows( input, first, kChunk );
            const auto& chunk_out = chunked->prefill( chunk, first );

            for ( dim_t b = 0; b < 2; ++b )
            {
                for ( dim_t t = 0; t < kChunk; ++t )
                {
                    for ( dim_t c = 0; c < kChannels; ++c )
                    {
                        EXPECT_EQ( chunk_out.data()[ ( b * kChunk + t ) * kChannels + c ],
                            whole_out.data()[ ( b * kLength + first + t ) * kChannels + c ] )
                            << "batch " << b << " position " << first + t << " channel " << c;
                    }
                }
            }
        }
    }

    TEST_P( CausalConv1dCpuTests, PrefillThenTokenByTokenDecodeEqualsWholeSequence )
    {
        const dim_t dilation = GetParam();
        constexpr dim_t kLength = 13;
        constexpr dim_t kPrompt = 4;

        auto input = sineInput( 1, kLength );
        auto whole = builtConv( dilation, 1, kLength );
        const auto& whole_out = whole->prefill( input, 0 );

        auto stepped = builtConv( dilation, 1, kPrompt );
        auto prompt = rows( input, 0, kPrompt );
        const auto& prompt_out = stepped->prefill( prompt, 0 );

        for ( dim_t i = 0; i < kPrompt * kChannels; ++i )
        {
            EXPECT_EQ( prompt_out.data()[ i ], whole_out.data()[ i ] ) << "prompt index " << i;
        }

        for ( dim_t t = kPrompt; t < kLength; ++t )
        {
            auto token = rows( input, t, 1 );
            const auto& token_out = stepped->decode( token, t );

            for ( dim_t c = 0; c < kChannels; ++c )
            {
                EXPECT_EQ( token_out.data()[ c ], whole_out.data()[ t * kChannels + c ] )
                    << "position " << t << " channel " << c;
            }
        }
    }

    // Positive control: a second chunk convolved as a fresh sequence must differ at the boundary.
    TEST_P( CausalConv1dCpuTests, DiscardingTheStateChangesTheChunkBoundary )
    {
        const dim_t dilation = GetParam();
        constexpr dim_t kLength = 12;
        constexpr dim_t kChunk = 6;

        auto input = sineInput( 1, kLength );
        auto whole = builtConv( dilation, 1, kLength );
        const auto& whole_out = whole->prefill( input, 0 );

        auto fresh = builtConv( dilation, 1, kChunk );
        auto second = rows( input, kChunk, kChunk );
        const auto& fresh_out = fresh->prefill( second, 0 );

        float largest = 0.0f;

        for ( dim_t c = 0; c < kChannels; ++c )
        {
            largest = std::max( largest, std::fabs( fresh_out.data()[ c ] - whole_out.data()[ kChunk * kChannels + c ] ) );
        }

        EXPECT_GT( largest, 0.1f );
    }

    // ====================================================================
    // C. Dilation 3 against torch.nn.Conv1d (Qwen4.md Phase 1)
    // ====================================================================

    TEST( CausalConv1dCpuDilatedTests, DilationThreeMatchesTorchConv1d )
    {
        auto conv = builtConv( 3, 1, kReferenceLength );
        auto input = sineInput( 1, kReferenceLength );

        const auto& output = conv->prefill( input, 0 );

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], kTorchDilatedReference[ i ], 1e-6f ) << "at index " << i;
        }
    }

    TEST( CausalConv1dCpuDilatedTests, DilationOneDiffersFromTheTorchDilatedReference )
    {
        auto conv = builtConv( 1, 1, kReferenceLength );
        auto input = sineInput( 1, kReferenceLength );

        const auto& output = conv->prefill( input, 0 );

        float largest = 0.0f;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            largest = std::max( largest, std::fabs( output.data()[ i ] - kTorchDilatedReference[ i ] ) );
        }

        EXPECT_GT( largest, 0.1f );
    }

    // ====================================================================
    // D. Snapshot and restore across the dilated window
    // ====================================================================

    TEST( CausalConv1dCpuDilatedTests, RestoredSnapshotContinuesTheSequence )
    {
        constexpr dim_t kLength = 14;
        constexpr dim_t kPrompt = 10;

        auto input = sineInput( 1, kLength );
        auto whole = builtConv( 3, 1, kLength );
        const auto& whole_out = whole->prefill( input, 0 );

        auto source = builtConv( 3, 1, kPrompt );
        auto prompt = rows( input, 0, kPrompt );
        (void)source->prefill( prompt, 0 );

        auto snapshot = source->makeStateSnapshot();
        source->snapshotState( snapshot );

        auto restored = builtConv( 3, 1, kPrompt );
        restored->restoreState( snapshot );

        for ( dim_t t = kPrompt; t < kLength; ++t )
        {
            auto token = rows( input, t, 1 );
            const auto& token_out = restored->decode( token, t );

            for ( dim_t c = 0; c < kChannels; ++c )
            {
                EXPECT_EQ( token_out.data()[ c ], whole_out.data()[ t * kChannels + c ] )
                    << "position " << t << " channel " << c;
            }
        }
    }

    TEST( CausalConv1dCpuDilatedTests, DecodeBeforeAnyPrefillIsRefused )
    {
        auto conv = builtConv( 3, 1, 1 );
        auto token = sineInput( 1, 1 );

        EXPECT_THROW( (void)conv->decode( token, 0 ), std::logic_error );
    }
}
