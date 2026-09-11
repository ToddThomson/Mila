/**
 * @file CausalConv1d.Cuda.cpp
 * @brief Concrete-component tests for CausalConv1d<DeviceType::Cuda, {FP32, BF16}>.
 *
 * The convolution itself is four multiply-adds and would not need much proving. What needs
 * proving is its MEMORY: that a sequence fed in chunks, or one token at a time, produces
 * exactly what the whole sequence produces in one pass. Those two equivalences are the
 * reason the component holds state at all, and every test below that is not a guard-clause
 * check exists to pin one of them.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Convolutions
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-4f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using CausalConv1dPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class CausalConv1dCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using ConvType = Mila::Dnn::CausalConv1d<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static constexpr dim_t kChannels = 3;
        static constexpr dim_t kKernelWidth = 4;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        CausalConv1dConfig config( bool has_bias = false ) const
        {
            return CausalConv1dConfig( kChannels, kKernelWidth ).withBias( has_bias );
        }

        /// A built convolution with deterministic, channel-distinct filters loaded.
        std::unique_ptr<ConvType> builtConv( dim_t batch, dim_t seq_len, bool has_bias = false )
        {
            auto conv = std::make_unique<ConvType>( "conv", config( has_bias ), Device::Cuda( 0 ) );
            conv->build( BuildContext( shape_t{ batch, seq_len, kChannels }, RuntimeMode::Inference, false ) );

            auto parameters = conv->getParameters();

            HostFp32 host_weight( Device::Cpu(), shape_t{ kChannels, kKernelWidth } );

            for ( dim_t c = 0; c < kChannels; ++c )
            {
                for ( dim_t k = 0; k < kKernelWidth; ++k )
                {
                    // Distinct per (channel, tap) so a transposed or channel-collapsed
                    // index cannot pass by symmetry.
                    host_weight.data()[ c * kKernelWidth + k ] =
                        0.25f * static_cast<float>( k + 1 ) - 0.1f * static_cast<float>( c );
                }
            }

            copy( host_weight, *static_cast<DeviceTensor*>( parameters[ 0 ] ), cuda_context_.get() );

            if ( has_bias )
            {
                HostFp32 host_bias( Device::Cpu(), shape_t{ kChannels } );

                for ( dim_t c = 0; c < kChannels; ++c )
                {
                    host_bias.data()[ c ] = 0.5f + static_cast<float>( c );
                }

                copy( host_bias, *static_cast<DeviceTensor*>( parameters[ 1 ] ), cuda_context_.get() );
            }

            cuda_context_->synchronize();

            return conv;
        }

        HostFp32 weightHost() const
        {
            HostFp32 host( Device::Cpu(), shape_t{ kChannels, kKernelWidth } );

            for ( dim_t c = 0; c < kChannels; ++c )
            {
                for ( dim_t k = 0; k < kKernelWidth; ++k )
                {
                    host.data()[ c * kKernelWidth + k ] =
                        0.25f * static_cast<float>( k + 1 ) - 0.1f * static_cast<float>( c );
                }
            }

            return host;
        }

        HostFp32 rampHost( const shape_t& shape, float start, float step )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = start + step * static_cast<float>( i );
            }

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        /// Whole-sequence causal convolution with zero left padding, computed on the host.
        std::vector<float> referenceConv(
            const HostFp32& x, dim_t batch, dim_t seq_len, bool has_bias ) const
        {
            auto weight = weightHost();
            std::vector<float> out( static_cast<size_t>( batch * seq_len * kChannels ), 0.0f );

            for ( dim_t b = 0; b < batch; ++b )
            {
                for ( dim_t t = 0; t < seq_len; ++t )
                {
                    for ( dim_t c = 0; c < kChannels; ++c )
                    {
                        float accumulator = has_bias ? (0.5f + static_cast<float>( c )) : 0.0f;

                        for ( dim_t k = 0; k < kKernelWidth; ++k )
                        {
                            const dim_t source_t = t - (kKernelWidth - 1) + k;

                            if ( source_t < 0 )
                                continue;

                            accumulator += weight.data()[ c * kKernelWidth + k ]
                                * x.data()[ (b * seq_len + source_t) * kChannels + c ];
                        }

                        out[ static_cast<size_t>( (b * seq_len + t) * kChannels + c ) ] = accumulator;
                    }
                }
            }

            return out;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( CausalConv1dCudaTests, CausalConv1dPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction and configuration
    // ====================================================================

    TYPED_TEST( CausalConv1dCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::ConvType conv( "conv", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( conv.getDeviceId().type, DeviceType::Cuda );
        EXPECT_EQ( conv.getType(), ComponentType::CausalConv1d );
    }

    TYPED_TEST( CausalConv1dCudaTests, ParameterShapeIsChannelsByKernelWidth )
    {
        auto conv = this->builtConv( 1, 6 );

        EXPECT_EQ( conv->getParameterNames().size(), 1u );
        EXPECT_EQ( conv->parameterCount(),
            TestFixture::kChannels * TestFixture::kKernelWidth );
    }

    TYPED_TEST( CausalConv1dCudaTests, BiasAddsOneParameterPerChannel )
    {
        auto conv = this->builtConv( 1, 6, /*has_bias*/ true );

        EXPECT_EQ( conv->getParameterNames().size(), 2u );
        EXPECT_EQ( conv->parameterCount(),
            TestFixture::kChannels * TestFixture::kKernelWidth + TestFixture::kChannels );
    }

    // ====================================================================
    // B. Forward against a host reference
    // ====================================================================

    TYPED_TEST( CausalConv1dCudaTests, Prefill_MatchesHostReference )
    {
        constexpr dim_t B = 2;
        constexpr dim_t T = 7;
        const shape_t shape{ B, T, TestFixture::kChannels };

        auto conv = this->builtConv( B, T );
        auto host_x = this->rampHost( shape, -0.8f, 0.07f );
        auto device_x = this->toDevice( host_x );

        auto& device_out = conv->prefill( device_x, 0 );
        conv->synchronize();

        auto out = this->toFloat( device_out );
        auto expected = this->referenceConv( host_x, B, T, false );

        for ( size_t i = 0; i < expected.size(); ++i )
        {
            EXPECT_NEAR( out.data()[ i ], expected[ i ], TypeParam::atol ) << "at index " << i;
        }
    }

    TYPED_TEST( CausalConv1dCudaTests, Prefill_AppliesBias )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 5;
        const shape_t shape{ B, T, TestFixture::kChannels };

        auto conv = this->builtConv( B, T, /*has_bias*/ true );
        auto host_x = this->rampHost( shape, 0.3f, -0.11f );
        auto device_x = this->toDevice( host_x );

        auto& device_out = conv->prefill( device_x, 0 );
        conv->synchronize();

        auto out = this->toFloat( device_out );
        auto expected = this->referenceConv( host_x, B, T, true );

        for ( size_t i = 0; i < expected.size(); ++i )
        {
            EXPECT_NEAR( out.data()[ i ], expected[ i ], TypeParam::atol ) << "at index " << i;
        }
    }

    TYPED_TEST( CausalConv1dCudaTests, FirstPositionsSeeZeroLeftPadding )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 1;
        const shape_t shape{ B, T, TestFixture::kChannels };

        auto conv = this->builtConv( B, T );
        auto host_x = this->rampHost( shape, 1.0f, 1.0f );
        auto device_x = this->toDevice( host_x );

        auto& device_out = conv->prefill( device_x, 0 );
        conv->synchronize();

        auto out = this->toFloat( device_out );
        auto weight = this->weightHost();

        // A single-token sequence sees only the LAST tap; the other three multiply zeros.
        for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
        {
            const float expected =
                weight.data()[ c * TestFixture::kKernelWidth + (TestFixture::kKernelWidth - 1) ]
                * host_x.data()[ c ];

            EXPECT_NEAR( out.data()[ c ], expected, TypeParam::atol ) << "channel " << c;
        }
    }

    // ====================================================================
    // C. The state equivalences -- what the component exists for
    // ====================================================================

    TYPED_TEST( CausalConv1dCudaTests, ChunkedPrefillEqualsWholeSequence )
    {
        constexpr dim_t B = 2;
        constexpr dim_t T = 8;
        constexpr dim_t kChunk = 4;
        const shape_t whole_shape{ B, T, TestFixture::kChannels };

        auto host_x = this->rampHost( whole_shape, -0.5f, 0.09f );

        // Arm 1: the whole sequence in one pass.
        auto whole_conv = this->builtConv( B, T );
        auto device_whole = this->toDevice( host_x );
        auto& whole_out_device = whole_conv->prefill( device_whole, 0 );
        whole_conv->synchronize();
        auto whole_out = this->toFloat( whole_out_device );

        // Arm 2: the same sequence in two chunks through a second component, whose only
        // link between chunks is the retained state.
        auto chunked_conv = this->builtConv( B, kChunk );
        std::vector<float> chunked_out( static_cast<size_t>( B * T * TestFixture::kChannels ), 0.0f );

        for ( dim_t chunk_start = 0; chunk_start < T; chunk_start += kChunk )
        {
            typename TestFixture::HostFp32 host_chunk(
                Device::Cpu(), shape_t{ B, kChunk, TestFixture::kChannels } );

            for ( dim_t b = 0; b < B; ++b )
            {
                for ( dim_t t = 0; t < kChunk; ++t )
                {
                    for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
                    {
                        host_chunk.data()[ (b * kChunk + t) * TestFixture::kChannels + c ] =
                            host_x.data()[ (b * T + chunk_start + t) * TestFixture::kChannels + c ];
                    }
                }
            }

            auto device_chunk = this->toDevice( host_chunk );
            auto& chunk_out_device = chunked_conv->prefill( device_chunk, chunk_start );
            chunked_conv->synchronize();
            auto chunk_out = this->toFloat( chunk_out_device );

            for ( dim_t b = 0; b < B; ++b )
            {
                for ( dim_t t = 0; t < kChunk; ++t )
                {
                    for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
                    {
                        chunked_out[ static_cast<size_t>(
                            (b * T + chunk_start + t) * TestFixture::kChannels + c ) ] =
                            chunk_out.data()[ (b * kChunk + t) * TestFixture::kChannels + c ];
                    }
                }
            }
        }

        for ( size_t i = 0; i < chunked_out.size(); ++i )
        {
            EXPECT_NEAR( chunked_out[ i ], whole_out.data()[ i ], TypeParam::atol )
                << "at index " << i;
        }
    }

    TYPED_TEST( CausalConv1dCudaTests, TokenByTokenDecodeEqualsWholeSequence )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 6;
        const shape_t whole_shape{ B, T, TestFixture::kChannels };

        auto host_x = this->rampHost( whole_shape, 0.4f, -0.13f );

        auto whole_conv = this->builtConv( B, T );
        auto device_whole = this->toDevice( host_x );
        auto& whole_out_device = whole_conv->prefill( device_whole, 0 );
        whole_conv->synchronize();
        auto whole_out = this->toFloat( whole_out_device );

        // One token at a time. The first goes through prefill at position 0 -- that is what
        // establishes the sequence start -- and the rest through decode.
        auto decode_conv = this->builtConv( B, 1 );

        for ( dim_t t = 0; t < T; ++t )
        {
            typename TestFixture::HostFp32 host_step(
                Device::Cpu(), shape_t{ B, 1, TestFixture::kChannels } );

            for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
            {
                host_step.data()[ c ] = host_x.data()[ t * TestFixture::kChannels + c ];
            }

            auto device_step = this->toDevice( host_step );

            auto& step_out_device = (t == 0)
                ? decode_conv->prefill( device_step, 0 )
                : decode_conv->decode( device_step, t );

            decode_conv->synchronize();
            auto step_out = this->toFloat( step_out_device );

            for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
            {
                EXPECT_NEAR( step_out.data()[ c ],
                    whole_out.data()[ t * TestFixture::kChannels + c ], TypeParam::atol )
                    << "position " << t << " channel " << c;
            }
        }
    }

    /**
     * @brief Positive control for the two equivalence tests above.
     *
     * Those tests pass when the state works -- and would also pass if the state were
     * irrelevant, e.g. if the retained rows were never read. This one pins the other
     * direction: convolving the second chunk as a FRESH sequence must NOT reproduce the
     * whole-sequence answer, because the first K-1 of its positions genuinely depend on
     * the tail of chunk one. If this ever starts passing as an equality, the equivalences
     * above have stopped proving anything.
     */
    TYPED_TEST( CausalConv1dCudaTests, DiscardingTheStateChangesTheChunkBoundary )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 8;
        constexpr dim_t kChunk = 4;
        const shape_t whole_shape{ B, T, TestFixture::kChannels };
        const shape_t chunk_shape{ B, kChunk, TestFixture::kChannels };

        // A steeper ramp than the equivalence tests use, deliberately. The separation this
        // control measures is the sum of the three DROPPED taps, so it scales with the
        // input; at a shallow ramp it lands near BF16's noise floor and the control cannot
        // do its job in both precisions.
        auto host_x = this->rampHost( whole_shape, 1.0f, 0.5f );

        auto whole_conv = this->builtConv( B, T );
        auto device_whole = this->toDevice( host_x );
        auto& whole_out_device = whole_conv->prefill( device_whole, 0 );
        whole_conv->synchronize();
        auto whole_out = this->toFloat( whole_out_device );

        typename TestFixture::HostFp32 host_second( Device::Cpu(), chunk_shape );

        for ( dim_t t = 0; t < kChunk; ++t )
        {
            for ( dim_t c = 0; c < TestFixture::kChannels; ++c )
            {
                host_second.data()[ t * TestFixture::kChannels + c ] =
                    host_x.data()[ (kChunk + t) * TestFixture::kChannels + c ];
            }
        }

        auto fresh_conv = this->builtConv( B, kChunk );
        auto device_second = this->toDevice( host_second );
        auto& fresh_device = fresh_conv->prefill( device_second, 0 );
        fresh_conv->synchronize();
        auto fresh = this->toFloat( fresh_device );

        // Position 0 of the chunk loses all K-1 left taps, so it moves the most. At this
        // ramp the dropped taps are worth about 6.75, which is three orders above BF16's
        // noise at these magnitudes -- so the threshold below separates "state ignored"
        // from "state working" rather than separating precisions.
        const float whole_first =
            whole_out.data()[ kChunk * TestFixture::kChannels + 0 ];
        const float fresh_first = fresh.data()[ 0 ];

        EXPECT_GT( std::fabs( whole_first - fresh_first ), 1.0f )
            << "discarding the retained rows changed nothing -- the equivalence tests "
               "above cannot be distinguishing a working state from an ignored one";
    }

    TYPED_TEST( CausalConv1dCudaTests, ResetStateStartsAFreshSequence )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 4;
        const shape_t shape{ B, T, TestFixture::kChannels };

        auto conv = this->builtConv( B, T );
        auto host_x = this->rampHost( shape, 0.2f, 0.17f );
        auto device_x = this->toDevice( host_x );

        auto& first_device = conv->prefill( device_x, 0 );
        conv->synchronize();
        auto first = this->toFloat( first_device );

        conv->resetState();

        auto& second_device = conv->prefill( device_x, 0 );
        conv->synchronize();
        auto second = this->toFloat( second_device );

        for ( dim_t i = 0; i < first.size(); ++i )
        {
            EXPECT_NEAR( second.data()[ i ], first.data()[ i ], TypeParam::atol ) << "at index " << i;
        }
    }

    // ====================================================================
    // D. Guard clauses
    // ====================================================================

    TYPED_TEST( CausalConv1dCudaTests, DecodeBeforeAnyPrefillIsRefused )
    {
        constexpr dim_t B = 1;
        auto conv = this->builtConv( B, 1 );

        auto host_x = this->rampHost( shape_t{ B, 1, TestFixture::kChannels }, 1.0f, 1.0f );
        auto device_x = this->toDevice( host_x );

        // Zeros would be a plausible-looking answer to an unanswerable question: there is
        // no left context because no chunk has been seen.
        EXPECT_THROW( (void)conv->decode( device_x, 0 ), std::logic_error );
    }

    TYPED_TEST( CausalConv1dCudaTests, ContinuationPrefillBeforeAnyChunkIsRefused )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 3;
        auto conv = this->builtConv( B, T );

        auto host_x = this->rampHost( shape_t{ B, T, TestFixture::kChannels }, 1.0f, 1.0f );
        auto device_x = this->toDevice( host_x );

        EXPECT_THROW( (void)conv->prefill( device_x, T ), std::logic_error );
    }

    TYPED_TEST( CausalConv1dCudaTests, ChannelMismatchIsRefused )
    {
        constexpr dim_t B = 1;
        constexpr dim_t T = 3;
        auto conv = this->builtConv( B, T );

        auto host_x = this->rampHost( shape_t{ B, T, TestFixture::kChannels + 1 }, 1.0f, 1.0f );
        auto device_x = this->toDevice( host_x );

        EXPECT_THROW( (void)conv->prefill( device_x, 0 ), std::invalid_argument );
    }

    TYPED_TEST( CausalConv1dCudaTests, ForwardBeforeBuildIsRefused )
    {
        typename TestFixture::ConvType conv( "conv", this->config(), Device::Cuda( 0 ) );

        auto host_x = this->rampHost( shape_t{ 1, 2, TestFixture::kChannels }, 1.0f, 1.0f );
        auto device_x = this->toDevice( host_x );

        EXPECT_THROW( (void)conv.prefill( device_x, 0 ), std::runtime_error );
    }
}
