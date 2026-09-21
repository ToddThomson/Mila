/**
 * @file GatedMLP.Cuda.cpp
 * @brief Concrete-component tests for GatedMLP<DeviceType::Cuda, {FP32, BF16}>.
 *
 * GatedMLP's gate is the CUDA-only Swiglu op, so there is no CPU companion -- this
 * is the whole functional surface (the config contract lives in GatedMLPConfig.cpp).
 * A precision sweep (FP32, BF16) covers the composite wiring at the value-agnostic
 * level (build shape contract, aggregate parameter count, backward shape, the
 * footprint prediction). The pooling contract and the Gemma equivalence gate follow.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device is present.
 */

#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::FFN::GatedMLP
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // hidden_size is the Swiglu half_width; must be a multiple of the BF16 kernel
        // vector width (8). in_features is unconstrained but 8 keeps the GEMMs small.
        constexpr int64_t kInFeatures = 8;
        constexpr int64_t kHiddenSize = 8;

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr const char* name = "Bf16";
        };

        using GatedMLPPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int ) { return TPrecisionTag::name; }
        };

        // Aggregate trainable count: fc_gate_up (2H x in) + fc_down (in x H), bias-free.
        constexpr size_t kExpectedParameters =
            static_cast<size_t>( 2 * kHiddenSize * kInFeatures + kInFeatures * kHiddenSize );

        // bfloat16 is exactly the top 16 bits of the IEEE float32 pattern: sign, the full
        // 8-bit exponent, and the leading 7 mantissa bits.
        std::vector<std::uint16_t> randomBf16Bits( dim_t count, std::uint32_t seed )
        {
            std::mt19937 generator( seed );
            std::normal_distribution<float> distribution( 0.0f, 0.05f );

            std::vector<std::uint16_t> bits( static_cast<std::size_t>( count ) );

            for ( auto& value : bits )
            {
                value = static_cast<std::uint16_t>( std::bit_cast<std::uint32_t>( distribution( generator ) ) >> 16 );
            }

            return bits;
        }

        // The quantize-on-load upload is async and the blob is stack-owned, so the load
        // synchronizes before the vector can go out of scope.
        template<typename TComponent, typename TValue>
        void loadWeight( TComponent& target, TensorDataType dtype, const std::vector<TValue>& values, dim_t rows, dim_t columns )
        {
            const std::size_t bytes = values.size() * sizeof( TValue );

            Serialization::TensorMetadata meta{ dtype, shape_t{ rows, columns }, bytes };
            Serialization::TensorBlobView blob( meta, values.data(), bytes );

            target.loadParameter( "weight", blob );
            target.synchronize();
        }
    }

    template<typename TPrecisionTag>
    class GatedMLPCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using GatedMLPType = Mila::Dnn::GatedMLP<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( GatedMLPType::getDeviceType() == DeviceType::Cuda );
        static_assert( GatedMLPType::getPrecision() == P );

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

        // Training build so the child Linear weights are initialized (finite).
        std::unique_ptr<GatedMLPType> built( const shape_t& shape )
        {
            auto gated = std::make_unique<GatedMLPType>(
                "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
            gated->build( BuildContext( shape, RuntimeMode::Training ) );

            return gated;
        }

        DeviceTensor zeros( const shape_t& shape )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = 0.0f;
            }

            DeviceTensor device( Device::Cuda( 0 ), shape );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( GatedMLPCudaTests, GatedMLPPrecisions, PrecisionNames );

    TYPED_TEST( GatedMLPCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::GatedMLPType gated(
            "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );

        EXPECT_EQ( gated.getDeviceId().type, DeviceType::Cuda );
        EXPECT_EQ( gated.getType(), ComponentType::GatedMlp );
    }

    TYPED_TEST( GatedMLPCudaTests, Forward_PreservesInputShape )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        auto gated = this->built( shape );
        auto input = this->zeros( shape );

        auto& output = gated->forward( input );
        gated->synchronize();

        ASSERT_EQ( output.shape().size(), 3u );
        EXPECT_EQ( output.shape()[ 0 ], 2 );
        EXPECT_EQ( output.shape()[ 1 ], 3 );
        EXPECT_EQ( output.shape()[ 2 ], kInFeatures );
    }

    TYPED_TEST( GatedMLPCudaTests, ParameterCount_IsSumOfProjections )
    {
        auto gated = this->built( shape_t{ 2, 3, kInFeatures } );

        EXPECT_EQ( gated->parameterCount(), kExpectedParameters );
    }

    TYPED_TEST( GatedMLPCudaTests, Backward_ProducesInputShapeGradient )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        auto gated = this->built( shape );
        auto input = this->zeros( shape );
        auto output_grad = this->zeros( shape );

        gated->forward( input );
        auto& input_grad = gated->backward( input, output_grad );
        gated->synchronize();

        ASSERT_EQ( input_grad.shape().size(), 3u );
        EXPECT_EQ( input_grad.shape()[ 2 ], kInFeatures );
    }

    // The drift gate: the prediction must equal what a real build reports. Each child is
    // sized at its own width, which the generic single-shape recursion would get wrong.
    TYPED_TEST( GatedMLPCudaTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        typename TestFixture::GatedMLPType predictor(
            "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( BuildContext( shape, RuntimeMode::Training ) );

        auto built = this->built( shape );
        const MemoryStats actual = built->getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }

    // ====================================================================
    // FP32: numeric identity and the pooling contract.
    // ====================================================================

    class GatedMLPCudaFp32Tests : public ::testing::Test
    {
    protected:
        using GatedMLPType = Mila::Dnn::GatedMLP<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceFp32 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

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

        DeviceFp32 filled( const shape_t& shape, float value )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = value;
            }

            DeviceFp32 device( Device::Cuda( 0 ), shape );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    // Bias-free, zero input -> zero output: 0 -> fc_gate_up=0 -> SiLU(0)*0 = 0 -> fc_down(0) = 0.
    // Training build so weights are finite; 0 * finite = 0 exactly.
    TEST_F( GatedMLPCudaFp32Tests, Forward_ZeroInputYieldsZero )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        GatedMLPType gated( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        gated.build( BuildContext( shape, RuntimeMode::Training ) );

        auto input = filled( shape, 0.0f );

        auto& output = gated.forward( input );
        gated.synchronize();

        auto host_out = toHost<TensorDataType::FP32>( output, cuda_context_.get() );

        for ( dim_t i = 0; i < host_out.size(); ++i )
        {
            EXPECT_NEAR( host_out.data()[ i ], 0.0f, 1e-6f )
                << "non-zero output at index " << i;
        }
    }

    // Installed slots are what the children write, and neither the prediction nor the
    // built report counts them -- the installer owns them. Without this a pooling block
    // would lose its pooling on delegation and nothing would say so.
    TEST_F( GatedMLPCudaFp32Tests, InstalledOutputs_AreWrittenAndNotCounted )
    {
        constexpr int64_t batch = 2;
        constexpr int64_t tokens = 3;
        const shape_t shape{ batch, tokens, kInFeatures };
        const BuildContext context = BuildContext( shape, RuntimeMode::Inference ).withPrefillSize( tokens );

        GatedMLPType unpooled_predictor( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        const MemoryStats unpooled = unpooled_predictor.getRequiredMemory( context );

        GatedMLPType pooled_predictor( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        const MemoryStats predicted = pooled_predictor.getRequiredMemory( context.withInstalledOutput( true ) );

        auto gate_up_slot = std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape_t{ batch, tokens, 2 * kHiddenSize } );
        auto gate_slot = std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape_t{ batch, tokens, kHiddenSize } );
        auto down_slot = std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape );

        GatedMLPType gated( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        gated.installSharedOutputs( gate_up_slot, gate_slot, down_slot );
        gated.build( context );

        EXPECT_EQ( predicted.device_state_bytes, gated.getMemoryStats().device_state_bytes );
        EXPECT_LT( predicted.device_state_bytes, unpooled.device_state_bytes );

        const std::vector<float> gate_up_weight( static_cast<std::size_t>( 2 * kHiddenSize * kInFeatures ), 0.25f );
        const std::vector<float> down_weight( static_cast<std::size_t>( kInFeatures * kHiddenSize ), 0.5f );
        loadWeight( *gated.findComponent( "gmlp.fc_gate_up" ), TensorDataType::FP32, gate_up_weight, 2 * kHiddenSize, kInFeatures );
        loadWeight( *gated.findComponent( "gmlp.fc_down" ), TensorDataType::FP32, down_weight, kInFeatures, kHiddenSize );

        auto input = filled( shape, 1.0f );
        auto& output = gated.forward( input );
        gated.synchronize();

        auto host_output = toHost<TensorDataType::FP32>( output, cuda_context_.get() );
        auto host_slot = toHost<TensorDataType::FP32>( *down_slot, cuda_context_.get() );

        ASSERT_EQ( host_output.size(), host_slot.size() );
        ASSERT_NE( host_output.data()[ 0 ], 0.0f ) << "the forward produced nothing to compare";

        for ( dim_t i = 0; i < host_output.size(); ++i )
        {
            EXPECT_EQ( host_output.data()[ i ], host_slot.data()[ i ] ) << "index " << i;
        }
    }

    TEST_F( GatedMLPCudaFp32Tests, InstallSharedOutputs_AfterBuildThrows )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        GatedMLPType gated( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        gated.build( BuildContext( shape, RuntimeMode::Inference ).withPrefillSize( 3 ) );

        EXPECT_THROW(
            gated.installSharedOutputs(
                std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape_t{ 2, 3, 2 * kHiddenSize } ),
                std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape_t{ 2, 3, kHiddenSize } ),
                std::make_shared<DeviceFp32>( Device::Cuda( 0 ), shape ) ),
            std::logic_error );
    }

    // ====================================================================
    // Gemma equivalence: the gate on delegating GemmaBlock's FFN.
    // ====================================================================
    //
    // GemmaBlock wires its FFN inline as Linear -> Swiglu<Gelu> -> Linear at the block's
    // weight quantization. GatedMLP<Gelu, TWeightQuantization> must reproduce that chain
    // bit for bit on the same weights, so that moving the block onto it changes tensor
    // names and nothing else. Width 128 because PerGroupFp4<128> groups both projection
    // inputs by 128.

    class GatedMLPGemmaEquivalenceCudaTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kWidth = 128;
        static constexpr int64_t kBatch = 1;
        static constexpr int64_t kTokens = 4;

        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "CUDA device not available";
            }

            cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        template<typename TWeightQuantization>
        void expectBitIdenticalToInlineChain()
        {
            using GatedType = Mila::Dnn::GatedMLP<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu, TWeightQuantization>;
            using LinearType = Linear<DeviceType::Cuda, TensorDataType::BF16, TWeightQuantization>;
            using GateType = Swiglu<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu>;
            using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
            using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            const shape_t shape{ kBatch, kTokens, kWidth };
            const BuildContext context = BuildContext( shape, RuntimeMode::Inference ).withPrefillSize( kTokens );

            const auto gate_up_bits = randomBf16Bits( 2 * kWidth * kWidth, 11 );
            const auto down_bits = randomBf16Bits( kWidth * kWidth, 23 );

            GatedType gated( "gmlp",
                GatedMLPConfig( kWidth, kWidth ).withGateActivation( ActivationType::Gelu ), Device::Cuda( 0 ) );
            gated.build( context );
            loadWeight( *gated.findComponent( "gmlp.fc_gate_up" ), TensorDataType::BF16, gate_up_bits, 2 * kWidth, kWidth );
            loadWeight( *gated.findComponent( "gmlp.fc_down" ), TensorDataType::BF16, down_bits, kWidth, kWidth );

            LinearType fc_gate_up( "fc_gate_up", LinearConfig( kWidth, 2 * kWidth ).withBias( false ), Device::Cuda( 0 ) );
            GateType geglu( "geglu", SwigluConfig(), Device::Cuda( 0 ) );
            LinearType fc_down( "fc_down", LinearConfig( kWidth, kWidth ).withBias( false ), Device::Cuda( 0 ) );

            fc_gate_up.build( context );
            geglu.build( context.withShape( shape_t{ kBatch, kTokens, 2 * kWidth } ) );
            fc_down.build( context.withShape( shape_t{ kBatch, kTokens, kWidth } ) );
            loadWeight( fc_gate_up, TensorDataType::BF16, gate_up_bits, 2 * kWidth, kWidth );
            loadWeight( fc_down, TensorDataType::BF16, down_bits, kWidth, kWidth );

            std::mt19937 generator( 7 );
            std::normal_distribution<float> distribution( 0.0f, 1.0f );

            HostFp32 host_input( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host_input.size(); ++i )
            {
                host_input.data()[ i ] = distribution( generator );
            }

            DeviceBf16 input( Device::Cuda( 0 ), shape );
            copy( host_input, input, cuda_context_.get() );
            cuda_context_->synchronize();

            auto& gated_output = gated.forward( input );
            gated.synchronize();

            auto& gate_up = fc_gate_up.forward( input );
            fc_gate_up.synchronize();
            auto& activation = geglu.forward( gate_up );
            geglu.synchronize();
            auto& reference_output = fc_down.forward( activation );
            fc_down.synchronize();

            // BF16 -> FP32 is exact, so equal FP32 values are equal BF16 bit patterns.
            auto gated_host = toHost<TensorDataType::FP32>( gated_output, cuda_context_.get() );
            auto reference_host = toHost<TensorDataType::FP32>( reference_output, cuda_context_.get() );

            ASSERT_EQ( gated_host.size(), reference_host.size() );

            dim_t mismatches = 0;
            dim_t nonzero = 0;

            for ( dim_t i = 0; i < gated_host.size(); ++i )
            {
                if ( gated_host.data()[ i ] != reference_host.data()[ i ] )
                {
                    ++mismatches;
                }

                if ( reference_host.data()[ i ] != 0.0f )
                {
                    ++nonzero;
                }
            }

            EXPECT_GT( nonzero, 0 ) << "the reference produced all zeros; the comparison proves nothing";
            EXPECT_EQ( mismatches, 0 ) << "of " << gated_host.size() << " elements";
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( GatedMLPGemmaEquivalenceCudaTests, Bf16_BitIdenticalToGemmaInlineFfn )
    {
        expectBitIdenticalToInlineChain<Mila::Dnn::Quant::Weight::NoWeightQuant>();
    }

    TEST_F( GatedMLPGemmaEquivalenceCudaTests, PerGroupFp4_BitIdenticalToGemmaInlineFfn )
    {
        expectBitIdenticalToInlineChain<Mila::Dnn::Quant::Weight::PerGroupFp4<128>>();
    }
}
