/**
 * @file GatedResidual.Cuda.cpp
 * @brief GatedResidual<DeviceType::Cuda, {FP32, BF16}> and its Qwen 4 reference gate.
 *
 * The gate (Specifications/Qwen4.md section 9, Phase 2) reads the tiny reference produced by
 * Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py --variant moe: every gated residual of every layer, at every
 * prefill chunk and decode step, and the network's final read-only mixer. It skips when the capture is absent.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Connections
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // The tiny reference's geometry: hidden 256, 4 streams, rank 32.
        constexpr dim_t kModelDim = 256;
        constexpr dim_t kStreams = 4;
        constexpr dim_t kRank = 32;
        constexpr dim_t kLayers = 4;
        constexpr dim_t kWidestStep = 13;

        const std::vector<std::string> kSteps{
            "prefill0", "prefill1", "decode0", "decode1", "decode2", "decode3", "decode4", "decode5", "decode6", "decode7" };

        // FP32 against FP32 keeps the CPU gate's tolerance: only summation order and exp() differ. BF16 rounds the
        // stream, the weights and every intermediate to 8 bits of mantissa.
        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float absolute = 1e-5f;
            static constexpr float relative = 1e-5f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float absolute = 3e-2f;
            static constexpr float relative = 3e-2f;
            static constexpr const char* name = "Bf16";
        };

        using GatedResidualPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };

        fs::path captureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "Models" / "Qwen4" / "qwen4_tiny_moe";
        }

        GatedResidualConfig config( bool has_injection = true )
        {
            return GatedResidualConfig( kModelDim, kStreams, kRank ).withEpsilon( 1e-6f ).withInjection( has_injection );
        }

        HostFp32 sineHost( const shape_t& shape, double phase )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( std::sin( 0.37 * static_cast<double>( i ) + phase ) );
            }

            return host;
        }
    }

    template<typename TPrecisionTag>
    class GatedResidualCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using ResidualType = Mila::Dnn::GatedResidual<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;

        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        // Initialized for the structural tests, whose claims must not depend on what the allocation held; left alone
        // for the gate, which loads every weight.
        std::unique_ptr<ResidualType> builtResidual( const GatedResidualConfig& residual_config, dim_t rows, bool initialize = true )
        {
            auto residual = std::make_unique<ResidualType>( "residual", residual_config, Device::Cuda( 0 ) );
            residual->build( BuildContext( shape_t{ rows, kStreams * kModelDim }, RuntimeMode::Inference, initialize ) );

            return residual;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, context_.get() );
            context_->synchronize();

            return device;
        }

        // The component computes on its own stream, so the device is drained before the copy reads its output.
        HostFp32 toFloat( const DeviceTensor& device )
        {
            cudaDeviceSynchronize();

            auto host = toHost<TensorDataType::FP32>( device, context_.get() );
            context_->synchronize();

            return host;
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    TYPED_TEST_SUITE( GatedResidualCudaTests, GatedResidualPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction and validation
    // ====================================================================

    TYPED_TEST( GatedResidualCudaTests, Construct_HasTheReferenceChildren )
    {
        typename TestFixture::ResidualType residual( "residual", config(), Device::Cuda( 0 ) );

        EXPECT_EQ( residual.getType(), ComponentType::GatedResidual );
        EXPECT_NE( residual.findComponent( "residual.norm" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_down" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_up" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_inject" ), nullptr );
    }

    TYPED_TEST( GatedResidualCudaTests, ReadOnly_HasNoInjectProjection )
    {
        typename TestFixture::ResidualType residual( "residual", config( false ), Device::Cuda( 0 ) );

        EXPECT_THROW( (void)residual.findComponent( "residual.fc_inject" ), std::exception );
    }

    TYPED_TEST( GatedResidualCudaTests, Build_RefusesAStreamOfTheWrongWidth )
    {
        typename TestFixture::ResidualType residual( "residual", config(), Device::Cuda( 0 ) );

        EXPECT_THROW( residual.build( BuildContext( shape_t{ 2, kModelDim }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TYPED_TEST( GatedResidualCudaTests, InjectBeforeReadIsRefused )
    {
        auto residual = this->builtResidual( config(), 2 );
        typename TestFixture::DeviceTensor stream( Device::Cuda( 0 ), shape_t{ 2, kStreams * kModelDim } );
        typename TestFixture::DeviceTensor output( Device::Cuda( 0 ), shape_t{ 2, kModelDim } );

        EXPECT_THROW( (void)residual->inject( stream, output ), std::invalid_argument );
    }

    TYPED_TEST( GatedResidualCudaTests, InjectOnAReadOnlyResidualIsRefused )
    {
        auto residual = this->builtResidual( config( false ), 2 );
        auto stream = this->toDevice( sineHost( shape_t{ 2, kStreams * kModelDim }, 0.1 ) );
        typename TestFixture::DeviceTensor output( Device::Cuda( 0 ), shape_t{ 2, kModelDim } );

        (void)residual->read( stream );

        EXPECT_THROW( (void)residual->inject( stream, output ), std::logic_error );
    }

    // ====================================================================
    // B. Structure of the arithmetic, independent of the reference
    // ====================================================================

    // A zero sublayer output leaves the stream unchanged: the write adds to S, never replaces it with N.
    TYPED_TEST( GatedResidualCudaTests, Inject_ZeroSublayerOutputLeavesTheStreamUnchanged )
    {
        auto residual = this->builtResidual( config(), 3 );
        auto stream = this->toDevice( sineHost( shape_t{ 3, kStreams * kModelDim }, 0.4 ) );

        HostFp32 zero( Device::Cpu(), shape_t{ 3, kModelDim } );
        std::fill( zero.data(), zero.data() + zero.size(), 0.0f );
        auto zero_output = this->toDevice( zero );

        (void)residual->read( stream );
        const auto updated = this->toFloat( residual->inject( stream, zero_output ) );
        const auto original = this->toFloat( stream );

        for ( dim_t i = 0; i < original.size(); ++i )
        {
            EXPECT_EQ( updated.data()[ i ], original.data()[ i ] ) << "at index " << i;
        }
    }

    // Write gates are 2 * sigmoid, so they lie in (0, 2) whatever the weights.
    TYPED_TEST( GatedResidualCudaTests, InjectionWeightsLieBetweenZeroAndTwo )
    {
        auto residual = this->builtResidual( config(), 3 );
        auto stream = this->toDevice( sineHost( shape_t{ 3, kStreams * kModelDim }, 0.2 ) );

        (void)residual->read( stream );
        const auto& device_weights = residual->injectionWeights();

        ASSERT_EQ( device_weights.shape(), ( shape_t{ 3, kStreams } ) );

        const auto weights = this->toFloat( device_weights );

        for ( dim_t i = 0; i < weights.size(); ++i )
        {
            EXPECT_GT( weights.data()[ i ], 0.0f );
            EXPECT_LT( weights.data()[ i ], 2.0f );
        }
    }

    // ====================================================================
    // C. The Qwen 4 reference gate (Qwen4.md Phase 2)
    // ====================================================================

    template<typename TPrecisionTag>
    class GatedResidualCudaReferenceTests : public GatedResidualCudaTests<TPrecisionTag>
    {
    protected:
        using Base = GatedResidualCudaTests<TPrecisionTag>;

        void SetUp() override
        {
            Base::SetUp();

            if ( this->IsSkipped() )
            {
                return;
            }

            if ( !fs::exists( capturePath() ) || !fs::exists( captureDirectory() / "weights_fp32.safetensors" ) )
            {
                GTEST_SKIP() << "Qwen 4 tiny reference not present at: " << capturePath().string();
            }

            capture_ = std::make_unique<Serialization::WeightsReader>( capturePath() );
            weights_ = std::make_unique<Serialization::WeightsReader>( captureDirectory() / "weights_fp32.safetensors" );
        }

        static fs::path capturePath()
        {
            return captureDirectory() / "qwen4_tiny_moe_reference.safetensors";
        }

        HostFp32 captured( const std::string& name )
        {
            const auto blob = capture_->readTensorBlob<CpuMemoryResource>( name );
            HostFp32 tensor( Device::Cpu(), blob.getMetadata().shape );
            std::memcpy( tensor.data(), blob.data(), blob.sizeBytes() );

            return tensor;
        }

        using Fp32Residual = Mila::Dnn::GatedResidual<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceFp32 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        // The capture's weights are FP32, which a BF16 parameter refuses to load; a BF16 residual receives them
        // through an FP32 twin, converted on the device.
        std::unique_ptr<typename Base::ResidualType> loadedResidual( const std::string& source, bool has_injection )
        {
            auto fp32 = std::make_unique<Fp32Residual>( "residual", config( has_injection ), Device::Cuda( 0 ) );
            fp32->build( BuildContext( shape_t{ kWidestStep, kStreams * kModelDim }, RuntimeMode::Inference, false ) );

            loadChild( *fp32, "residual.norm", source + ".hc_norm.weight" );
            loadChild( *fp32, "residual.fc_down", source + ".input_mix_weight_down.weight" );
            loadChild( *fp32, "residual.fc_up", source + ".input_mix_weight_up.weight" );

            if ( has_injection )
            {
                loadChild( *fp32, "residual.fc_inject", source + ".block_inject_weight.weight" );
            }

            auto residual = this->builtResidual( config( has_injection ), kWidestStep, false );
            const auto sources = fp32->getParameters();
            const auto targets = residual->getParameters();

            for ( size_t i = 0; i < sources.size(); ++i )
            {
                copy( *static_cast<const DeviceFp32*>( sources[ i ] ), *static_cast<typename Base::DeviceTensor*>( targets[ i ] ), this->context_.get() );
            }

            cudaDeviceSynchronize();

            return residual;
        }

        void loadChild( Fp32Residual& residual, const std::string& child, const std::string& source )
        {
            residual.findComponent( child )->loadParameter( "weight", weights_->readTensorBlob<CpuMemoryResource>( source ) );
        }

        /// Largest error beyond tolerance, as a multiple of the tolerance; at most 1 passes.
        static double worstToleranceMultiple( const HostFp32& actual, const HostFp32& expected )
        {
            double worst = 0.0;

            for ( dim_t i = 0; i < expected.size(); ++i )
            {
                const double allowed = TPrecisionTag::absolute + TPrecisionTag::relative * std::fabs( expected.data()[ i ] );
                worst = std::max( worst, std::fabs( actual.data()[ i ] - expected.data()[ i ] ) / allowed );
            }

            return worst;
        }

        std::unique_ptr<Serialization::WeightsReader> capture_;
        std::unique_ptr<Serialization::WeightsReader> weights_;
    };

    TYPED_TEST_SUITE( GatedResidualCudaReferenceTests, GatedResidualPrecisions, PrecisionNames );

    TYPED_TEST( GatedResidualCudaReferenceTests, EveryResidualOfEveryStepMatchesTheCapture )
    {
        double worst = 0.0;
        std::string worst_at;

        for ( dim_t layer = 0; layer < kLayers; ++layer )
        {
            for ( const char* residual_name : { "attn_residual", "mlp_residual" } )
            {
                const std::string kind( residual_name );
                const std::string source = std::format( "model.layers.{}.{}", layer,
                    kind == "attn_residual" ? "attn_hyper_connection" : "mlp_hyper_connection" );

                auto residual = this->loadedResidual( source, true );

                for ( const auto& step : kSteps )
                {
                    const std::string prefix = std::format( "{}.layer{}.", step, layer );
                    const auto stream = this->toDevice( this->captured( prefix + kind + ".stream" ) );

                    const auto mixed = this->toFloat( residual->read( stream ) );
                    const auto weights = this->toFloat( residual->injectionWeights() );

                    // The sublayer's output, and the stream the next residual (or the next layer) reads.
                    const auto sublayer = this->toDevice( this->captured( prefix + ( kind == "attn_residual" ? "mixer" : "mlp" ) ) );
                    const auto next = this->captured( prefix + ( kind == "attn_residual" ? "mlp_residual.stream" : "output" ) );

                    const auto updated = this->toFloat( residual->inject( stream, sublayer ) );

                    const double errors[] = {
                        this->worstToleranceMultiple( mixed, this->captured( prefix + kind + ".mixed" ) ),
                        this->worstToleranceMultiple( weights, this->captured( prefix + kind + ".injection" ) ),
                        this->worstToleranceMultiple( updated, next ) };

                    const char* what[] = { "mixed", "injection", "updated stream" };

                    for ( int i = 0; i < 3; ++i )
                    {
                        EXPECT_LE( errors[ i ], 1.0 ) << prefix << kind << " " << what[ i ];

                        if ( errors[ i ] > worst )
                        {
                            worst = errors[ i ];
                            worst_at = prefix + kind + " " + what[ i ];
                        }
                    }
                }
            }
        }

        std::cout << "[ GatedResidual CUDA " << TypeParam::name << " gate ] worst error " << worst << " x tolerance, at " << worst_at << std::endl;
    }

    TYPED_TEST( GatedResidualCudaReferenceTests, TheFinalMixerMatchesTheCaptureAtEveryStep )
    {
        auto mixer = this->loadedResidual( "model.hyper_connection_mixer", false );

        for ( const auto& step : kSteps )
        {
            const auto stream = this->toDevice( this->captured( step + ".final_mixer.stream" ) );
            const auto mixed = this->toFloat( mixer->read( stream ) );

            EXPECT_LE( this->worstToleranceMultiple( mixed, this->captured( step + ".final_mixer.mixed" ) ), 1.0 ) << step;
        }
    }
}
