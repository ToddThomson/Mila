/**
 * @file GatedResidual.Cpu.cpp
 * @brief Concrete-component tests for GatedResidual<DeviceType::Cpu, FP32>, and its Qwen 4 reference gate.
 *
 * The gate (Specifications/Qwen4.md section 9, Phase 2) reads the tiny reference produced by
 * Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py --variant moe: every gated residual of every layer, at
 * every prefill chunk and decode step, and the network's final read-only mixer. It skips when the capture is absent.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
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

        using GatedResidualCpu = Mila::Dnn::GatedResidual<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // The tiny reference's geometry: hidden 256, 4 streams, rank 32.
        constexpr dim_t kModelDim = 256;
        constexpr dim_t kStreams = 4;
        constexpr dim_t kRank = 32;
        constexpr dim_t kLayers = 4;
        constexpr dim_t kWidestStep = 13;

        // Written before the first run: FP32 against FP32, so only summation order and exp() differ.
        constexpr float kAbsoluteTolerance = 1e-5f;
        constexpr float kRelativeTolerance = 1e-5f;

        const std::vector<std::string> kSteps{
            "prefill0", "prefill1", "decode0", "decode1", "decode2", "decode3", "decode4", "decode5", "decode6", "decode7" };

        fs::path captureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "Models" / "Qwen4" / "qwen4_tiny_moe";
        }

        GatedResidualConfig config( bool has_injection = true )
        {
            return GatedResidualConfig( kModelDim, kStreams, kRank ).withEpsilon( 1e-6f ).withInjection( has_injection );
        }

        // Initialized for the structural tests, whose claims must not depend on what the allocation held; left
        // alone for the gate, which loads every weight.
        std::unique_ptr<GatedResidualCpu> builtResidual( const GatedResidualConfig& residual_config, dim_t rows, bool initialize = true )
        {
            auto residual = std::make_unique<GatedResidualCpu>( "residual", residual_config, Device::Cpu() );
            residual->build( BuildContext( shape_t{ rows, kStreams * kModelDim }, RuntimeMode::Inference, initialize ) );

            return residual;
        }

        template<typename TBlob>
        TensorFp32 tensorFrom( const TBlob& blob )
        {
            TensorFp32 tensor( Device::Cpu(), blob.getMetadata().shape );
            std::memcpy( tensor.data(), blob.data(), blob.sizeBytes() );

            return tensor;
        }

        /// Largest error beyond tolerance, as a multiple of the tolerance; at most 1 passes.
        double worstToleranceMultiple( const TensorFp32& actual, const TensorFp32& expected )
        {
            double worst = 0.0;

            for ( dim_t i = 0; i < expected.size(); ++i )
            {
                const double allowed = kAbsoluteTolerance + kRelativeTolerance * std::fabs( expected.data()[ i ] );
                worst = std::max( worst, std::fabs( actual.data()[ i ] - expected.data()[ i ] ) / allowed );
            }

            return worst;
        }

        void fillSine( TensorFp32& tensor, double phase )
        {
            for ( dim_t i = 0; i < tensor.size(); ++i )
            {
                tensor.data()[ i ] = static_cast<float>( std::sin( 0.37 * static_cast<double>( i ) + phase ) );
            }
        }
    }

    // ====================================================================
    // A. Construction and validation
    // ====================================================================

    TEST( GatedResidualCpuTests, Construct_HasTheReferenceChildren )
    {
        GatedResidualCpu residual( "residual", config(), Device::Cpu() );

        EXPECT_EQ( residual.getType(), ComponentType::GatedResidual );
        EXPECT_NE( residual.findComponent( "residual.norm" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_down" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_up" ), nullptr );
        EXPECT_NE( residual.findComponent( "residual.fc_inject" ), nullptr );
    }

    TEST( GatedResidualCpuTests, ReadOnly_HasNoInjectProjection )
    {
        GatedResidualCpu residual( "residual", config( false ), Device::Cpu() );

        EXPECT_THROW( (void)residual.findComponent( "residual.fc_inject" ), std::exception );
    }

    TEST( GatedResidualCpuTests, Build_RefusesAStreamOfTheWrongWidth )
    {
        GatedResidualCpu residual( "residual", config(), Device::Cpu() );

        EXPECT_THROW( residual.build( BuildContext( shape_t{ 2, kModelDim }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST( GatedResidualCpuTests, InjectBeforeReadIsRefused )
    {
        auto residual = builtResidual( config(), 2 );
        TensorFp32 stream( Device::Cpu(), shape_t{ 2, kStreams * kModelDim } );
        TensorFp32 output( Device::Cpu(), shape_t{ 2, kModelDim } );

        EXPECT_THROW( (void)residual->inject( stream, output ), std::invalid_argument );
    }

    TEST( GatedResidualCpuTests, InjectOnAReadOnlyResidualIsRefused )
    {
        auto residual = builtResidual( config( false ), 2 );
        TensorFp32 stream( Device::Cpu(), shape_t{ 2, kStreams * kModelDim } );
        TensorFp32 output( Device::Cpu(), shape_t{ 2, kModelDim } );
        fillSine( stream, 0.1 );

        (void)residual->read( stream );

        EXPECT_THROW( (void)residual->inject( stream, output ), std::logic_error );
    }

    // ====================================================================
    // B. Structure of the arithmetic, independent of the reference
    // ====================================================================

    // A zero sublayer output leaves the stream unchanged: the write adds to S, never replaces it with N.
    TEST( GatedResidualCpuTests, Inject_ZeroSublayerOutputLeavesTheStreamUnchanged )
    {
        auto residual = builtResidual( config(), 3 );
        TensorFp32 stream( Device::Cpu(), shape_t{ 3, kStreams * kModelDim } );
        TensorFp32 zero_output( Device::Cpu(), shape_t{ 3, kModelDim } );
        fillSine( stream, 0.4 );
        std::fill( zero_output.data(), zero_output.data() + zero_output.size(), 0.0f );

        (void)residual->read( stream );
        const auto& updated = residual->inject( stream, zero_output );

        for ( dim_t i = 0; i < stream.size(); ++i )
        {
            EXPECT_EQ( updated.data()[ i ], stream.data()[ i ] ) << "at index " << i;
        }
    }

    // Write gates are 2 * sigmoid, so they lie in (0, 2) whatever the weights.
    TEST( GatedResidualCpuTests, InjectionWeightsLieBetweenZeroAndTwo )
    {
        auto residual = builtResidual( config(), 3 );
        TensorFp32 stream( Device::Cpu(), shape_t{ 3, kStreams * kModelDim } );
        fillSine( stream, 0.2 );

        (void)residual->read( stream );
        const auto& weights = residual->injectionWeights();

        ASSERT_EQ( weights.shape(), ( shape_t{ 3, kStreams } ) );

        for ( dim_t i = 0; i < weights.size(); ++i )
        {
            EXPECT_GT( weights.data()[ i ], 0.0f );
            EXPECT_LT( weights.data()[ i ], 2.0f );
        }
    }

    // ====================================================================
    // C. The Qwen 4 reference gate (Qwen4.md Phase 2)
    // ====================================================================

    class GatedResidualReferenceTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
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

        TensorFp32 captured( const std::string& name )
        {
            return tensorFrom( capture_->readTensorBlob<CpuMemoryResource>( name ) );
        }

        std::unique_ptr<GatedResidualCpu> loadedResidual( const std::string& source, bool has_injection )
        {
            auto residual = builtResidual( config( has_injection ), kWidestStep, false );

            loadChild( *residual, "residual.norm", source + ".hc_norm.weight" );
            loadChild( *residual, "residual.fc_down", source + ".input_mix_weight_down.weight" );
            loadChild( *residual, "residual.fc_up", source + ".input_mix_weight_up.weight" );

            if ( has_injection )
            {
                loadChild( *residual, "residual.fc_inject", source + ".block_inject_weight.weight" );
            }

            return residual;
        }

        void loadChild( GatedResidualCpu& residual, const std::string& child, const std::string& source )
        {
            residual.findComponent( child )->loadParameter( "weight", weights_->readTensorBlob<CpuMemoryResource>( source ) );
        }

        std::unique_ptr<Serialization::WeightsReader> capture_;
        std::unique_ptr<Serialization::WeightsReader> weights_;
    };

    TEST_F( GatedResidualReferenceTests, EveryResidualOfEveryStepMatchesTheCapture )
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

                auto residual = loadedResidual( source, true );

                for ( const auto& step : kSteps )
                {
                    const std::string prefix = std::format( "{}.layer{}.", step, layer );
                    const auto stream = captured( prefix + kind + ".stream" );

                    const auto& mixed = residual->read( stream );
                    const auto& weights = residual->injectionWeights();

                    // The sublayer's output, and the stream the next residual (or the next layer) reads.
                    const auto sublayer = captured( prefix + ( kind == "attn_residual" ? "mixer" : "mlp" ) );
                    const auto next = captured( prefix + ( kind == "attn_residual" ? "mlp_residual.stream" : "output" ) );

                    const auto& updated = residual->inject( stream, sublayer );

                    const double errors[] = {
                        worstToleranceMultiple( mixed, captured( prefix + kind + ".mixed" ) ),
                        worstToleranceMultiple( weights, captured( prefix + kind + ".injection" ) ),
                        worstToleranceMultiple( updated, next ) };

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

        std::cout << "[ GatedResidual gate ] worst error " << worst << " x tolerance, at " << worst_at << std::endl;
    }

    TEST_F( GatedResidualReferenceTests, TheFinalMixerMatchesTheCaptureAtEveryStep )
    {
        auto mixer = loadedResidual( "model.hyper_connection_mixer", false );

        for ( const auto& step : kSteps )
        {
            const auto stream = captured( step + ".final_mixer.stream" );
            const auto& mixed = mixer->read( stream );

            EXPECT_LE( worstToleranceMultiple( mixed, captured( step + ".final_mixer.mixed" ) ), 1.0 ) << step;
        }
    }
}
