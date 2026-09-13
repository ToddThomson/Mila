/**
 * @file MixtureOfExperts.Cuda.cpp
 * @brief Concrete-component tests for MixtureOfExperts<DeviceType::Cuda, {FP32, BF16}, Gelu>.
 *
 * The CUDA half of the Phase 6 gate (Specifications/Gemma4MoE.md), through the component, with the
 * tolerances the record fixed before the first run. The HuggingFace gates skip without the capture
 * from Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_experts_reference.py.
 *
 * The two cards disagree in the last bits: pin one by UUID (CUDA_VISIBLE_DEVICES) and name it with
 * any number reported from here.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::MixtureOfExperts
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostInt32 = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using DeviceInt32 = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;

        constexpr int64_t kHidden = 4;
        constexpr int64_t kIntermediate = 2;
        constexpr int64_t kExperts = 3;
        constexpr int64_t kTopK = 2;

        // Written in Gemma4MoE.md Phase 6, "CUDA gate", before the first run.
        constexpr double kFp32ReferenceTolerance = 1e-5;
        constexpr double kFp32DefinitionTolerance = 1e-6;
        constexpr double kBf16Relative = 0.05;
        constexpr double kBf16Absolute = 0.01;

        fs::path referencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_experts_reference.safetensors";
        }

        double geluTanh( double x )
        {
            return 0.5 * x * ( 1.0 + std::tanh( std::sqrt( 2.0 / 3.141592653589793 ) * ( x + 0.044715 * x * x * x ) ) );
        }

        bool withinTolerance( TensorDataType precision, double actual, double expected, double fp32_tolerance )
        {
            const double error = std::fabs( actual - expected );

            if ( precision == TensorDataType::BF16 )
            {
                return error <= kBf16Relative * std::fabs( expected ) + kBf16Absolute;
            }

            return error <= fp32_tolerance;
        }

        // FP32 values into a parameter of either precision; bfloat16 is the top 16 bits of the float32 pattern.
        template<typename TExperts>
        void loadValues( TExperts& experts, const std::string& parameter, TensorDataType precision,
            const float* values, std::size_t count, const shape_t& shape )
        {
            if ( precision == TensorDataType::BF16 )
            {
                std::vector<std::uint16_t> bits( count );

                for ( std::size_t i = 0; i < count; ++i )
                {
                    bits[ i ] = static_cast<std::uint16_t>( std::bit_cast<std::uint32_t>( values[ i ] ) >> 16 );
                }

                const std::size_t bytes = count * sizeof( std::uint16_t );
                Serialization::TensorMetadata meta{ TensorDataType::BF16, shape, bytes };
                Serialization::TensorBlobView blob( meta, bits.data(), bytes );

                experts.loadParameter( parameter, blob );
            }
            else
            {
                const std::size_t bytes = count * sizeof( float );
                Serialization::TensorMetadata meta{ TensorDataType::FP32, shape, bytes };
                Serialization::TensorBlobView blob( meta, values, bytes );

                experts.loadParameter( parameter, blob );
            }

            experts.synchronize();
        }
    }

    class MixtureOfExpertsCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        template<TensorDataType TPrecision>
        std::vector<float> run( Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu>& experts,
            const float* input, const float* weights, const std::int32_t* indices, int64_t tokens, int64_t hidden, int64_t top_k )
        {
            using DeviceTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

            HostFp32 host_input( Device::Cpu(), shape_t{ tokens, hidden } );
            HostFp32 host_weights( Device::Cpu(), shape_t{ tokens, top_k } );
            HostInt32 host_indices( Device::Cpu(), shape_t{ tokens, top_k } );

            std::memcpy( host_input.data(), input, static_cast<std::size_t>( tokens * hidden ) * sizeof( float ) );
            std::memcpy( host_weights.data(), weights, static_cast<std::size_t>( tokens * top_k ) * sizeof( float ) );
            std::memcpy( host_indices.data(), indices, static_cast<std::size_t>( tokens * top_k ) * sizeof( std::int32_t ) );

            DeviceTensor device_input( Device::Cuda( 0 ), shape_t{ tokens, hidden } );
            DeviceTensor device_weights( Device::Cuda( 0 ), shape_t{ tokens, top_k } );
            DeviceInt32 device_indices( Device::Cuda( 0 ), shape_t{ tokens, top_k } );

            copy( host_input, device_input, context_.get() );
            copy( host_weights, device_weights, context_.get() );
            copy( host_indices, device_indices, context_.get() );
            context_->synchronize();

            auto& output = experts.forward( device_input, device_weights, device_indices );
            experts.synchronize();

            auto host_output = toHost<TensorDataType::FP32>( output, context_.get() );
            context_->synchronize();

            return std::vector<float>( host_output.data(), host_output.data() + host_output.size() );
        }

        template<TensorDataType TPrecision>
        void expectMatchesHuggingFaceReference()
        {
            const TensorDataType precision = TPrecision;

            Serialization::PretrainedModelReader reader( referencePath() );
            auto read = [&]( const std::string& name ) { return reader.readTensorBlob<CpuMemoryResource>( name ); };
            auto floats = []( const auto& blob ) { return static_cast<const float*>( static_cast<const void*>( blob.data() ) ); };

            auto hidden_states = read( "hidden_states" );
            auto gate_up = read( "gate_up_proj" );
            auto down = read( "down_proj" );
            auto indices = read( "indices" );
            auto weights = read( "weights" );
            auto reference = read( "output" );

            const auto& gate_up_shape = gate_up.getMetadata().shape;
            const int64_t tokens = static_cast<int64_t>( hidden_states.getMetadata().shape[ 0 ] );
            const int64_t hidden = static_cast<int64_t>( hidden_states.getMetadata().shape[ 1 ] );
            const int64_t experts_count = static_cast<int64_t>( gate_up_shape[ 0 ] );
            const int64_t intermediate = static_cast<int64_t>( gate_up_shape[ 1 ] ) / 2;
            const int64_t top_k = static_cast<int64_t>( indices.getMetadata().shape[ 1 ] );

            Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu> experts(
                "experts", MixtureOfExpertsConfig( hidden, intermediate, experts_count, top_k ), Device::Cuda( 0 ) );
            experts.build( BuildContext( shape_t{ tokens, hidden }, RuntimeMode::Inference, false ) );

            loadValues( experts, "gate_up_proj", precision, floats( gate_up ),
                static_cast<std::size_t>( experts_count * 2 * intermediate * hidden ), shape_t{ experts_count, 2 * intermediate, hidden } );
            loadValues( experts, "down_proj", precision, floats( down ),
                static_cast<std::size_t>( experts_count * hidden * intermediate ), shape_t{ experts_count, hidden, intermediate } );

            const std::vector<float> output = run( experts, floats( hidden_states ), floats( weights ),
                static_cast<const std::int32_t*>( static_cast<const void*>( indices.data() ) ), tokens, hidden, top_k );

            const float* expected = floats( reference );

            ASSERT_EQ( output.size() * sizeof( float ), reference.sizeBytes() );

            int64_t mismatches = 0;
            double worst = 0.0;

            for ( std::size_t i = 0; i < output.size(); ++i )
            {
                worst = std::max( worst, std::fabs( static_cast<double>( output[ i ] ) - expected[ i ] ) );

                if ( !withinTolerance( precision, output[ i ], expected[ i ], kFp32ReferenceTolerance ) )
                {
                    ++mismatches;
                }
            }

            EXPECT_EQ( mismatches, 0 ) << "of " << output.size() << " elements; worst error " << worst;

            std::cout << std::format( "[ reference ] {} experts {} x {} top-{}, {} tokens: worst error {:.3e}\n",
                precision == TensorDataType::BF16 ? "BF16" : "FP32", experts_count, intermediate, top_k, tokens, worst );
        }

        template<TensorDataType TPrecision>
        void expectMatchesDefinition()
        {
            const TensorDataType precision = TPrecision;
            constexpr int64_t tokens = 2;

            Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu> experts(
                "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
            experts.build( BuildContext( shape_t{ tokens, kHidden }, RuntimeMode::Inference, false ) );

            std::vector<float> gate_up( kExperts * 2 * kIntermediate * kHidden );
            std::vector<float> down( kExperts * kHidden * kIntermediate );

            for ( int64_t e = 0; e < kExperts; ++e )
            {
                for ( int64_t r = 0; r < 2 * kIntermediate; ++r )
                {
                    for ( int64_t c = 0; c < kHidden; ++c )
                    {
                        gate_up[ ( e * 2 * kIntermediate + r ) * kHidden + c ] =
                            0.125f * static_cast<float>( e + 1 ) * ( r < kIntermediate ? 1.0f : -0.5f ) + 0.0625f * static_cast<float>( r * kHidden + c );
                    }
                }

                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    for ( int64_t i = 0; i < kIntermediate; ++i )
                    {
                        down[ ( e * kHidden + j ) * kIntermediate + i ] = 0.25f * static_cast<float>( e + 1 ) - 0.0625f * static_cast<float>( j + i );
                    }
                }
            }

            loadValues( experts, "gate_up_proj", precision, gate_up.data(), gate_up.size(), shape_t{ kExperts, 2 * kIntermediate, kHidden } );
            loadValues( experts, "down_proj", precision, down.data(), down.size(), shape_t{ kExperts, kHidden, kIntermediate } );

            std::vector<float> input( tokens * kHidden );

            for ( std::size_t i = 0; i < input.size(); ++i )
            {
                input[ i ] = -1.0f + 0.375f * static_cast<float>( i );
            }

            const std::int32_t selections[ tokens * kTopK ] = { 2, 0, 1, 2 };
            const float combine[ tokens * kTopK ] = { 0.75f, 0.25f, 1.5f, -0.5f };

            const std::vector<float> output = run( experts, input.data(), combine, selections, tokens, kHidden, kTopK );

            for ( int64_t t = 0; t < tokens; ++t )
            {
                std::vector<double> expected( kHidden, 0.0 );

                for ( int64_t slot = 0; slot < kTopK; ++slot )
                {
                    const int64_t e = selections[ t * kTopK + slot ];
                    std::vector<double> gated( kIntermediate );

                    for ( int64_t i = 0; i < kIntermediate; ++i )
                    {
                        double gate = 0.0;
                        double up = 0.0;

                        for ( int64_t c = 0; c < kHidden; ++c )
                        {
                            gate += gate_up[ ( e * 2 * kIntermediate + i ) * kHidden + c ] * input[ t * kHidden + c ];
                            up += gate_up[ ( e * 2 * kIntermediate + kIntermediate + i ) * kHidden + c ] * input[ t * kHidden + c ];
                        }

                        gated[ i ] = geluTanh( gate ) * up;
                    }

                    for ( int64_t j = 0; j < kHidden; ++j )
                    {
                        double projected = 0.0;

                        for ( int64_t i = 0; i < kIntermediate; ++i )
                        {
                            projected += down[ ( e * kHidden + j ) * kIntermediate + i ] * gated[ i ];
                        }

                        expected[ j ] += combine[ t * kTopK + slot ] * projected;
                    }
                }

                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    EXPECT_TRUE( withinTolerance( precision, output[ t * kHidden + j ], expected[ j ], kFp32DefinitionTolerance ) )
                        << "token " << t << " column " << j << ": " << output[ t * kHidden + j ] << " vs " << expected[ j ];
                }
            }
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( MixtureOfExpertsCudaTests, Fp32_MatchesDefinition )
    {
        expectMatchesDefinition<TensorDataType::FP32>();
    }

    TEST_F( MixtureOfExpertsCudaTests, Bf16_MatchesDefinition )
    {
        expectMatchesDefinition<TensorDataType::BF16>();
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp32_MatchesHuggingFaceReference )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "experts reference not present at: " << referencePath().string();
        }

        expectMatchesHuggingFaceReference<TensorDataType::FP32>();
    }

    TEST_F( MixtureOfExpertsCudaTests, Bf16_MatchesHuggingFaceReference )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "experts reference not present at: " << referencePath().string();
        }

        expectMatchesHuggingFaceReference<TensorDataType::BF16>();
    }

    // A kernel cannot throw: an out-of-range index poisons that token and leaves the others finite.
    TEST_F( MixtureOfExpertsCudaTests, Fp32_OutOfRangeExpertPoisonsOnlyThatToken )
    {
        constexpr int64_t tokens = 2;

        Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TensorDataType::FP32, ActivationType::Gelu> experts(
            "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        experts.build( BuildContext( shape_t{ tokens, kHidden }, RuntimeMode::Inference, true ) );

        const std::vector<float> input( tokens * kHidden, 0.5f );
        const float combine[ tokens * kTopK ] = { 0.5f, 0.5f, 0.5f, 0.5f };
        const std::int32_t selections[ tokens * kTopK ] = { 0, 1, 2, static_cast<std::int32_t>( kExperts ) };

        const std::vector<float> output = run( experts, input.data(), combine, selections, tokens, kHidden, kTopK );

        for ( int64_t j = 0; j < kHidden; ++j )
        {
            EXPECT_TRUE( std::isfinite( output[ j ] ) ) << "valid token, column " << j;
            EXPECT_TRUE( std::isnan( output[ kHidden + j ] ) ) << "poisoned token, column " << j;
        }
    }

    // ====================================================================
    // G. Footprint -- includes the op's gated scratch
    // ====================================================================

    TEST_F( MixtureOfExpertsCudaTests, Bf16_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ 3, kHidden }, RuntimeMode::Inference, false );

        Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu> predictor(
            "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu> built(
            "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";

        // The gated scratch is part of the state: 3 tokens x top-2 x intermediate 2, FP32.
        EXPECT_GE( actual.device_state_bytes, static_cast<std::size_t>( 3 * kTopK * kIntermediate ) * sizeof( float ) );
    }
}
