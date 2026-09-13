/**
 * @file MixtureOfExperts.Cpu.cpp
 * @brief Concrete-component tests for MixtureOfExperts<DeviceType::Cpu, FP32, Gelu>.
 *
 * The Phase 6 gate (Specifications/Gemma4MoE.md), through the component: a hand-built bank against
 * its definition, and HuggingFace's eager Gemma4TextExperts on synthetic weights. The HuggingFace gate
 * skips without the capture from
 * Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_experts_reference.py.
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
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::MixtureOfExperts
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using ExpertsCpu = Mila::Dnn::MixtureOfExperts<DeviceType::Cpu, TensorDataType::FP32, ActivationType::Gelu>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostInt32 = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        constexpr int64_t kHidden = 4;
        constexpr int64_t kIntermediate = 2;
        constexpr int64_t kExperts = 3;
        constexpr int64_t kTopK = 2;

        // Written in Gemma4MoE.md Phase 6 before the first run.
        constexpr double kReferenceTolerance = 1e-5;
        constexpr double kDefinitionTolerance = 1e-6;

        fs::path referencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_experts_reference.safetensors";
        }

        // gelu_pytorch_tanh, in double, independently of Mila's functor.
        double geluTanh( double x )
        {
            return 0.5 * x * ( 1.0 + std::tanh( std::sqrt( 2.0 / 3.141592653589793 ) * ( x + 0.044715 * x * x * x ) ) );
        }

        void loadValues( ExpertsCpu& experts, const std::string& parameter, const std::vector<float>& values, const shape_t& shape )
        {
            const std::size_t bytes = values.size() * sizeof( float );
            Serialization::TensorMetadata meta{ TensorDataType::FP32, shape, bytes };
            Serialization::TensorBlobView blob( meta, values.data(), bytes );

            experts.loadParameter( parameter, blob );
        }

        static_assert( ExpertsCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( ExpertsCpu::getPrecision() == TensorDataType::FP32 );
    }

    class MixtureOfExpertsCpuTests : public ::testing::Test
    {
    protected:
        static MixtureOfExpertsConfig smallConfig()
        {
            return MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK );
        }

        static std::unique_ptr<ExpertsCpu> builtExperts( const MixtureOfExpertsConfig& config, const shape_t& shape )
        {
            auto experts = std::make_unique<ExpertsCpu>( "experts", config, Device::Cpu() );
            experts->build( BuildContext( shape, RuntimeMode::Inference, false ) );

            return experts;
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( MixtureOfExpertsCpuTests, Construct_StandaloneSucceeds )
    {
        ExpertsCpu experts( "experts", smallConfig(), Device::Cpu() );

        EXPECT_EQ( experts.getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( experts.getType(), ComponentType::MixtureOfExperts );
    }

    TEST_F( MixtureOfExpertsCpuTests, Build_ThrowsOnHiddenSizeMismatch )
    {
        ExpertsCpu experts( "experts", smallConfig(), Device::Cpu() );

        EXPECT_THROW( experts.build( BuildContext( shape_t{ 3, kHidden + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( MixtureOfExpertsCpuTests, Forward_ThrowsOnExpertIndexOutOfRange )
    {
        auto experts = builtExperts( smallConfig(), shape_t{ 1, kHidden } );

        HostFp32 input( Device::Cpu(), shape_t{ 1, kHidden } );
        HostFp32 weights( Device::Cpu(), shape_t{ 1, kTopK } );
        HostInt32 indices( Device::Cpu(), shape_t{ 1, kTopK } );

        for ( dim_t i = 0; i < input.size(); ++i ) input.data()[ i ] = 1.0f;
        weights.data()[ 0 ] = 0.5f;
        weights.data()[ 1 ] = 0.5f;
        indices.data()[ 0 ] = 0;
        indices.data()[ 1 ] = static_cast<int32_t>( kExperts );

        EXPECT_THROW( experts->forward( input, weights, indices ), std::out_of_range );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    // Every expert distinguishable, so a wrong row, a swapped gate/up half or a dropped weight all
    // change the answer. Expected values come from the definition in double.
    TEST_F( MixtureOfExpertsCpuTests, Forward_MatchesDefinition )
    {
        constexpr int64_t tokens = 2;
        auto experts = builtExperts( smallConfig(), shape_t{ tokens, kHidden } );

        std::vector<float> gate_up( kExperts * 2 * kIntermediate * kHidden );
        std::vector<float> down( kExperts * kHidden * kIntermediate );

        for ( int64_t e = 0; e < kExperts; ++e )
        {
            for ( int64_t r = 0; r < 2 * kIntermediate; ++r )
            {
                for ( int64_t c = 0; c < kHidden; ++c )
                {
                    gate_up[ ( e * 2 * kIntermediate + r ) * kHidden + c ] =
                        0.1f * static_cast<float>( e + 1 ) * ( r < kIntermediate ? 1.0f : -0.5f ) + 0.01f * static_cast<float>( r * kHidden + c );
                }
            }

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                for ( int64_t i = 0; i < kIntermediate; ++i )
                {
                    down[ ( e * kHidden + j ) * kIntermediate + i ] = 0.2f * static_cast<float>( e + 1 ) - 0.05f * static_cast<float>( j + i );
                }
            }
        }

        loadValues( *experts, "gate_up_proj", gate_up, shape_t{ kExperts, 2 * kIntermediate, kHidden } );
        loadValues( *experts, "down_proj", down, shape_t{ kExperts, kHidden, kIntermediate } );

        HostFp32 input( Device::Cpu(), shape_t{ tokens, kHidden } );
        HostFp32 weights( Device::Cpu(), shape_t{ tokens, kTopK } );
        HostInt32 indices( Device::Cpu(), shape_t{ tokens, kTopK } );

        for ( dim_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = -1.0f + 0.375f * static_cast<float>( i );
        }

        const int32_t selections[ tokens * kTopK ] = { 2, 0, 1, 2 };
        const float combine[ tokens * kTopK ] = { 0.75f, 0.25f, 1.5f, -0.5f };

        for ( int64_t s = 0; s < tokens * kTopK; ++s )
        {
            indices.data()[ s ] = selections[ s ];
            weights.data()[ s ] = combine[ s ];
        }

        auto& output = experts->forward( input, weights, indices );

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
                        gate += gate_up[ ( e * 2 * kIntermediate + i ) * kHidden + c ] * input.data()[ t * kHidden + c ];
                        up += gate_up[ ( e * 2 * kIntermediate + kIntermediate + i ) * kHidden + c ] * input.data()[ t * kHidden + c ];
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
                EXPECT_NEAR( output.data()[ t * kHidden + j ], expected[ j ], kDefinitionTolerance )
                    << "token " << t << " column " << j;
            }
        }
    }

    // Gemma4MoE.md Phase 6, "HuggingFace, CPU, FP32".
    TEST_F( MixtureOfExpertsCpuTests, Forward_MatchesHuggingFaceReference )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "experts reference not present at: " << referencePath().string();
        }

        Serialization::PretrainedModelReader reader( referencePath() );
        auto read = [&]( const std::string& name ) { return reader.readTensorBlob<CpuMemoryResource>( name ); };

        auto hidden_states = read( "hidden_states" );
        auto gate_up = read( "gate_up_proj" );
        auto down = read( "down_proj" );
        auto reference_indices = read( "indices" );
        auto reference_weights = read( "weights" );
        auto reference_output = read( "output" );

        const auto& gate_up_shape = gate_up.getMetadata().shape;
        const int64_t tokens = static_cast<int64_t>( hidden_states.getMetadata().shape[ 0 ] );
        const int64_t hidden = static_cast<int64_t>( hidden_states.getMetadata().shape[ 1 ] );
        const int64_t experts_count = static_cast<int64_t>( gate_up_shape[ 0 ] );
        const int64_t intermediate = static_cast<int64_t>( gate_up_shape[ 1 ] ) / 2;
        const int64_t top_k = static_cast<int64_t>( reference_indices.getMetadata().shape[ 1 ] );

        auto experts = builtExperts( MixtureOfExpertsConfig( hidden, intermediate, experts_count, top_k ), shape_t{ tokens, hidden } );

        experts->loadParameter( "gate_up_proj", gate_up );
        experts->loadParameter( "down_proj", down );

        HostFp32 input( Device::Cpu(), shape_t{ tokens, hidden } );
        HostFp32 weights( Device::Cpu(), shape_t{ tokens, top_k } );
        HostInt32 indices( Device::Cpu(), shape_t{ tokens, top_k } );

        std::memcpy( input.data(), hidden_states.data(), hidden_states.sizeBytes() );
        std::memcpy( weights.data(), reference_weights.data(), reference_weights.sizeBytes() );
        std::memcpy( indices.data(), reference_indices.data(), reference_indices.sizeBytes() );

        auto& output = experts->forward( input, weights, indices );

        const auto* expected = static_cast<const float*>( static_cast<const void*>( reference_output.data() ) );

        ASSERT_EQ( output.size() * sizeof( float ), reference_output.sizeBytes() );

        int64_t mismatches = 0;
        double worst = 0.0;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            const double error = std::fabs( static_cast<double>( output.data()[ i ] ) - expected[ i ] );
            worst = std::max( worst, error );

            if ( error > kReferenceTolerance )
            {
                ++mismatches;
            }
        }

        EXPECT_EQ( mismatches, 0 ) << "of " << output.size() << " elements; worst error " << worst;

        std::cout << std::format( "[ reference ] experts {} x {} top-{}, {} tokens: worst error {:.3e}\n",
            experts_count, intermediate, top_k, tokens, worst );
    }

    // ====================================================================
    // G. Parameters & serialization
    // ====================================================================

    TEST_F( MixtureOfExpertsCpuTests, FlatNames_AreTheCheckpointVocabulary )
    {
        auto experts = builtExperts( smallConfig(), shape_t{ 1, kHidden } );

        const fs::path path = fs::temp_directory_path() / "mila_experts_flat_names.safetensors";

        {
            Serialization::SafeTensorsWriter writer( path );
            experts->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Declare );
            writer.beginData();
            experts->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Write );
            writer.close();
        }

        std::vector<std::string> names;
        {
            Serialization::PretrainedModelReader reader( path );
            names = reader.getTensorNames();
        }

        std::error_code ignored;
        fs::remove( path, ignored );

        std::sort( names.begin(), names.end() );

        EXPECT_EQ( names, ( std::vector<std::string>{ "experts.down_proj", "experts.gate_up_proj" } ) );
        EXPECT_EQ( experts->parameterCount(), kExperts * 3 * kIntermediate * kHidden );
    }

    TEST_F( MixtureOfExpertsCpuTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ 3, kHidden }, RuntimeMode::Inference, false );

        ExpertsCpu predictor( "experts", smallConfig(), Device::Cpu() );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        ExpertsCpu built( "experts", smallConfig(), Device::Cpu() );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }
}
