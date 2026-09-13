/**
 * @file Router.Cpu.cpp
 * @brief Concrete-component tests for Router<DeviceType::Cpu, FP32>.
 *
 * The HuggingFace reference gate runs here, through the component, so it runs in CI whenever the
 * capture is present: Specifications/Gemma4MoE.md Phase 5 fixes its tolerances, written before the
 * first run. The capture is produced by
 * Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_router_reference.py and the gate skips
 * without it.
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
#include <set>
#include <string>
#include <stdexcept>
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

        using RouterCpu = Mila::Dnn::Router<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kHidden = 8;
        constexpr int64_t kExperts = 4;
        constexpr int64_t kTopK = 2;

        // Written in Gemma4MoE.md Phase 5 before the first run.
        constexpr float kReferenceWeightTolerance = 1e-5f;
        constexpr float kReferenceLogitTieWidth = 1e-5f;

        fs::path referencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_router_layer5_reference.safetensors";
        }

        template<typename TValue>
        void loadInto( auto& target, const std::string& parameter, const std::vector<TValue>& values, const shape_t& shape )
        {
            const std::size_t bytes = values.size() * sizeof( TValue );
            Serialization::TensorMetadata meta{ TensorDataType::FP32, shape, bytes };
            Serialization::TensorBlobView blob( meta, values.data(), bytes );

            target.loadParameter( parameter, blob );
        }

        static_assert( RouterCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( RouterCpu::getPrecision() == TensorDataType::FP32 );
    }

    class RouterCpuTests : public ::testing::Test
    {
    protected:
        static RouterConfig smallConfig()
        {
            return RouterConfig( kHidden, kExperts, kTopK );
        }

        static std::unique_ptr<RouterCpu> builtRouter( const RouterConfig& config, const shape_t& shape )
        {
            auto router = std::make_unique<RouterCpu>( "router", config, Device::Cpu() );
            router->build( BuildContext( shape, RuntimeMode::Inference, false ) );

            return router;
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( RouterCpuTests, Construct_StandaloneSucceeds )
    {
        RouterCpu router( "router", smallConfig(), Device::Cpu() );

        EXPECT_EQ( router.getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( router.getType(), ComponentType::Router );
    }

    TEST_F( RouterCpuTests, Build_ThrowsOnHiddenSizeMismatch )
    {
        RouterCpu router( "router", smallConfig(), Device::Cpu() );

        EXPECT_THROW( router.build( BuildContext( shape_t{ 3, kHidden + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( RouterCpuTests, Forward_ThrowsBeforeBuild )
    {
        RouterCpu router( "router", smallConfig(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 3, kHidden } );

        EXPECT_THROW( router.forward( input ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    // Experts 1 and 2 share a projection row, so their logits are EXACTLY equal; expert 0 leads and
    // expert 3 trails. Top-2 is therefore {0, 1}: the tie goes to the lower index. The weights are
    // softmax over all four, renormalized over the two, times per_expert_scale -- computed here from
    // the definition, in double.
    TEST_F( RouterCpuTests, Forward_ExactTieSelectsLowerIndexAndWeightsMatchDefinition )
    {
        auto router = builtRouter( smallConfig(), shape_t{ 1, kHidden } );

        const std::vector<float> scale( kHidden, 1.0f );
        const std::vector<float> per_expert_scale{ 1.0f, 0.5f, 2.0f, 1.0f };
        std::vector<float> projection( kExperts * kHidden );

        for ( int64_t i = 0; i < kHidden; ++i )
        {
            projection[ 0 * kHidden + i ] = 2.0f / kHidden;
            projection[ 1 * kHidden + i ] = 1.0f / kHidden;
            projection[ 2 * kHidden + i ] = 1.0f / kHidden;
            projection[ 3 * kHidden + i ] = -1.0f / kHidden;
        }

        loadInto( *router->findComponent( "router.proj" ), "weight", projection, shape_t{ kExperts, kHidden } );
        loadInto( *router, "scale", scale, shape_t{ kHidden } );
        loadInto( *router, "per_expert_scale", per_expert_scale, shape_t{ kExperts } );

        TensorFp32 input( Device::Cpu(), shape_t{ 1, kHidden } );

        for ( int64_t i = 0; i < kHidden; ++i )
        {
            input.data()[ i ] = 0.25f + 0.125f * static_cast<float>( i );
        }

        auto routing = router->forward( input );

        // The definition, independently: unscaled RMS norm, times scale and hidden^-0.5, projected.
        double sum_of_squares = 0.0;

        for ( int64_t i = 0; i < kHidden; ++i )
        {
            sum_of_squares += static_cast<double>( input.data()[ i ] ) * input.data()[ i ];
        }

        const double rstd = 1.0 / std::sqrt( sum_of_squares / kHidden + 1e-6 );
        std::vector<double> logits( kExperts, 0.0 );

        for ( int64_t e = 0; e < kExperts; ++e )
        {
            for ( int64_t i = 0; i < kHidden; ++i )
            {
                logits[ e ] += projection[ e * kHidden + i ] * input.data()[ i ] * rstd / std::sqrt( static_cast<double>( kHidden ) );
            }
        }

        const double selected_mass = std::exp( logits[ 0 ] ) + std::exp( logits[ 1 ] );
        const double expected_weight_0 = std::exp( logits[ 0 ] ) / selected_mass * per_expert_scale[ 0 ];
        const double expected_weight_1 = std::exp( logits[ 1 ] ) / selected_mass * per_expert_scale[ 1 ];

        ASSERT_EQ( routing.indices.shape(), ( shape_t{ 1, kTopK } ) );

        std::set<int32_t> selected{ routing.indices.data()[ 0 ], routing.indices.data()[ 1 ] };
        EXPECT_EQ( selected, ( std::set<int32_t>{ 0, 1 } ) );

        for ( int64_t slot = 0; slot < kTopK; ++slot )
        {
            const int32_t expert = routing.indices.data()[ slot ];
            const double expected = expert == 0 ? expected_weight_0 : expected_weight_1;

            EXPECT_NEAR( routing.weights.data()[ slot ], expected, 1e-6 ) << "expert " << expert;
        }
    }

    // The HuggingFace gate (Gemma4MoE.md Phase 5, "FP32, end to end").
    TEST_F( RouterCpuTests, Forward_MatchesHuggingFaceReferenceLayer5 )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "router reference not present at: " << referencePath().string();
        }

        Serialization::PretrainedModelReader reader( referencePath() );

        auto read = [&]( const std::string& name ) { return reader.readTensorBlob<CpuMemoryResource>( name ); };

        auto hidden_states = read( "hidden_states" );
        const auto& hidden_shape = hidden_states.getMetadata().shape;
        const int64_t tokens = static_cast<int64_t>( hidden_shape[ 0 ] );
        const int64_t hidden = static_cast<int64_t>( hidden_shape[ 1 ] );
        auto per_expert_scale = read( "per_expert_scale" );
        const int64_t experts = static_cast<int64_t>( per_expert_scale.getMetadata().shape[ 0 ] );
        auto reference_indices = read( "fp32.indices" );
        const int64_t top_k = static_cast<int64_t>( reference_indices.getMetadata().shape[ 1 ] );

        auto router = builtRouter( RouterConfig( hidden, experts, top_k ), shape_t{ tokens, hidden } );

        router->findComponent( "router.proj" )->loadParameter( "weight", read( "proj_weight" ) );
        router->loadParameter( "scale", read( "scale" ) );
        router->loadParameter( "per_expert_scale", per_expert_scale );

        TensorFp32 input( Device::Cpu(), shape_t{ tokens, hidden } );
        std::memcpy( input.data(), hidden_states.data(), hidden_states.sizeBytes() );

        auto routing = router->forward( input );

        const auto reference_logits_blob = read( "fp32.logits" );
        const auto reference_weights_blob = read( "fp32.weights" );
        const auto* reference_logits = static_cast<const float*>( static_cast<const void*>( reference_logits_blob.data() ) );
        const auto* reference_weights = static_cast<const float*>( static_cast<const void*>( reference_weights_blob.data() ) );
        const auto* reference_index_data = static_cast<const int32_t*>( static_cast<const void*>( reference_indices.data() ) );

        int64_t rows_with_different_sets = 0;
        int64_t unexplained_set_differences = 0;
        int64_t weight_mismatches = 0;
        double worst_weight_error = 0.0;

        for ( int64_t row = 0; row < tokens; ++row )
        {
            std::vector<std::pair<int32_t, float>> mila;
            std::vector<std::pair<int32_t, float>> reference;

            for ( int64_t slot = 0; slot < top_k; ++slot )
            {
                mila.emplace_back( routing.indices.data()[ row * top_k + slot ], routing.weights.data()[ row * top_k + slot ] );
                reference.emplace_back( reference_index_data[ row * top_k + slot ], reference_weights[ row * top_k + slot ] );
            }

            std::set<int32_t> mila_set;
            std::set<int32_t> reference_set;

            for ( const auto& [ expert, weight ] : mila ) mila_set.insert( expert );
            for ( const auto& [ expert, weight ] : reference ) reference_set.insert( expert );

            if ( mila_set != reference_set )
            {
                ++rows_with_different_sets;

                const float* row_logits = reference_logits + row * experts;

                for ( int32_t only_mila : mila_set )
                {
                    if ( reference_set.contains( only_mila ) )
                    {
                        continue;
                    }

                    bool within_tie_width = false;

                    for ( int32_t only_reference : reference_set )
                    {
                        if ( !mila_set.contains( only_reference )
                             && std::fabs( row_logits[ only_mila ] - row_logits[ only_reference ] ) <= kReferenceLogitTieWidth )
                        {
                            within_tie_width = true;
                        }
                    }

                    if ( !within_tie_width )
                    {
                        ++unexplained_set_differences;
                    }
                }
            }

            for ( const auto& [ expert, weight ] : mila )
            {
                const auto match = std::find_if( reference.begin(), reference.end(),
                    [expert]( const auto& entry ) { return entry.first == expert; } );

                if ( match == reference.end() )
                {
                    continue;
                }

                const double error = std::fabs( static_cast<double>( weight ) - match->second );
                worst_weight_error = std::max( worst_weight_error, error );

                if ( error > kReferenceWeightTolerance )
                {
                    ++weight_mismatches;
                }
            }
        }

        EXPECT_EQ( unexplained_set_differences, 0 ) << rows_with_different_sets << " rows chose a different set";
        EXPECT_EQ( weight_mismatches, 0 ) << "worst weight error " << worst_weight_error;

        std::cout << std::format( "[ reference ] layer 5, {} tokens: {} rows with a different set, worst weight error {:.3e}\n",
            tokens, rows_with_different_sets, worst_weight_error );
    }

    // ====================================================================
    // G. Parameters & serialization
    // ====================================================================

    TEST_F( RouterCpuTests, ParameterCount_ExcludesDerivedNormWeight )
    {
        auto router = builtRouter( smallConfig(), shape_t{ 3, kHidden } );

        // proj [4 x 8] + scale [8] + per_expert_scale [4]; the norm child's [8] weight is derived.
        EXPECT_EQ( router->parameterCount(), kExperts * kHidden + kHidden + kExperts );
    }

    TEST_F( RouterCpuTests, FlatNames_AreTheCheckpointVocabulary )
    {
        auto router = builtRouter( smallConfig(), shape_t{ 3, kHidden } );

        const fs::path path = fs::temp_directory_path() / "mila_router_flat_names.safetensors";

        {
            Serialization::SafeTensorsWriter writer( path );
            router->saveFlatTensors( writer, "router", Serialization::TensorSavePass::Declare );
            writer.beginData();
            router->saveFlatTensors( writer, "router", Serialization::TensorSavePass::Write );
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

        EXPECT_EQ( names, ( std::vector<std::string>{ "router.per_expert_scale", "router.proj.weight", "router.scale" } ) );
    }

    TEST_F( RouterCpuTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ 3, kHidden }, RuntimeMode::Inference, false );

        RouterCpu predictor( "router", smallConfig(), Device::Cpu() );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        RouterCpu built( "router", smallConfig(), Device::Cpu() );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }
}
