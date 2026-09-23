/**
 * @file Router.Cuda.cpp
 * @brief Concrete-component tests for Router<DeviceType::Cuda, {FP32, BF16}>.
 *
 * The CUDA half of the Phase 5 gate (Specifications/Gemma4MoE.md), through the component. FP32 is
 * held to the same tolerance as the CPU router; BF16 is held against the FP32 reference with the
 * bfloat16-resolution allowance the record fixes, because a BF16 projection rounds before anything
 * is ranked. The reference gates skip without the capture from
 * Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_router_reference.py.
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
#include <functional>
#include <iostream>
#include <memory>
#include <set>
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

        constexpr int64_t kHidden = 8;
        constexpr int64_t kExperts = 4;
        constexpr int64_t kTopK = 2;

        // Written in Gemma4MoE.md Phase 5 before the first run.
        constexpr double kFp32WeightTolerance = 1e-5;
        constexpr double kFp32LogitTieWidth = 1e-5;
        constexpr double kBf16Relative = 1.0 / 64.0;

        fs::path referencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_router_layer5_reference.safetensors";
        }

        // bfloat16 is exactly the top 16 bits of the IEEE float32 pattern.
        std::vector<std::uint16_t> toBf16Bits( const float* values, std::size_t count )
        {
            std::vector<std::uint16_t> bits( count );

            for ( std::size_t i = 0; i < count; ++i )
            {
                bits[ i ] = static_cast<std::uint16_t>( std::bit_cast<std::uint32_t>( values[ i ] ) >> 16 );
            }

            return bits;
        }

        // Loads FP32 values into a parameter of either precision, truncating to bfloat16 for BF16.
        void loadValues( auto& target, const std::string& parameter, TensorDataType precision,
            const float* values, std::size_t count, const shape_t& shape )
        {
            if ( precision == TensorDataType::BF16 )
            {
                const auto bits = toBf16Bits( values, count );
                const std::size_t bytes = bits.size() * sizeof( std::uint16_t );
                Serialization::TensorMetadata meta{ TensorDataType::BF16, shape, bytes };
                Serialization::TensorBlobView blob( meta, bits.data(), bytes );

                target.loadParameter( parameter, blob );
            }
            else
            {
                const std::size_t bytes = count * sizeof( float );
                Serialization::TensorMetadata meta{ TensorDataType::FP32, shape, bytes };
                Serialization::TensorBlobView blob( meta, values, bytes );

                target.loadParameter( parameter, blob );
            }

            target.synchronize();
        }

        struct RoutingResult
        {
            std::vector<float> weights;
            std::vector<std::int32_t> indices;
        };

        struct ReferenceComparison
        {
            int64_t rows_with_different_sets{ 0 };
            int64_t unexplained_set_differences{ 0 };
            int64_t weight_mismatches{ 0 };
            double worst_weight_error{ 0.0 };
        };

        // Set-based: the combine is a sum over the selected experts, so order carries no meaning.
        ReferenceComparison compareToReference(
            const RoutingResult& mila, const float* reference_logits, const float* reference_weights,
            const std::int32_t* reference_indices, int64_t tokens, int64_t experts, int64_t top_k,
            const std::function<bool( double, double )>& logits_tie,
            const std::function<bool( double, double )>& weights_agree )
        {
            ReferenceComparison comparison;

            for ( int64_t row = 0; row < tokens; ++row )
            {
                std::set<std::int32_t> mila_set;
                std::set<std::int32_t> reference_set;

                for ( int64_t slot = 0; slot < top_k; ++slot )
                {
                    mila_set.insert( mila.indices[ row * top_k + slot ] );
                    reference_set.insert( reference_indices[ row * top_k + slot ] );
                }

                if ( mila_set != reference_set )
                {
                    ++comparison.rows_with_different_sets;

                    const float* row_logits = reference_logits + row * experts;

                    for ( std::int32_t only_mila : mila_set )
                    {
                        if ( reference_set.contains( only_mila ) )
                        {
                            continue;
                        }

                        const bool explained = std::any_of( reference_set.begin(), reference_set.end(),
                            [&]( std::int32_t only_reference )
                            {
                                return !mila_set.contains( only_reference )
                                    && logits_tie( row_logits[ only_mila ], row_logits[ only_reference ] );
                            } );

                        if ( !explained )
                        {
                            ++comparison.unexplained_set_differences;
                        }
                    }
                }

                for ( int64_t slot = 0; slot < top_k; ++slot )
                {
                    const std::int32_t expert = mila.indices[ row * top_k + slot ];

                    for ( int64_t reference_slot = 0; reference_slot < top_k; ++reference_slot )
                    {
                        if ( reference_indices[ row * top_k + reference_slot ] != expert )
                        {
                            continue;
                        }

                        const double reference = reference_weights[ row * top_k + reference_slot ];
                        const double actual = mila.weights[ row * top_k + slot ];

                        comparison.worst_weight_error = std::max( comparison.worst_weight_error, std::fabs( actual - reference ) );

                        if ( !weights_agree( actual, reference ) )
                        {
                            ++comparison.weight_mismatches;
                        }
                    }
                }
            }

            return comparison;
        }
    }

    class RouterCudaTests : public ::testing::Test
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
        RoutingResult route( Router<DeviceType::Cuda, TPrecision>& router, const float* hidden, int64_t tokens, int64_t hidden_size )
        {
            HostFp32 host_input( Device::Cpu(), shape_t{ tokens, hidden_size } );
            std::memcpy( host_input.data(), hidden, static_cast<std::size_t>( tokens * hidden_size ) * sizeof( float ) );

            Tensor<TPrecision, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ tokens, hidden_size } );
            copy( host_input, input, context_.get() );
            context_->synchronize();

            auto routing = router.forward( input );
            router.synchronize();

            auto host_weights = toHost<TensorDataType::FP32>( routing.weights, context_.get() );

            HostInt32 host_indices( Device::Cpu(), routing.indices.shape() );
            copy( routing.indices, host_indices, context_.get() );
            context_->synchronize();

            RoutingResult result;
            result.weights.assign( host_weights.data(), host_weights.data() + host_weights.size() );
            result.indices.assign( host_indices.data(), host_indices.data() + host_indices.size() );

            return result;
        }

        template<TensorDataType TPrecision>
        ReferenceComparison routeReference( TensorDataType precision,
            const std::function<bool( double, double )>& logits_tie,
            const std::function<bool( double, double )>& weights_agree )
        {
            Serialization::WeightsReader reader( referencePath() );
            auto read = [&]( const std::string& name ) { return reader.readTensorBlob<CpuMemoryResource>( name ); };
            auto as_floats = []( const auto& blob ) { return static_cast<const float*>( static_cast<const void*>( blob.data() ) ); };

            auto hidden_states = read( "hidden_states" );
            auto projection = read( "proj_weight" );
            auto scale = read( "scale" );
            auto per_expert_scale = read( "per_expert_scale" );
            auto reference_logits = read( "fp32.logits" );
            auto reference_weights = read( "fp32.weights" );
            auto reference_indices = read( "fp32.indices" );

            const int64_t tokens = static_cast<int64_t>( hidden_states.getMetadata().shape[ 0 ] );
            const int64_t hidden = static_cast<int64_t>( hidden_states.getMetadata().shape[ 1 ] );
            const int64_t experts = static_cast<int64_t>( per_expert_scale.getMetadata().shape[ 0 ] );
            const int64_t top_k = static_cast<int64_t>( reference_indices.getMetadata().shape[ 1 ] );

            Router<DeviceType::Cuda, TPrecision> router( "router", RouterConfig( hidden, experts, top_k ), Device::Cuda( 0 ) );
            router.build( BuildContext( shape_t{ tokens, hidden }, RuntimeMode::Inference, false ) );

            loadValues( *router.findComponent( "router.proj" ), "weight", precision,
                as_floats( projection ), static_cast<std::size_t>( experts * hidden ), shape_t{ experts, hidden } );
            loadValues( router, "scale", precision, as_floats( scale ), static_cast<std::size_t>( hidden ), shape_t{ hidden } );
            loadValues( router, "per_expert_scale", precision, as_floats( per_expert_scale ),
                static_cast<std::size_t>( experts ), shape_t{ experts } );

            const RoutingResult mila = route( router, as_floats( hidden_states ), tokens, hidden );

            const ReferenceComparison comparison = compareToReference( mila,
                as_floats( reference_logits ), as_floats( reference_weights ),
                static_cast<const std::int32_t*>( static_cast<const void*>( reference_indices.data() ) ),
                tokens, experts, top_k, logits_tie, weights_agree );

            std::cout << std::format( "[ reference ] {} layer 5, {} tokens: {} rows with a different set, "
                "{} unexplained, worst weight error {:.3e}\n",
                precision == TensorDataType::BF16 ? "BF16" : "FP32", tokens, comparison.rows_with_different_sets,
                comparison.unexplained_set_differences, comparison.worst_weight_error );

            return comparison;
        }

        // Experts 1 and 2 share a projection row, so their logits are exactly equal; the lower index
        // must win. Every value here is exactly representable in bfloat16.
        template<TensorDataType TPrecision>
        void expectExactTieSelectsLowerIndex( double weight_tolerance_relative, double weight_tolerance_absolute )
        {
            const TensorDataType precision = TPrecision;

            Router<DeviceType::Cuda, TPrecision> router( "router", RouterConfig( kHidden, kExperts, kTopK ), Device::Cuda( 0 ) );
            router.build( BuildContext( shape_t{ 1, kHidden }, RuntimeMode::Inference, false ) );

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

            loadValues( *router.findComponent( "router.proj" ), "weight", precision,
                projection.data(), projection.size(), shape_t{ kExperts, kHidden } );
            loadValues( router, "scale", precision, scale.data(), scale.size(), shape_t{ kHidden } );
            loadValues( router, "per_expert_scale", precision, per_expert_scale.data(), per_expert_scale.size(), shape_t{ kExperts } );

            std::vector<float> input( kHidden );

            for ( int64_t i = 0; i < kHidden; ++i )
            {
                input[ i ] = 0.25f + 0.125f * static_cast<float>( i );
            }

            const RoutingResult routing = route( router, input.data(), 1, kHidden );

            double sum_of_squares = 0.0;

            for ( float value : input )
            {
                sum_of_squares += static_cast<double>( value ) * value;
            }

            const double rstd = 1.0 / std::sqrt( sum_of_squares / kHidden + 1e-6 );
            std::vector<double> logits( kExperts, 0.0 );

            for ( int64_t e = 0; e < kExperts; ++e )
            {
                for ( int64_t i = 0; i < kHidden; ++i )
                {
                    logits[ e ] += projection[ e * kHidden + i ] * input[ i ] * rstd / std::sqrt( static_cast<double>( kHidden ) );
                }
            }

            const double selected_mass = std::exp( logits[ 0 ] ) + std::exp( logits[ 1 ] );
            const double expected[ 2 ] = {
                std::exp( logits[ 0 ] ) / selected_mass * per_expert_scale[ 0 ],
                std::exp( logits[ 1 ] ) / selected_mass * per_expert_scale[ 1 ] };

            EXPECT_EQ( ( std::set<std::int32_t>( routing.indices.begin(), routing.indices.end() ) ),
                ( std::set<std::int32_t>{ 0, 1 } ) );

            for ( int64_t slot = 0; slot < kTopK; ++slot )
            {
                const std::int32_t expert = routing.indices[ slot ];
                ASSERT_TRUE( expert == 0 || expert == 1 );

                const double want = expected[ expert ];
                const double tolerance = weight_tolerance_absolute + weight_tolerance_relative * std::fabs( want );

                EXPECT_NEAR( routing.weights[ slot ], want, tolerance ) << "expert " << expert;
            }
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( RouterCudaTests, Fp32_ExactTieSelectsLowerIndexAndWeightsMatchDefinition )
    {
        expectExactTieSelectsLowerIndex<TensorDataType::FP32>( 0.0, 1e-6 );
    }

    TEST_F( RouterCudaTests, Bf16_ExactTieSelectsLowerIndexAndWeightsMatchDefinition )
    {
        expectExactTieSelectsLowerIndex<TensorDataType::BF16>( kBf16Relative, 0.0 );
    }

    // Gemma4MoE.md Phase 5, "FP32, end to end".
    TEST_F( RouterCudaTests, Fp32_MatchesHuggingFaceReferenceLayer5 )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "router reference not present at: " << referencePath().string();
        }

        const ReferenceComparison comparison = routeReference<TensorDataType::FP32>( TensorDataType::FP32,
            []( double a, double b ) { return std::fabs( a - b ) <= kFp32LogitTieWidth; },
            []( double actual, double reference ) { return std::fabs( actual - reference ) <= kFp32WeightTolerance; } );

        EXPECT_EQ( comparison.unexplained_set_differences, 0 ) << comparison.rows_with_different_sets << " rows chose a different set";
        EXPECT_EQ( comparison.weight_mismatches, 0 ) << "worst weight error " << comparison.worst_weight_error;
    }

    // Gemma4MoE.md Phase 5, "BF16, end to end": against the FP32 reference.
    TEST_F( RouterCudaTests, Bf16_MatchesHuggingFaceReferenceLayer5WithinBf16Resolution )
    {
        if ( !fs::exists( referencePath() ) )
        {
            GTEST_SKIP() << "router reference not present at: " << referencePath().string();
        }

        const ReferenceComparison comparison = routeReference<TensorDataType::BF16>( TensorDataType::BF16,
            []( double a, double b ) { return std::fabs( a - b ) <= kBf16Relative * std::max( std::fabs( a ), std::fabs( b ) ); },
            []( double actual, double reference ) { return std::fabs( actual - reference ) <= kBf16Relative * std::fabs( reference ); } );

        EXPECT_EQ( comparison.unexplained_set_differences, 0 ) << comparison.rows_with_different_sets << " rows chose a different set";
        EXPECT_EQ( comparison.weight_mismatches, 0 ) << "worst weight error " << comparison.worst_weight_error;
    }

    // ====================================================================
    // G. Footprint
    // ====================================================================

    TEST_F( RouterCudaTests, Bf16_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context = BuildContext( shape_t{ 3, kHidden }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

        Router<DeviceType::Cuda, TensorDataType::BF16> predictor( "router", RouterConfig( kHidden, kExperts, kTopK ), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        Router<DeviceType::Cuda, TensorDataType::BF16> built( "router", RouterConfig( kHidden, kExperts, kTopK ), Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }
}
