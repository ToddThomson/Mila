/**
 * @file MixtureOfExperts.Cuda.cpp
 * @brief Concrete-component tests for MixtureOfExperts<DeviceType::Cuda, {FP32, BF16}, Gelu>.
 *
 * The CUDA half of the Phase 6 gate and the Phase 7 decode gate (Specifications/Gemma4MoE.md), through
 * the component, with the tolerances the record fixed before the first run. The HuggingFace gates skip
 * without the capture from Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_experts_reference.py.
 *
 * Pin a card by UUID (CUDA_VISIBLE_DEVICES) and name it with any number reported from here.
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

        // Deterministic values in [-scale, scale).
        std::vector<float> synthetic( std::size_t count, std::uint32_t seed, float scale )
        {
            std::vector<float> values( count );
            std::uint32_t state = seed;

            for ( float& value : values )
            {
                state = state * 1664525u + 1013904223u;
                value = scale * ( static_cast<float>( state >> 8 ) / 8388608.0f - 1.0f );
            }

            return values;
        }
    }

    namespace
    {
        using Fp4Group64 = Mila::Dnn::Quant::Weight::PerGroupFp4<64>;
        using Fp8PerChannel = Mila::Dnn::Quant::Weight::PerChannelFp8<>;

        template<TensorDataType TPrecision, typename TWeightQuantization = Mila::Dnn::Quant::Weight::NoWeightQuant>
        using CudaExperts = Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu, TWeightQuantization>;

        constexpr int64_t kFp4Group = 64;
        constexpr int64_t kFp4Hidden = 128;
        constexpr int64_t kFp4Intermediate = 64;
        constexpr int64_t kFp4Experts = 16;
        constexpr int64_t kFp4TopK = 4;
        constexpr int64_t kFp4Tokens = 48;

        // Weights the per-group FP4 quantizer reproduces exactly: E2M1 grid values times a power-of-two
        // group scale, with each group's largest magnitude 6 * 2^k so absmax recovers that scale.
        std::vector<float> fp4ExactWeights( int64_t rows, int64_t columns, int64_t group, std::uint32_t seed )
        {
            constexpr float kMagnitudes[] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

            std::vector<float> values( static_cast<std::size_t>( rows * columns ) );
            std::uint32_t state = seed;

            auto next = [&state]()
            {
                state = state * 1664525u + 1013904223u;

                return state >> 8;
            };

            for ( int64_t row = 0; row < rows; ++row )
            {
                for ( int64_t start = 0; start < columns; start += group )
                {
                    const float scale = std::ldexp( 1.0f, -static_cast<int>( 2 + next() % 5 ) );

                    for ( int64_t column = start; column < start + group; ++column )
                    {
                        const float sign = ( next() & 1u ) ? -1.0f : 1.0f;

                        values[ static_cast<std::size_t>( row * columns + column ) ] = sign * kMagnitudes[ next() % 8 ] * scale;
                    }

                    const int64_t largest = start + static_cast<int64_t>( next() % group );

                    values[ static_cast<std::size_t>( row * columns + largest ) ] = ( ( next() & 1u ) ? -6.0f : 6.0f ) * scale;
                }
            }

            return values;
        }

        int64_t countBitMismatches( const std::vector<float>& actual, const std::vector<float>& expected )
        {
            if ( actual.size() != expected.size() )
            {
                return static_cast<int64_t>( std::max( actual.size(), expected.size() ) );
            }

            int64_t mismatches = 0;

            for ( std::size_t i = 0; i < actual.size(); ++i )
            {
                if ( std::bit_cast<std::uint32_t>( actual[ i ] ) != std::bit_cast<std::uint32_t>( expected[ i ] ) )
                {
                    ++mismatches;
                }
            }

            return mismatches;
        }

        void constructFp8Bank()
        {
            CudaExperts<TensorDataType::BF16, Fp8PerChannel> experts(
                "experts", MixtureOfExpertsConfig( kFp4Hidden, kFp4Intermediate, kFp4Experts, kFp4TopK ), Device::Cuda( 0 ) );
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

        template<typename TExperts>
        std::vector<float> run( TExperts& experts,
            const float* input, const float* weights, const std::int32_t* indices,
            const shape_t& input_shape, const shape_t& routing_shape )
        {
            using DeviceTensor = typename TExperts::TensorType;

            HostFp32 host_input( Device::Cpu(), input_shape );
            HostFp32 host_weights( Device::Cpu(), routing_shape );
            HostInt32 host_indices( Device::Cpu(), routing_shape );

            std::memcpy( host_input.data(), input, static_cast<std::size_t>( elementCount( input_shape ) ) * sizeof( float ) );
            std::memcpy( host_weights.data(), weights, static_cast<std::size_t>( elementCount( routing_shape ) ) * sizeof( float ) );
            std::memcpy( host_indices.data(), indices, static_cast<std::size_t>( elementCount( routing_shape ) ) * sizeof( std::int32_t ) );

            DeviceTensor device_input( Device::Cuda( 0 ), input_shape );
            DeviceTensor device_weights( Device::Cuda( 0 ), routing_shape );
            DeviceInt32 device_indices( Device::Cuda( 0 ), routing_shape );

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
                static_cast<const std::int32_t*>( static_cast<const void*>( indices.data() ) ),
                shape_t{ tokens, hidden }, shape_t{ tokens, top_k } );

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

            const std::vector<float> output = run( experts, input.data(), combine, selections,
                shape_t{ tokens, kHidden }, shape_t{ tokens, kTopK } );

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

        // Phase 7: each token decoded alone as [1, 1, H] must reproduce its prefill row bit for bit.
        template<TensorDataType TPrecision>
        void expectSingleTokenMatchesPrefillRow( bool build_for_single_token )
        {
            using Experts = Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu>;

            const TensorDataType precision = TPrecision;
            constexpr int64_t tokens = 48;
            constexpr int64_t hidden = 64;
            constexpr int64_t intermediate = 32;
            constexpr int64_t experts_count = 16;
            constexpr int64_t top_k = 4;

            const std::vector<float> gate_up = synthetic( experts_count * 2 * intermediate * hidden, 1, 0.25f );
            const std::vector<float> down = synthetic( experts_count * hidden * intermediate, 2, 0.25f );
            const std::vector<float> input = synthetic( tokens * hidden, 3, 1.0f );
            const std::vector<float> combine = synthetic( tokens * top_k, 4, 0.5f );
            std::vector<std::int32_t> selections( tokens * top_k );

            for ( int64_t token = 0; token < tokens; ++token )
            {
                for ( int64_t slot = 0; slot < top_k; ++slot )
                {
                    selections[ token * top_k + slot ] = static_cast<std::int32_t>( ( token * 7 + slot * 5 ) % experts_count );
                }
            }

            auto makeExperts = [&]( const shape_t& build_shape )
            {
                auto experts = std::make_unique<Experts>(
                    "experts", MixtureOfExpertsConfig( hidden, intermediate, experts_count, top_k ), Device::Cuda( 0 ) );
                experts->build( BuildContext( build_shape, RuntimeMode::Inference, false ) );

                loadValues( *experts, "gate_up_proj", precision, gate_up.data(), gate_up.size(),
                    shape_t{ experts_count, 2 * intermediate, hidden } );
                loadValues( *experts, "down_proj", precision, down.data(), down.size(),
                    shape_t{ experts_count, hidden, intermediate } );

                return experts;
            };

            auto prefill = makeExperts( shape_t{ tokens, hidden } );
            const std::vector<float> rows = run( *prefill, input.data(), combine.data(), selections.data(),
                shape_t{ tokens, hidden }, shape_t{ tokens, top_k } );

            std::unique_ptr<Experts> single;
            Experts* decoder = prefill.get();

            if ( build_for_single_token )
            {
                single = makeExperts( shape_t{ 1, 1, hidden } );
                decoder = single.get();
            }

            int64_t mismatches = 0;
            double worst = 0.0;

            for ( int64_t token = 0; token < tokens; ++token )
            {
                const std::vector<float> row = run( *decoder, input.data() + token * hidden, combine.data() + token * top_k,
                    selections.data() + token * top_k, shape_t{ 1, 1, hidden }, shape_t{ 1, 1, top_k } );

                ASSERT_EQ( row.size(), static_cast<std::size_t>( hidden ) );

                for ( int64_t column = 0; column < hidden; ++column )
                {
                    const float expected = rows[ token * hidden + column ];

                    if ( std::bit_cast<std::uint32_t>( row[ column ] ) != std::bit_cast<std::uint32_t>( expected ) )
                    {
                        ++mismatches;
                        worst = std::max( worst, std::fabs( static_cast<double>( row[ column ] ) - expected ) );
                    }
                }
            }

            EXPECT_EQ( mismatches, 0 ) << "of " << tokens * hidden << " elements differ from the prefill row; worst " << worst;

            std::cout << std::format( "[ decode ] {} {}: {} of {} elements differ from prefill\n",
                precision == TensorDataType::BF16 ? "BF16" : "FP32",
                build_for_single_token ? "built for one token" : "built for prefill", mismatches, tokens * hidden );
        }

        struct Fp4Case
        {
            std::vector<float> gate_up;
            std::vector<float> down;
            std::vector<float> input;
            std::vector<float> combine;
            std::vector<std::int32_t> selections;
        };

        static Fp4Case fp4Case()
        {
            Fp4Case result;
            result.gate_up = fp4ExactWeights( kFp4Experts * 2 * kFp4Intermediate, kFp4Hidden, kFp4Group, 11 );
            result.down = fp4ExactWeights( kFp4Experts * kFp4Hidden, kFp4Intermediate, kFp4Group, 12 );
            result.input = synthetic( kFp4Tokens * kFp4Hidden, 13, 1.0f );
            result.combine = synthetic( kFp4Tokens * kFp4TopK, 14, 0.5f );
            result.selections.resize( static_cast<std::size_t>( kFp4Tokens * kFp4TopK ) );

            for ( int64_t token = 0; token < kFp4Tokens; ++token )
            {
                for ( int64_t slot = 0; slot < kFp4TopK; ++slot )
                {
                    result.selections[ static_cast<std::size_t>( token * kFp4TopK + slot ) ] =
                        static_cast<std::int32_t>( ( token * 7 + slot * 5 ) % kFp4Experts );
                }
            }

            return result;
        }

        // Both banks load the same BF16 bits; the FP4 one quantizes them on load.
        template<typename TExperts>
        static std::unique_ptr<TExperts> makeFp4CaseBank( const Fp4Case& weights_case )
        {
            auto experts = std::make_unique<TExperts>(
                "experts", MixtureOfExpertsConfig( kFp4Hidden, kFp4Intermediate, kFp4Experts, kFp4TopK ), Device::Cuda( 0 ) );
            experts->build( BuildContext( shape_t{ kFp4Tokens, kFp4Hidden }, RuntimeMode::Inference, false ) );

            loadValues( *experts, "gate_up_proj", TensorDataType::BF16, weights_case.gate_up.data(), weights_case.gate_up.size(),
                shape_t{ kFp4Experts, 2 * kFp4Intermediate, kFp4Hidden } );
            loadValues( *experts, "down_proj", TensorDataType::BF16, weights_case.down.data(), weights_case.down.size(),
                shape_t{ kFp4Experts, kFp4Hidden, kFp4Intermediate } );

            return experts;
        }

        template<typename TExperts>
        std::vector<float> runFp4Case( TExperts& experts, const Fp4Case& weights_case )
        {
            return run( experts, weights_case.input.data(), weights_case.combine.data(), weights_case.selections.data(),
                shape_t{ kFp4Tokens, kFp4Hidden }, shape_t{ kFp4Tokens, kFp4TopK } );
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

    TEST_F( MixtureOfExpertsCudaTests, Fp32_DecodeMatchesPrefillRow_BuiltForPrefill )
    {
        expectSingleTokenMatchesPrefillRow<TensorDataType::FP32>( false );
    }

    TEST_F( MixtureOfExpertsCudaTests, Bf16_DecodeMatchesPrefillRow_BuiltForPrefill )
    {
        expectSingleTokenMatchesPrefillRow<TensorDataType::BF16>( false );
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp32_DecodeMatchesPrefillRow_BuiltForOneToken )
    {
        expectSingleTokenMatchesPrefillRow<TensorDataType::FP32>( true );
    }

    TEST_F( MixtureOfExpertsCudaTests, Bf16_DecodeMatchesPrefillRow_BuiltForOneToken )
    {
        expectSingleTokenMatchesPrefillRow<TensorDataType::BF16>( true );
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

        const std::vector<float> output = run( experts, input.data(), combine, selections,
            shape_t{ tokens, kHidden }, shape_t{ tokens, kTopK } );

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
        EXPECT_EQ( predicted.device_inactive_parameter_bytes, actual.device_inactive_parameter_bytes ) << "inactive parameters";

        // 3 experts x 3 x intermediate 2 x hidden 4 at BF16 is 144 bytes; top-2 leaves one expert, 48, unread.
        EXPECT_EQ( actual.device_parameter_bytes, std::size_t{ 144 } );
        EXPECT_EQ( actual.device_inactive_parameter_bytes, std::size_t{ 48 } );
        EXPECT_EQ( actual.activeDeviceParameterBytes(), std::size_t{ 96 } );

        // The gated scratch is part of the state: 3 tokens x top-2 x intermediate 2, FP32.
        EXPECT_GE( actual.device_state_bytes, static_cast<std::size_t>( 3 * kTopK * kIntermediate ) * sizeof( float ) );
    }

    // ====================================================================
    // FP4 expert bank (Gemma4MoE.md Phase 8, "FP4 expert bank -- gate")
    // ====================================================================

    TEST_F( MixtureOfExpertsCudaTests, Fp4_ExactWeightsBitIdenticalToBf16Bank )
    {
        const Fp4Case weights_case = fp4Case();

        std::vector<float> expected;
        {
            auto reference = makeFp4CaseBank<CudaExperts<TensorDataType::BF16>>( weights_case );
            expected = runFp4Case( *reference, weights_case );
        }

        auto quantized = makeFp4CaseBank<CudaExperts<TensorDataType::BF16, Fp4Group64>>( weights_case );
        const std::vector<float> prefill = runFp4Case( *quantized, weights_case );

        std::vector<float> decoded;

        for ( int64_t token = 0; token < kFp4Tokens; ++token )
        {
            const std::vector<float> row = run( *quantized,
                weights_case.input.data() + token * kFp4Hidden,
                weights_case.combine.data() + token * kFp4TopK,
                weights_case.selections.data() + token * kFp4TopK,
                shape_t{ 1, 1, kFp4Hidden }, shape_t{ 1, 1, kFp4TopK } );

            decoded.insert( decoded.end(), row.begin(), row.end() );
        }

        const auto nonzero = std::count_if( expected.begin(), expected.end(), []( float value ) { return value != 0.0f; } );

        EXPECT_GT( nonzero, 0 ) << "the BF16 bank produced all zeros; the comparison proves nothing";
        EXPECT_EQ( countBitMismatches( prefill, expected ), 0 ) << "prefill, of " << expected.size();
        EXPECT_EQ( countBitMismatches( decoded, expected ), 0 ) << "one token at a time, of " << expected.size();

        std::cout << std::format( "[ fp4 ] {} of {} prefill and {} one-token elements differ from the BF16 bank\n",
            countBitMismatches( prefill, expected ), expected.size(), countBitMismatches( decoded, expected ) );
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp4_SavedBankReloadsBitIdentical )
    {
        const Fp4Case weights_case = fp4Case();
        const fs::path saved = fs::temp_directory_path() / "mila_moe_fp4_round_trip.safetensors";

        std::vector<float> original;
        {
            auto bank = makeFp4CaseBank<CudaExperts<TensorDataType::BF16, Fp4Group64>>( weights_case );
            original = runFp4Case( *bank, weights_case );

            Serialization::SafeTensorsWriter writer( saved );
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Declare );
            writer.beginData();
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Write );
            writer.close();
        }

        CudaExperts<TensorDataType::BF16, Fp4Group64> reloaded(
            "experts", MixtureOfExpertsConfig( kFp4Hidden, kFp4Intermediate, kFp4Experts, kFp4TopK ), Device::Cuda( 0 ) );
        reloaded.build( BuildContext( shape_t{ kFp4Tokens, kFp4Hidden }, RuntimeMode::Inference, false ) );

        std::vector<std::string> names;
        {
            Serialization::PretrainedModelReader reader( saved );
            names = reader.getTensorNames();

            for ( const auto& name : names )
            {
                auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

                reloaded.loadParameter( name.substr( name.rfind( '.' ) + 1 ), blob );
                reloaded.synchronize();
            }
        }

        std::error_code ignored;
        fs::remove( saved, ignored );

        std::sort( names.begin(), names.end() );

        EXPECT_EQ( names, ( std::vector<std::string>{
            "experts.down_proj", "experts.down_proj_scale", "experts.gate_up_proj", "experts.gate_up_proj_scale" } ) );
        EXPECT_EQ( countBitMismatches( runFp4Case( reloaded, weights_case ), original ), 0 );
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp4_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ 3, kFp4Hidden }, RuntimeMode::Inference, false );
        const MixtureOfExpertsConfig config( kFp4Hidden, kFp4Intermediate, kFp4Experts, kFp4TopK );

        CudaExperts<TensorDataType::BF16, Fp4Group64> predictor( "experts", config, Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        CudaExperts<TensorDataType::BF16, Fp4Group64> built( "experts", config, Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_inactive_parameter_bytes, actual.device_inactive_parameter_bytes ) << "inactive parameters";

        // Packed gate_up 16x128x64 and its scales 16x128x2 x4; packed down 16x128x32 and its scales 16x128x1 x4.
        EXPECT_EQ( actual.device_parameter_bytes, std::size_t{ 131072 + 16384 + 65536 + 8192 } );
        // Four of sixteen experts per token: twelve sixteenths of every tensor unread.
        EXPECT_EQ( actual.device_inactive_parameter_bytes, std::size_t{ 221184 } / 16 * 12 );
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp4_ExpertWidthNotAMultipleOfTheGroupIsRefused )
    {
        CudaExperts<TensorDataType::BF16, Fp4Group64> experts(
            "experts", MixtureOfExpertsConfig( kFp4Hidden, 96, kFp4Experts, kFp4TopK ), Device::Cuda( 0 ) );

        EXPECT_THROW( experts.build( BuildContext( shape_t{ 4, kFp4Hidden }, RuntimeMode::Inference, false ) ), std::invalid_argument );
    }

    // The op refuses at construction; Component::setExecutionContext rethrows that as runtime_error.
    TEST_F( MixtureOfExpertsCudaTests, Fp8Bank_IsRefused )
    {
        try
        {
            constructFp8Bank();
            FAIL() << "an FP8 expert bank was constructed";
        }
        catch ( const std::runtime_error& error )
        {
            EXPECT_NE( std::string( error.what() ).find( "per-group FP4" ), std::string::npos ) << error.what();
        }
    }
}
