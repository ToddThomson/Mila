/**
 * @file MixtureOfExperts.Cuda.cpp
 * @brief Concrete-component tests for MixtureOfExperts<DeviceType::Cuda, {FP32, BF16}, Gelu>.
 *
 * The CUDA half of the Phase 6 gate and the Phase 7 decode gate (Specifications/Notebooks/Gemma4MoE.md), through
 * the component, with the tolerances the record fixed before the first run. The HuggingFace gates skip
 * without the capture from Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_experts_reference.py.
 *
 * Pin a card by UUID (CUDA_VISIBLE_DEVICES) and name it with any number reported from here.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

import Mila;
import Dnn.Quantization.Weight.Int4Packing;

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
        using Q4_0 = Mila::Dnn::Quant::Weight::PerGroupInt4<32>;
        using Fp8PerChannel = Mila::Dnn::Quant::Weight::PerChannelFp8<>;

        template<TensorDataType TPrecision, typename TWeightQuantization = Mila::Dnn::Quant::Weight::NoWeightQuant>
        using CudaExperts = Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TPrecision, ActivationType::Gelu, TWeightQuantization>;

        constexpr int64_t kFp4Group = 64;
        constexpr int64_t kPackedHidden = 128;
        constexpr int64_t kPackedIntermediate = 64;
        constexpr int64_t kPackedExperts = 16;
        constexpr int64_t kPackedTopK = 4;
        constexpr int64_t kPackedTokens = 48;

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

        // Weights the Q4_0 quantizer reproduces exactly (Gemma4MoE.md Phase 9, gate 2): ( code - 8 ) * d with d a
        // power of two, and one element of each group at -8 d, so the group's extreme is -8 d and its scale d.
        // Every value has at most four significant bits, so it is exact in BF16 too.
        std::vector<float> q4ExactWeights( int64_t rows, int64_t columns, std::uint32_t seed )
        {
            constexpr int64_t kGroup = 32;

            std::vector<float> values( static_cast<std::size_t>( rows * columns ) );
            std::uint32_t state = seed;

            auto next = [&state]()
            {
                state = state * 1664525u + 1013904223u;

                return state >> 8;
            };

            for ( int64_t row = 0; row < rows; ++row )
            {
                for ( int64_t start = 0; start < columns; start += kGroup )
                {
                    const float scale = std::ldexp( 1.0f, -static_cast<int>( 2 + next() % 5 ) );

                    // Codes 1 to 15 here, so the one -8 d below is the group's only largest magnitude.
                    for ( int64_t column = start; column < start + kGroup; ++column )
                    {
                        const int code = 1 + static_cast<int>( next() % 15 );

                        values[ static_cast<std::size_t>( row * columns + column ) ] = static_cast<float>( code - 8 ) * scale;
                    }

                    const int64_t extreme = start + static_cast<int64_t>( next() % kGroup );

                    values[ static_cast<std::size_t>( row * columns + extreme ) ] = -8.0f * scale;
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
                "experts", MixtureOfExpertsConfig( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );
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

            Serialization::WeightsReader reader( referencePath() );
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

        struct PackedCase
        {
            std::vector<float> gate_up;
            std::vector<float> down;
            std::vector<float> input;
            std::vector<float> combine;
            std::vector<std::int32_t> selections;
        };

        static PackedCase fp4Case()
        {
            PackedCase result;
            result.gate_up = fp4ExactWeights( kPackedExperts * 2 * kPackedIntermediate, kPackedHidden, kFp4Group, 11 );
            result.down = fp4ExactWeights( kPackedExperts * kPackedHidden, kPackedIntermediate, kFp4Group, 12 );
            result.input = synthetic( kPackedTokens * kPackedHidden, 13, 1.0f );
            result.combine = synthetic( kPackedTokens * kPackedTopK, 14, 0.5f );
            result.selections.resize( static_cast<std::size_t>( kPackedTokens * kPackedTopK ) );

            for ( int64_t token = 0; token < kPackedTokens; ++token )
            {
                for ( int64_t slot = 0; slot < kPackedTopK; ++slot )
                {
                    result.selections[ static_cast<std::size_t>( token * kPackedTopK + slot ) ] =
                        static_cast<std::int32_t>( ( token * 7 + slot * 5 ) % kPackedExperts );
                }
            }

            return result;
        }

        static PackedCase q4Case()
        {
            PackedCase result = fp4Case();
            result.gate_up = q4ExactWeights( kPackedExperts * 2 * kPackedIntermediate, kPackedHidden, 21 );
            result.down = q4ExactWeights( kPackedExperts * kPackedHidden, kPackedIntermediate, 22 );

            return result;
        }

        // Both banks load the same BF16 bits; the packed one quantizes them on load.
        template<typename TExperts>
        static std::unique_ptr<TExperts> makePackedCaseBank( const PackedCase& weights_case, int64_t build_tokens = kPackedTokens )
        {
            auto experts = std::make_unique<TExperts>(
                "experts", MixtureOfExpertsConfig( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );
            experts->build( BuildContext( shape_t{ build_tokens, kPackedHidden }, RuntimeMode::Inference, false ) );

            loadValues( *experts, "gate_up_proj", TensorDataType::BF16, weights_case.gate_up.data(), weights_case.gate_up.size(),
                shape_t{ kPackedExperts, 2 * kPackedIntermediate, kPackedHidden } );
            loadValues( *experts, "down_proj", TensorDataType::BF16, weights_case.down.data(), weights_case.down.size(),
                shape_t{ kPackedExperts, kPackedHidden, kPackedIntermediate } );

            return experts;
        }

        template<typename TExperts>
        std::vector<float> runPackedCase( TExperts& experts, const PackedCase& weights_case )
        {
            return run( experts, weights_case.input.data(), weights_case.combine.data(), weights_case.selections.data(),
                shape_t{ kPackedTokens, kPackedHidden }, shape_t{ kPackedTokens, kPackedTopK } );
        }

        template<typename TExperts>
        std::vector<float> runPackedCaseOneTokenAtATime( TExperts& experts, const PackedCase& weights_case )
        {
            std::vector<float> decoded;

            for ( int64_t token = 0; token < kPackedTokens; ++token )
            {
                const std::vector<float> row = run( experts,
                    weights_case.input.data() + token * kPackedHidden,
                    weights_case.combine.data() + token * kPackedTopK,
                    weights_case.selections.data() + token * kPackedTopK,
                    shape_t{ 1, 1, kPackedHidden }, shape_t{ 1, 1, kPackedTopK } );

                decoded.insert( decoded.end(), row.begin(), row.end() );
            }

            return decoded;
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
    // G. Footprint -- includes the bank's gated scratch
    // ====================================================================

    TEST_F( MixtureOfExpertsCudaTests, Bf16_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context = BuildContext( shape_t{ 3, kHidden }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

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

    // Installed slots belong to the installer, in the prediction and the build alike, and must cover the build.
    TEST_F( MixtureOfExpertsCudaTests, Bf16_InstalledSlots_UncountedAndCheckedAgainstTheBuild )
    {
        using Experts = Mila::Dnn::MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu>;
        using OutputTensor = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using GatedTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        const shape_t input_shape{ 3, kHidden };
        const dim_t gated_elements = 3 * kTopK * kIntermediate;
        const BuildContext context = BuildContext( input_shape, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );
        const MixtureOfExpertsConfig config( kHidden, kIntermediate, kExperts, kTopK );

        {
            Experts experts( "experts", config, Device::Cuda( 0 ) );
            experts.installSharedOutputs(
                std::make_shared<OutputTensor>( Device::Cuda( 0 ), input_shape ),
                std::make_shared<GatedTensor>( Device::Cuda( 0 ), shape_t{ gated_elements } ) );

            EXPECT_EQ( experts.getRequiredMemory( context ).device_state_bytes, std::size_t{ 0 } ) << "predicted";

            experts.build( context );

            EXPECT_EQ( experts.getMemoryStats().device_state_bytes, std::size_t{ 0 } ) << "built";
        }

        {
            Experts experts( "experts", config, Device::Cuda( 0 ) );
            experts.installSharedOutputs(
                std::make_shared<OutputTensor>( Device::Cuda( 0 ), input_shape ),
                std::make_shared<GatedTensor>( Device::Cuda( 0 ), shape_t{ gated_elements - 1 } ) );

            EXPECT_THROW( experts.build( context ), std::invalid_argument ) << "a gated slot one element short";
        }
    }

    // ====================================================================
    // FP4 expert bank (Gemma4MoE.md Phase 8, "FP4 expert bank -- gate")
    // ====================================================================

    TEST_F( MixtureOfExpertsCudaTests, Fp4_ExactWeightsBitIdenticalToBf16Bank )
    {
        const PackedCase weights_case = fp4Case();

        std::vector<float> expected;
        {
            auto reference = makePackedCaseBank<CudaExperts<TensorDataType::BF16>>( weights_case );
            expected = runPackedCase( *reference, weights_case );
        }

        auto quantized = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Fp4Group64>>( weights_case );
        const std::vector<float> prefill = runPackedCase( *quantized, weights_case );

        std::vector<float> decoded;

        for ( int64_t token = 0; token < kPackedTokens; ++token )
        {
            const std::vector<float> row = run( *quantized,
                weights_case.input.data() + token * kPackedHidden,
                weights_case.combine.data() + token * kPackedTopK,
                weights_case.selections.data() + token * kPackedTopK,
                shape_t{ 1, 1, kPackedHidden }, shape_t{ 1, 1, kPackedTopK } );

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
        const PackedCase weights_case = fp4Case();
        const fs::path saved = fs::temp_directory_path() / "mila_moe_fp4_round_trip.safetensors";

        std::vector<float> original;
        {
            auto bank = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Fp4Group64>>( weights_case );
            original = runPackedCase( *bank, weights_case );

            Serialization::SafeTensorsWriter writer( saved );
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Declare );
            writer.beginData();
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Write );
            writer.close();
        }

        CudaExperts<TensorDataType::BF16, Fp4Group64> reloaded(
            "experts", MixtureOfExpertsConfig( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );
        reloaded.build( BuildContext( shape_t{ kPackedTokens, kPackedHidden }, RuntimeMode::Inference, false ) );

        std::vector<std::string> names;
        {
            Serialization::WeightsReader reader( saved );
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
        EXPECT_EQ( countBitMismatches( runPackedCase( reloaded, weights_case ), original ), 0 );
    }

    TEST_F( MixtureOfExpertsCudaTests, Fp4_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context = BuildContext( shape_t{ 3, kPackedHidden }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );
        const MixtureOfExpertsConfig config( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK );

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
            "experts", MixtureOfExpertsConfig( kPackedHidden, 96, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );

        EXPECT_THROW( experts.build( BuildContext( shape_t{ 4, kPackedHidden }, RuntimeMode::Inference, false ) ), std::invalid_argument );
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

    // ====================================================================
    // Q4_0 expert bank (Gemma4MoE.md Phase 9, "Gate, written before any run", G5 items 1-5)
    // ====================================================================

    static_assert( expertBankImplements<Q4_0> );
    static_assert( !expertBankImplements<Mila::Dnn::Quant::Weight::PerGroupInt4<64>> );
    static_assert( !expertBankImplements<Fp8PerChannel> );

    // Item 1: a stack quantized on load is E x rows output channels of Linear's quantizer, at the 26B's shapes.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_QuantizerEqualsTheCodecAtTheRealShapes )
    {
        constexpr int64_t kHidden = 2816;
        constexpr int64_t kIntermediate = 704;
        constexpr int64_t kExperts = 128;
        constexpr int64_t kGroup = 32;

        CudaExperts<TensorDataType::BF16, Q4_0> bank(
            "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, 8 ), Device::Cuda( 0 ) );
        bank.build( BuildContext( shape_t{ 1, kHidden }, RuntimeMode::Inference, false ) );

        // gate_up_proj, its scales, down_proj, its scales.
        const std::vector<ITensor*> stored = bank.getParameters();
        ASSERT_EQ( stored.size(), 4u );

        struct Projection
        {
            const char* name;
            int64_t rows_per_expert;
            int64_t columns;
            const ITensor* packed;
            const ITensor* scales;
        };

        const Projection projections[] = {
            { "gate_up_proj", 2 * kIntermediate, kHidden, stored[ 0 ], stored[ 1 ] },
            { "down_proj", kHidden, kIntermediate, stored[ 2 ], stored[ 3 ] },
        };

        std::uint32_t seed = 20260930u;

        for ( const Projection& projection : projections )
        {
            const int64_t rows = kExperts * projection.rows_per_expert;
            const int64_t columns = projection.columns;

            std::mt19937 generator( seed++ );
            std::normal_distribution<float> distribution( 0.0f, 0.02f );
            std::vector<std::uint16_t> weights( static_cast<std::size_t>( rows * columns ) );

            for ( auto& weight : weights )
            {
                weight = static_cast<std::uint16_t>( std::bit_cast<std::uint32_t>( distribution( generator ) ) >> 16 );
            }

            const std::size_t bytes = weights.size() * sizeof( std::uint16_t );
            Serialization::TensorMetadata meta{ TensorDataType::BF16, shape_t{ kExperts, projection.rows_per_expert, columns }, bytes };
            Serialization::TensorBlobView blob( meta, weights.data(), bytes );

            bank.loadParameter( projection.name, blob );
            bank.synchronize();

            std::vector<std::uint8_t> expected_codes( static_cast<std::size_t>( rows * columns / 2 ) );
            std::vector<std::uint16_t> expected_scales( static_cast<std::size_t>( rows * columns / kGroup ) );
            std::vector<float> row_values( static_cast<std::size_t>( columns ) );

            for ( int64_t row = 0; row < rows; ++row )
            {
                for ( int64_t column = 0; column < columns; ++column )
                {
                    row_values[ static_cast<std::size_t>( column ) ] = std::bit_cast<float>(
                        static_cast<std::uint32_t>( weights[ static_cast<std::size_t>( row * columns + column ) ] ) << 16 );
                }

                Mila::Dnn::Quant::Weight::quantizeInt4( row_values.data(), 1, columns, kGroup,
                    expected_codes.data() + row * columns / 2, expected_scales.data() + row * columns / kGroup );
            }

            weights = {};

            std::vector<std::uint8_t> device_codes( expected_codes.size() );
            std::vector<std::uint16_t> device_scales( expected_scales.size() );

            ASSERT_EQ( projection.packed->getStorageSize(), device_codes.size() ) << projection.name;
            ASSERT_EQ( projection.scales->getStorageSize(), device_scales.size() * sizeof( std::uint16_t ) ) << projection.name;
            ASSERT_EQ( cudaMemcpy( device_codes.data(), projection.packed->rawData(), device_codes.size(),
                cudaMemcpyDeviceToHost ), cudaSuccess );
            ASSERT_EQ( cudaMemcpy( device_scales.data(), projection.scales->rawData(),
                device_scales.size() * sizeof( std::uint16_t ), cudaMemcpyDeviceToHost ), cudaSuccess );

            std::size_t differing_codes = 0;
            std::size_t differing_scales = 0;

            for ( std::size_t index = 0; index < expected_codes.size(); ++index )
            {
                differing_codes += device_codes[ index ] != expected_codes[ index ];
            }

            for ( std::size_t index = 0; index < expected_scales.size(); ++index )
            {
                differing_scales += device_scales[ index ] != expected_scales[ index ];
            }

            EXPECT_EQ( differing_codes, 0u ) << projection.name;
            EXPECT_EQ( differing_scales, 0u ) << projection.name;

            std::cout << std::format( "[ q4_0 ] {} [{}, {}, {}]: {} of {} code bytes and {} of {} scales differ from the codec\n",
                projection.name, kExperts, projection.rows_per_expert, columns,
                differing_codes, expected_codes.size(), differing_scales, expected_scales.size() );
        }
    }

    // Item 2, as G5b changed it: the gather decode sums in another order than the BF16 bank and the grouped prefill
    // quantizes its activations to INT8, so both are gated against their own references below. Here one token at a
    // time must be bit-identical between a bank built for prefill and one built for one token (Phase 7's two
    // cases); each path's distance from the BF16 bank on exact weights is printed.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_ExactWeights_OneTokenBitIdenticalAcrossBuilds )
    {
        const PackedCase weights_case = q4Case();

        std::vector<float> expected;
        {
            auto reference = makePackedCaseBank<CudaExperts<TensorDataType::BF16>>( weights_case );
            expected = runPackedCase( *reference, weights_case );
        }

        auto quantized = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Q4_0>>( weights_case );
        const std::vector<float> prefill = runPackedCase( *quantized, weights_case );
        const std::vector<float> decoded = runPackedCaseOneTokenAtATime( *quantized, weights_case );

        auto single_token = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Q4_0>>( weights_case, 1 );
        const std::vector<float> single_token_decoded = runPackedCaseOneTokenAtATime( *single_token, weights_case );

        const auto nonzero = std::count_if( expected.begin(), expected.end(), []( float value ) { return value != 0.0f; } );

        EXPECT_GT( nonzero, 0 ) << "the BF16 bank produced all zeros; the comparison proves nothing";
        EXPECT_EQ( countBitMismatches( single_token_decoded, decoded ), 0 )
            << "one token at a time, built for one token against built for prefill, of " << expected.size();

        double prefill_worst = 0.0;

        for ( std::size_t i = 0; i < expected.size(); ++i )
        {
            prefill_worst = std::max( prefill_worst,
                std::fabs( static_cast<double>( prefill[ i ] ) - expected[ i ] ) / ( std::fabs( expected[ i ] ) + 1e-3 ) );
        }

        std::cout << std::format(
            "[ q4_0 ] one token at a time: {} of {} differ between the two builds, {} from the BF16 bank; prefill "
            "(INT8 activations): {} differ, worst relative {:.3e} (not gated)\n",
            countBitMismatches( single_token_decoded, decoded ), expected.size(), countBitMismatches( decoded, expected ),
            countBitMismatches( prefill, expected ), prefill_worst );
    }

    // Item 3.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_SavedBankReloadsBitIdentical )
    {
        const PackedCase weights_case = q4Case();
        const fs::path saved = fs::temp_directory_path() / "mila_moe_q4_0_round_trip.safetensors";

        std::vector<float> original;
        {
            auto bank = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Q4_0>>( weights_case );
            original = runPackedCase( *bank, weights_case );

            Serialization::SafeTensorsWriter writer( saved );
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Declare );
            writer.beginData();
            bank->saveFlatTensors( writer, "experts", Serialization::TensorSavePass::Write );
            writer.close();
        }

        CudaExperts<TensorDataType::BF16, Q4_0> reloaded(
            "experts", MixtureOfExpertsConfig( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );
        reloaded.build( BuildContext( shape_t{ kPackedTokens, kPackedHidden }, RuntimeMode::Inference, false ) );

        std::vector<std::string> names;
        {
            Serialization::WeightsReader reader( saved );
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
        EXPECT_EQ( countBitMismatches( runPackedCase( reloaded, weights_case ), original ), 0 );
    }

    // Item 4.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context = BuildContext( shape_t{ 3, kPackedHidden }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );
        const MixtureOfExpertsConfig config( kPackedHidden, kPackedIntermediate, kPackedExperts, kPackedTopK );

        CudaExperts<TensorDataType::BF16, Q4_0> predictor( "experts", config, Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        CudaExperts<TensorDataType::BF16, Q4_0> built( "experts", config, Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_inactive_parameter_bytes, actual.device_inactive_parameter_bytes ) << "inactive parameters";
        EXPECT_EQ( predicted.device_scratch_bytes, actual.device_scratch_bytes ) << "scratch";
        EXPECT_GT( actual.device_scratch_bytes, std::size_t{ 0 } ) << "three tokens prefill through the grouped path";

        // Packed gate_up 16x128x64 and its FP16 scales 16x128x4 x2; packed down 16x128x32 and its scales 16x128x2 x2.
        EXPECT_EQ( actual.device_parameter_bytes, std::size_t{ 131072 + 16384 + 65536 + 8192 } );
        EXPECT_EQ( actual.device_inactive_parameter_bytes, std::size_t{ 221184 } / 16 * 12 );

        // The 26B's bank, computed rather than built, unrounded: Gemma.md s10.4, the same bytes as PerGroupFp4<64>.
        CudaExperts<TensorDataType::BF16, Q4_0> real( "experts", MixtureOfExpertsConfig( 2816, 704, 128, 8 ), Device::Cuda( 0 ) );
        const MemoryStats real_predicted = real.getRequiredMemory(
            BuildContext( shape_t{ 1, 2816 }, RuntimeMode::Inference, false ).withAllocationGranularity( 0 ) );

        EXPECT_EQ( real_predicted.device_parameter_bytes, std::size_t{ 428'212'224 } );
    }

    // Item 5.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_ExpertWidthNotAMultipleOfTheGroupIsRefused )
    {
        CudaExperts<TensorDataType::BF16, Q4_0> experts(
            "experts", MixtureOfExpertsConfig( kPackedHidden, 48, kPackedExperts, kPackedTopK ), Device::Cuda( 0 ) );

        EXPECT_THROW( experts.build( BuildContext( shape_t{ 4, kPackedHidden }, RuntimeMode::Inference, false ) ), std::invalid_argument );
    }

    // ====================================================================
    // Q4_0 gather decode (Gemma4MoE.md Phase 9, "G5b, against G5's kernel", written before the first run)
    // ====================================================================

    namespace
    {
        // |ref| x 2^-8 for the BF16 output store, plus FP32 accumulation in any order.
        constexpr double kGatherOutputRelative = 1.0 / 256.0;
        constexpr double kGatherAccumulation = 1e-5;
        constexpr double kGatherAbsolute = 1e-7;

        // The largest slope of tanh-approximated GELU, about 1.129: carries the gate's accumulation error through
        // the activation into the gated product.
        constexpr double kGeluSlopeBound = 1.13;

        // FP16 bits of a value FP16 represents as a normal number.
        std::uint16_t halfBits( float value )
        {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
            const std::uint32_t sign = ( bits >> 16 ) & 0x8000u;
            const int exponent = static_cast<int>( ( bits >> 23 ) & 0xFFu ) - 127 + 15;

            if ( exponent <= 0 || exponent >= 31 || ( bits & 0x1FFFu ) != 0 )
            {
                throw std::invalid_argument( "halfBits: not an FP16 normal" );
            }

            return static_cast<std::uint16_t>( sign | ( static_cast<std::uint32_t>( exponent ) << 10 ) | ( ( bits >> 13 ) & 0x3FFu ) );
        }

        float halfValue( std::uint16_t bits )
        {
            const std::uint32_t sign = static_cast<std::uint32_t>( bits & 0x8000u ) << 16;
            const std::uint32_t exponent = ( ( bits >> 10 ) & 0x1Fu ) - 15 + 127;

            return std::bit_cast<float>( sign | ( exponent << 23 ) | ( static_cast<std::uint32_t>( bits & 0x3FFu ) << 13 ) );
        }

        float bf16Truncated( float value )
        {
            return std::bit_cast<float>( std::bit_cast<std::uint32_t>( value ) & 0xFFFF0000u );
        }

        // Random codes and scales for one stacked projection, as the bank stores them.
        struct Q4Projection
        {
            std::vector<std::uint8_t> codes;
            std::vector<std::uint16_t> scales;
            int64_t columns;

            // ( code - 8 ) x d; with swapped set, the nibble order a wrong decoder would read.
            double weight( int64_t row, int64_t column, bool swapped = false ) const
            {
                const std::uint8_t byte = codes[ static_cast<std::size_t>( ( row * columns + column ) / 2 ) ];
                const bool high = ( ( column & 1 ) != 0 ) != swapped;
                const int code = high ? ( byte >> 4 ) : ( byte & 0xF );

                return static_cast<double>( code - 8 ) * halfValue( scales[ static_cast<std::size_t>( ( row * columns + column ) / 32 ) ] );
            }
        };

        Q4Projection randomQ4Projection( int64_t rows, int64_t columns, std::uint32_t seed )
        {
            Q4Projection projection{ std::vector<std::uint8_t>( static_cast<std::size_t>( rows * columns / 2 ) ),
                std::vector<std::uint16_t>( static_cast<std::size_t>( rows * columns / 32 ) ), columns };
            std::mt19937 generator( seed );

            for ( auto& byte : projection.codes )
            {
                byte = static_cast<std::uint8_t>( generator() & 0xFFu );
            }

            // Scales of both signs across a few binades, as absmax / -8 gives them.
            std::uniform_int_distribution<int> mantissa( 1024, 2047 );
            std::uniform_int_distribution<int> binade( -12, -7 );

            for ( auto& scale : projection.scales )
            {
                const float magnitude = std::ldexp( static_cast<float>( mantissa( generator ) ), binade( generator ) - 10 );

                scale = halfBits( ( generator() & 1u ) ? -magnitude : magnitude );
            }

            return projection;
        }
    }

    // Each gather pass against a host reference of the exact weights the codes represent, at the 26B's shapes. The
    // gated pass is read from an installed slot, so the combine is checked on the gated values it actually read.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GatherDecodeMatchesExactReference )
    {
        using Bank = CudaExperts<TensorDataType::BF16, Q4_0>;
        using OutputTensor = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using GatedTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        constexpr int64_t kHidden = 2816;
        constexpr int64_t kIntermediate = 704;
        constexpr int64_t kExperts = 128;
        constexpr int64_t kTopK = 8;

        const Q4Projection gate_up = randomQ4Projection( kExperts * 2 * kIntermediate, kHidden, 20261001u );
        const Q4Projection down = randomQ4Projection( kExperts * kHidden, kIntermediate, 20261002u );

        std::vector<float> input = synthetic( kHidden, 31, 1.0f );
        std::vector<float> combine = synthetic( kTopK, 32, 0.5f );

        for ( float& value : input )
        {
            value = bf16Truncated( value );
        }

        for ( float& value : combine )
        {
            value = bf16Truncated( std::fabs( value ) );
        }

        const std::int32_t selections[ kTopK ] = { 3, 17, 42, 64, 90, 101, 127, 0 };

        auto gated_slot = std::make_shared<GatedTensor>( Device::Cuda( 0 ), shape_t{ kTopK * kIntermediate } );

        Bank bank( "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        bank.installSharedOutputs( std::make_shared<OutputTensor>( Device::Cuda( 0 ), shape_t{ 1, 1, kHidden } ), gated_slot );
        bank.build( BuildContext( shape_t{ 1, 1, kHidden }, RuntimeMode::Inference, false ) );

        const std::vector<ITensor*> stored = bank.getParameters();
        ASSERT_EQ( stored.size(), 4u );

        const std::pair<const void*, std::size_t> uploads[] = {
            { gate_up.codes.data(), gate_up.codes.size() },
            { gate_up.scales.data(), gate_up.scales.size() * sizeof( std::uint16_t ) },
            { down.codes.data(), down.codes.size() },
            { down.scales.data(), down.scales.size() * sizeof( std::uint16_t ) },
        };

        for ( std::size_t index = 0; index < 4; ++index )
        {
            ASSERT_EQ( stored[ index ]->getStorageSize(), uploads[ index ].second ) << "parameter " << index;
            ASSERT_EQ( cudaMemcpy( stored[ index ]->rawData(), uploads[ index ].first, uploads[ index ].second,
                cudaMemcpyHostToDevice ), cudaSuccess );
        }

        const std::vector<float> output = run( bank, input.data(), combine.data(), selections,
            shape_t{ 1, 1, kHidden }, shape_t{ 1, 1, kTopK } );

        auto host_gated = toHost<TensorDataType::FP32>( *gated_slot, context_.get() );
        context_->synchronize();
        const std::vector<float> gated( host_gated.data(), host_gated.data() + host_gated.size() );

        // Out-of-tolerance counts for the decoder's nibble order and for the swapped one, which must fail.
        auto checkGated = [&]( bool swapped, double& worst_ratio )
        {
            int64_t failures = 0;
            worst_ratio = 0.0;

            for ( int64_t slot = 0; slot < kTopK; ++slot )
            {
                for ( int64_t unit = 0; unit < kIntermediate; ++unit )
                {
                    const int64_t gate_row = selections[ slot ] * 2 * kIntermediate + unit;
                    const int64_t up_row = gate_row + kIntermediate;

                    double gate = 0.0;
                    double up = 0.0;
                    double gate_magnitude = 0.0;
                    double up_magnitude = 0.0;

                    for ( int64_t column = 0; column < kHidden; ++column )
                    {
                        const double x = input[ static_cast<std::size_t>( column ) ];
                        const double gate_product = gate_up.weight( gate_row, column, swapped ) * x;
                        const double up_product = gate_up.weight( up_row, column, swapped ) * x;

                        gate += gate_product;
                        up += up_product;
                        gate_magnitude += std::fabs( gate_product );
                        up_magnitude += std::fabs( up_product );
                    }

                    const double expected = geluTanh( gate ) * up;
                    const double tolerance = ( kGeluSlopeBound * gate_magnitude * std::fabs( up )
                        + std::fabs( geluTanh( gate ) ) * up_magnitude ) * kGatherAccumulation + kGatherAbsolute;
                    const double error = std::fabs( gated[ static_cast<std::size_t>( slot * kIntermediate + unit ) ] - expected );

                    worst_ratio = std::max( worst_ratio, error / tolerance );
                    failures += error > tolerance;
                }
            }

            return failures;
        };

        auto checkOutput = [&]( bool swapped, double& worst_ratio )
        {
            int64_t failures = 0;
            worst_ratio = 0.0;

            for ( int64_t column = 0; column < kHidden; ++column )
            {
                double expected = 0.0;
                double magnitude = 0.0;

                for ( int64_t slot = 0; slot < kTopK; ++slot )
                {
                    const int64_t row = selections[ slot ] * kHidden + column;

                    for ( int64_t unit = 0; unit < kIntermediate; ++unit )
                    {
                        const double product = combine[ static_cast<std::size_t>( slot ) ] * down.weight( row, unit, swapped )
                            * gated[ static_cast<std::size_t>( slot * kIntermediate + unit ) ];

                        expected += product;
                        magnitude += std::fabs( product );
                    }
                }

                const double tolerance = std::fabs( expected ) * kGatherOutputRelative + magnitude * kGatherAccumulation + kGatherAbsolute;
                const double error = std::fabs( output[ static_cast<std::size_t>( column ) ] - expected );

                worst_ratio = std::max( worst_ratio, error / tolerance );
                failures += error > tolerance;
            }

            return failures;
        };

        double gated_ratio = 0.0;
        double output_ratio = 0.0;
        double swapped_gated_ratio = 0.0;
        double swapped_output_ratio = 0.0;

        const int64_t gated_failures = checkGated( false, gated_ratio );
        const int64_t output_failures = checkOutput( false, output_ratio );
        const int64_t swapped_gated_failures = checkGated( true, swapped_gated_ratio );
        const int64_t swapped_output_failures = checkOutput( true, swapped_output_ratio );

        EXPECT_EQ( gated_failures, 0 ) << "gated pass, of " << kTopK * kIntermediate;
        EXPECT_EQ( output_failures, 0 ) << "combine pass, of " << kHidden;
        EXPECT_GT( swapped_gated_failures, kTopK * kIntermediate / 2 ) << "the gated check cannot see a swapped nibble order";
        EXPECT_GT( swapped_output_failures, kHidden / 2 ) << "the combine check cannot see a swapped nibble order";

        std::cout << std::format(
            "[ q4_0 gather ] gated {} of {} outside tolerance (worst {:.3f} of it), combine {} of {} (worst {:.3f}); "
            "nibbles swapped: {} and {}\n",
            gated_failures, kTopK * kIntermediate, gated_ratio, output_failures, kHidden, output_ratio,
            swapped_gated_failures, swapped_output_failures );
    }

    // A kernel cannot throw: an out-of-range index poisons the token it routes.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GatherDecodeOutOfRangeExpertPoisonsTheToken )
    {
        const PackedCase weights_case = q4Case();
        auto bank = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Q4_0>>( weights_case, 1 );

        std::vector<std::int32_t> selections( weights_case.selections.begin(), weights_case.selections.begin() + kPackedTopK );
        selections.back() = static_cast<std::int32_t>( kPackedExperts );

        const std::vector<float> output = run( *bank, weights_case.input.data(), weights_case.combine.data(), selections.data(),
            shape_t{ 1, 1, kPackedHidden }, shape_t{ 1, 1, kPackedTopK } );

        const auto poisoned = std::count_if( output.begin(), output.end(), []( float value ) { return std::isnan( value ); } );

        EXPECT_EQ( poisoned, kPackedHidden );
    }

    // ====================================================================
    // Q4_0 grouped prefill (Gemma4MoE.md Phase 9, "G5b, against G5's kernel", written before the first run)
    // ====================================================================

    namespace
    {
        // Linear's per-block activation quantizer as the device runs it, in FP32: scale = largest / 127 and
        // code = round-half-even( x * ( 127 / largest ) ).
        void quantizeInt8PerBlock( const float* values, std::size_t count, std::vector<int>& codes, std::vector<float>& scales )
        {
            codes.resize( count );
            scales.resize( count / 32 );

            for ( std::size_t block = 0; block < count / 32; ++block )
            {
                float largest = 0.0f;

                for ( std::size_t i = 0; i < 32; ++i )
                {
                    largest = std::max( largest, std::fabs( values[ block * 32 + i ] ) );
                }

                const float inverse = largest > 0.0f ? 127.0f / largest : 0.0f;

                for ( std::size_t i = 0; i < 32; ++i )
                {
                    const float scaled = values[ block * 32 + i ] * inverse;

                    codes[ block * 32 + i ] = static_cast<int>( std::nearbyint( scaled ) );
                }

                scales[ block ] = largest / 127.0f;
            }
        }

        // One output of the INT8 block arithmetic: sum over blocks of a_scale x w_scale x sum code_a x ( code_w - 8 ),
        // and the sum of the terms' magnitudes the tolerance scales.
        struct Int8Dot
        {
            double value;
            double magnitude;
        };

        Int8Dot int8Dot( const std::vector<int>& codes, const std::vector<float>& scales, std::size_t first,
            const Q4Projection& projection, int64_t row, bool swapped )
        {
            Int8Dot dot{ 0.0, 0.0 };

            for ( int64_t column = 0; column < projection.columns; ++column )
            {
                const std::size_t element = first + static_cast<std::size_t>( column );
                const double term = static_cast<double>( codes[ element ] ) * scales[ element / 32 ]
                    * projection.weight( row, column, swapped );

                dot.value += term;
                dot.magnitude += std::fabs( term );
            }

            return dot;
        }

        template<typename TBank>
        void uploadQ4Bank( TBank& bank, const Q4Projection& gate_up, const Q4Projection& down )
        {
            const std::vector<ITensor*> stored = bank.getParameters();
            ASSERT_EQ( stored.size(), 4u );

            const std::pair<const void*, std::size_t> uploads[] = {
                { gate_up.codes.data(), gate_up.codes.size() },
                { gate_up.scales.data(), gate_up.scales.size() * sizeof( std::uint16_t ) },
                { down.codes.data(), down.codes.size() },
                { down.scales.data(), down.scales.size() * sizeof( std::uint16_t ) },
            };

            for ( std::size_t index = 0; index < 4; ++index )
            {
                ASSERT_EQ( stored[ index ]->getStorageSize(), uploads[ index ].second ) << "parameter " << index;
                ASSERT_EQ( cudaMemcpy( stored[ index ]->rawData(), uploads[ index ].first, uploads[ index ].second,
                    cudaMemcpyHostToDevice ), cudaSuccess );
            }
        }
    }

    // Each grouped pass against a host reference of the INT8 block arithmetic. With hidden narrower than the expert
    // width, the combine's one pass overwrites only the first hidden / intermediate of the gated slot, so the gated
    // values of the last tokens survive in it: the gated pass is checked on those, and the combine on the same
    // tokens from the gated values it quantized. Hidden 256 takes the 128-deep tile, intermediate 320 the 64-deep;
    // slot 0 routes every token to expert 0, a segment of three M tiles.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GroupedPrefillMatchesInt8Reference )
    {
        using Bank = CudaExperts<TensorDataType::BF16, Q4_0>;
        using OutputTensor = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using GatedTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        constexpr int64_t kHidden = 256;
        constexpr int64_t kIntermediate = 320;
        constexpr int64_t kExperts = 16;
        constexpr int64_t kTopK = 4;
        constexpr int64_t kTokens = 160;
        constexpr int64_t kRows = kTokens * kTopK;

        // Rows whose gated values the combine's output leaves in place.
        constexpr int64_t kFirstSurvivingRow = ( kRows * kHidden + kIntermediate - 1 ) / kIntermediate;
        constexpr int64_t kFirstCheckedToken = ( kFirstSurvivingRow + kTopK - 1 ) / kTopK;

        const Q4Projection gate_up = randomQ4Projection( kExperts * 2 * kIntermediate, kHidden, 20261003u );
        const Q4Projection down = randomQ4Projection( kExperts * kHidden, kIntermediate, 20261004u );

        std::vector<float> input = synthetic( kTokens * kHidden, 41, 1.0f );
        std::vector<float> combine = synthetic( kRows, 42, 0.5f );
        std::vector<std::int32_t> selections( static_cast<std::size_t>( kRows ) );

        for ( float& value : input )
        {
            value = bf16Truncated( value );
        }

        for ( float& value : combine )
        {
            value = bf16Truncated( std::fabs( value ) );
        }

        for ( int64_t token = 0; token < kTokens; ++token )
        {
            selections[ static_cast<std::size_t>( token * kTopK ) ] = 0;

            for ( int64_t slot = 1; slot < kTopK; ++slot )
            {
                selections[ static_cast<std::size_t>( token * kTopK + slot ) ] =
                    static_cast<std::int32_t>( 1 + ( token * 3 + slot * 5 ) % ( kExperts - 1 ) );
            }
        }

        auto gated_slot = std::make_shared<GatedTensor>( Device::Cuda( 0 ), shape_t{ kRows * kIntermediate } );

        Bank bank( "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        bank.installSharedOutputs( std::make_shared<OutputTensor>( Device::Cuda( 0 ), shape_t{ kTokens, kHidden } ), gated_slot );
        bank.build( BuildContext( shape_t{ kTokens, kHidden }, RuntimeMode::Inference, false ) );
        uploadQ4Bank( bank, gate_up, down );

        const std::vector<float> output = run( bank, input.data(), combine.data(), selections.data(),
            shape_t{ kTokens, kHidden }, shape_t{ kTokens, kTopK } );

        auto host_gated = toHost<TensorDataType::FP32>( *gated_slot, context_.get() );
        context_->synchronize();
        const std::vector<float> gated( host_gated.data(), host_gated.data() + host_gated.size() );

        std::vector<int> input_codes;
        std::vector<float> input_scales;
        quantizeInt8PerBlock( input.data(), input.size(), input_codes, input_scales );

        auto checkGated = [&]( bool swapped, double& worst_ratio )
        {
            int64_t failures = 0;
            worst_ratio = 0.0;

            for ( int64_t flat = kFirstCheckedToken * kTopK; flat < kRows; ++flat )
            {
                const int64_t token = flat / kTopK;
                const int64_t expert = selections[ static_cast<std::size_t>( flat ) ];

                for ( int64_t unit = 0; unit < kIntermediate; ++unit )
                {
                    const int64_t gate_row = expert * 2 * kIntermediate + unit;
                    const std::size_t first = static_cast<std::size_t>( token * kHidden );
                    const Int8Dot gate = int8Dot( input_codes, input_scales, first, gate_up, gate_row, swapped );
                    const Int8Dot up = int8Dot( input_codes, input_scales, first, gate_up, gate_row + kIntermediate, swapped );

                    const double expected = geluTanh( gate.value ) * up.value;
                    const double tolerance = ( kGeluSlopeBound * gate.magnitude * std::fabs( up.value )
                        + std::fabs( geluTanh( gate.value ) ) * up.magnitude ) * kGatherAccumulation + kGatherAbsolute;
                    const double error = std::fabs( gated[ static_cast<std::size_t>( flat * kIntermediate + unit ) ] - expected );

                    worst_ratio = std::max( worst_ratio, error / tolerance );
                    failures += error > tolerance;
                }
            }

            return failures;
        };

        // The combine's input is the gated values the device wrote, quantized as the device quantizes them.
        std::vector<int> gated_codes;
        std::vector<float> gated_scales;
        quantizeInt8PerBlock( gated.data() + kFirstCheckedToken * kTopK * kIntermediate,
            static_cast<std::size_t>( ( kTokens - kFirstCheckedToken ) * kTopK * kIntermediate ), gated_codes, gated_scales );

        auto checkOutput = [&]( bool swapped, double& worst_ratio )
        {
            int64_t failures = 0;
            worst_ratio = 0.0;

            for ( int64_t token = kFirstCheckedToken; token < kTokens; ++token )
            {
                for ( int64_t column = 0; column < kHidden; ++column )
                {
                    double expected = 0.0;
                    double magnitude = 0.0;

                    for ( int64_t slot = 0; slot < kTopK; ++slot )
                    {
                        const int64_t flat = token * kTopK + slot;
                        const double weight = combine[ static_cast<std::size_t>( flat ) ];
                        const int64_t row = selections[ static_cast<std::size_t>( flat ) ] * kHidden + column;
                        const Int8Dot dot = int8Dot( gated_codes, gated_scales,
                            static_cast<std::size_t>( ( flat - kFirstCheckedToken * kTopK ) * kIntermediate ), down, row, swapped );

                        expected += weight * dot.value;
                        magnitude += std::fabs( weight ) * dot.magnitude;
                    }

                    const double tolerance = std::fabs( expected ) * kGatherOutputRelative + magnitude * kGatherAccumulation + kGatherAbsolute;
                    const double error = std::fabs( output[ static_cast<std::size_t>( token * kHidden + column ) ] - expected );

                    worst_ratio = std::max( worst_ratio, error / tolerance );
                    failures += error > tolerance;
                }
            }

            return failures;
        };

        double gated_ratio = 0.0;
        double output_ratio = 0.0;
        double ignored = 0.0;

        const int64_t checked_gated = ( kTokens - kFirstCheckedToken ) * kTopK * kIntermediate;
        const int64_t checked_outputs = ( kTokens - kFirstCheckedToken ) * kHidden;
        const int64_t gated_failures = checkGated( false, gated_ratio );
        const int64_t output_failures = checkOutput( false, output_ratio );
        const int64_t swapped_gated_failures = checkGated( true, ignored );
        const int64_t swapped_output_failures = checkOutput( true, ignored );

        ASSERT_GT( kTokens - kFirstCheckedToken, 16 ) << "too few surviving tokens to check";
        EXPECT_EQ( gated_failures, 0 ) << "gated pass, of " << checked_gated;
        EXPECT_EQ( output_failures, 0 ) << "combine pass, of " << checked_outputs;
        EXPECT_GT( swapped_gated_failures, checked_gated / 2 ) << "the gated check cannot see a swapped nibble order";
        EXPECT_GT( swapped_output_failures, checked_outputs / 2 ) << "the combine check cannot see a swapped nibble order";

        std::cout << std::format(
            "[ q4_0 grouped ] tokens {}-{}: gated {} of {} outside tolerance (worst {:.3f} of it), combine {} of {} "
            "(worst {:.3f}); nibbles swapped: {} and {}\n",
            kFirstCheckedToken, kTokens - 1, gated_failures, checked_gated, gated_ratio, output_failures, checked_outputs,
            output_ratio, swapped_gated_failures, swapped_output_failures );
    }

    // Every prefill output depends on its own row alone, so splitting the tokens across calls -- which moves the
    // combine's sub-chunk boundaries and every M tile -- changes no bit. At the 26B's shapes, 160 tokens run four
    // combine passes of 40; the split calls run passes of 40 and 20, and 40, 40 and 20. The 1280 rows take the
    // routing kernel through two of its 1024-row chunks. The distance from one token at a time (BF16 activations)
    // is the INT8 activations' own effect, printed.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GroupedPrefillIndependentOfBatching )
    {
        using Bank = CudaExperts<TensorDataType::BF16, Q4_0>;

        constexpr int64_t kHidden = 2816;
        constexpr int64_t kIntermediate = 704;
        constexpr int64_t kExperts = 128;
        constexpr int64_t kTopK = 8;
        constexpr int64_t kTokens = 160;
        constexpr int64_t kSplit = 60;

        const Q4Projection gate_up = randomQ4Projection( kExperts * 2 * kIntermediate, kHidden, 20261005u );
        const Q4Projection down = randomQ4Projection( kExperts * kHidden, kIntermediate, 20261006u );

        std::vector<float> input = synthetic( kTokens * kHidden, 51, 1.0f );
        std::vector<float> combine = synthetic( kTokens * kTopK, 52, 0.5f );
        std::vector<std::int32_t> selections( static_cast<std::size_t>( kTokens * kTopK ) );
        std::mt19937 generator( 53 );

        for ( float& value : combine )
        {
            value = std::fabs( value );
        }

        for ( int64_t token = 0; token < kTokens; ++token )
        {
            std::vector<std::int32_t> experts( kExperts );

            for ( int64_t expert = 0; expert < kExperts; ++expert )
            {
                experts[ static_cast<std::size_t>( expert ) ] = static_cast<std::int32_t>( expert );
            }

            std::shuffle( experts.begin(), experts.end(), generator );
            std::copy_n( experts.begin(), kTopK, selections.begin() + token * kTopK );
        }

        Bank bank( "experts", MixtureOfExpertsConfig( kHidden, kIntermediate, kExperts, kTopK ), Device::Cuda( 0 ) );
        bank.build( BuildContext( shape_t{ kTokens, kHidden }, RuntimeMode::Inference, false ) );
        uploadQ4Bank( bank, gate_up, down );

        auto runTokens = [&]( int64_t first, int64_t count )
        {
            return run( bank, input.data() + first * kHidden, combine.data() + first * kTopK,
                selections.data() + first * kTopK, shape_t{ count, kHidden }, shape_t{ count, kTopK } );
        };

        const std::vector<float> whole = runTokens( 0, kTokens );

        std::vector<float> split = runTokens( 0, kSplit );
        const std::vector<float> rest = runTokens( kSplit, kTokens - kSplit );
        split.insert( split.end(), rest.begin(), rest.end() );

        std::vector<float> one_at_a_time;

        for ( int64_t token = 0; token < kTokens; ++token )
        {
            const std::vector<float> row = run( bank, input.data() + token * kHidden, combine.data() + token * kTopK,
                selections.data() + token * kTopK, shape_t{ 1, 1, kHidden }, shape_t{ 1, 1, kTopK } );

            one_at_a_time.insert( one_at_a_time.end(), row.begin(), row.end() );
        }

        const auto finite = std::count_if( whole.begin(), whole.end(), []( float value ) { return std::isfinite( value ) && value != 0.0f; } );

        EXPECT_EQ( finite, kTokens * kHidden ) << "every output finite and nonzero";
        EXPECT_EQ( countBitMismatches( split, whole ), 0 ) << "split at token " << kSplit << ", of " << whole.size();

        double worst = 0.0;
        double reference_norm = 0.0;
        double difference_norm = 0.0;

        for ( std::size_t i = 0; i < whole.size(); ++i )
        {
            const double difference = static_cast<double>( whole[ i ] ) - one_at_a_time[ i ];

            worst = std::max( worst, std::fabs( difference ) );
            reference_norm += static_cast<double>( one_at_a_time[ i ] ) * one_at_a_time[ i ];
            difference_norm += difference * difference;
        }

        std::cout << std::format(
            "[ q4_0 grouped ] {} tokens: {} of {} differ when split at {}; against one token at a time (BF16 activations) "
            "worst {:.3e}, relative L2 {:.3e} (not gated)\n",
            kTokens, countBitMismatches( split, whole ), whole.size(), kSplit, worst,
            std::sqrt( difference_norm / std::max( reference_norm, 1e-30 ) ) );
    }

    // A kernel cannot throw: an out-of-range index poisons its own token and no other.
    TEST_F( MixtureOfExpertsCudaTests, Q4_0_GroupedPrefillOutOfRangeExpertPoisonsOnlyThatToken )
    {
        const PackedCase weights_case = q4Case();
        auto bank = makePackedCaseBank<CudaExperts<TensorDataType::BF16, Q4_0>>( weights_case, 2 );

        std::vector<std::int32_t> selections( weights_case.selections.begin(), weights_case.selections.begin() + 2 * kPackedTopK );
        selections.back() = static_cast<std::int32_t>( kPackedExperts );

        const std::vector<float> output = run( *bank, weights_case.input.data(), weights_case.combine.data(), selections.data(),
            shape_t{ 2, kPackedHidden }, shape_t{ 2, kPackedTopK } );

        for ( int64_t column = 0; column < kPackedHidden; ++column )
        {
            EXPECT_TRUE( std::isfinite( output[ static_cast<std::size_t>( column ) ] ) ) << "valid token, column " << column;
            EXPECT_TRUE( std::isnan( output[ static_cast<std::size_t>( kPackedHidden + column ) ] ) ) << "poisoned token, column " << column;
        }
    }
}
