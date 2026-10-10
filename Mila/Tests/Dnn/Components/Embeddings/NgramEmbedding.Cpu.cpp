/**
 * @file NgramEmbedding.Cpu.cpp
 * @brief Concrete-component tests for NgramEmbedding<DeviceType::Cpu, FP32>, and its Qwen 4 reference gate.
 *
 * The gate (Specifications/Qwen4.md section 9, Phase 2) is exact: every n-gram id of every step equals the
 * reference's, across the EOS tokens and the chunk boundary, and the embeddings then equal it bit for bit, since
 * they are a gather. It reads Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py --variant moe and skips when
 * the capture is absent.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Embeddings
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using NgramEmbeddingCpu = Mila::Dnn::NgramEmbedding<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using TensorInt32 = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        // The tiny reference: n-gram size 3, 8 heads per order, 256-wide output, EOS id 5, PLE on layer index 1.
        constexpr dim_t kNgramSize = 3;
        constexpr dim_t kHeadsPerNgram = 8;
        constexpr dim_t kEmbeddingDim = 256;
        constexpr int64_t kEos = 5;
        constexpr dim_t kWidestStep = 13;

        fs::path captureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "Models" / "Qwen4" / "qwen4_tiny_moe";
        }

        /// key=value lines of the reference's constants file.
        /// C stdio, not std::getline: stream input beside import Mila; fails to compile on MSVC (GitHub issue #30).
        std::map<std::string, std::string> readConstants( const fs::path& path )
        {
            std::map<std::string, std::string> values;
            std::string text;
            std::FILE* file = std::fopen( path.string().c_str(), "r" );

            if ( file == nullptr )
            {
                return values;
            }

            char buffer[ 4096 ];
            size_t read = 0;

            while ( (read = std::fread( buffer, 1, sizeof( buffer ), file )) > 0 )
            {
                text.append( buffer, read );
            }

            std::fclose( file );

            size_t start = 0;

            while ( start < text.size() )
            {
                auto end = text.find( '\n', start );

                if ( end == std::string::npos )
                {
                    end = text.size();
                }

                const std::string line = text.substr( start, end - start );
                const auto equals = line.find( '=' );

                if ( equals != std::string::npos )
                {
                    values[ line.substr( 0, equals ) ] = line.substr( equals + 1 );
                }

                start = end + 1;
            }

            return values;
        }

        /// A small hand-built table: sizes 7, 11, 13, 17 for two orders of two heads.
        NgramEmbeddingConfig smallConfig()
        {
            return NgramEmbeddingConfig( 3, 2, 8, 0 )
                .withHashConstants( { 3, 5, 7 }, { 7, 11, 13, 17 }, { 0, 7, 18, 31 }, 48 );
        }

        TensorInt32 idTensor( const std::vector<int32_t>& ids )
        {
            TensorInt32 tensor( Device::Cpu(), shape_t{ 1, static_cast<dim_t>( ids.size() ) } );

            for ( size_t i = 0; i < ids.size(); ++i )
            {
                tensor.data()[ i ] = ids[ i ];
            }

            return tensor;
        }
    }

    // ====================================================================
    // A. Configuration
    // ====================================================================

    TEST( NgramEmbeddingConfigTests, Validate_RefusesOffsetsThatAreNotRunningSums )
    {
        auto config = NgramEmbeddingConfig( 3, 2, 8, 0 ).withHashConstants( { 3, 5, 7 }, { 7, 11, 13, 17 }, { 0, 7, 19, 31 }, 48 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST( NgramEmbeddingConfigTests, Validate_RefusesATableTooSmallForTheHeads )
    {
        auto config = NgramEmbeddingConfig( 3, 2, 8, 0 ).withHashConstants( { 3, 5, 7 }, { 7, 11, 13, 17 }, { 0, 7, 18, 31 }, 47 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST( NgramEmbeddingConfigTests, Metadata_RoundTripsConstantsAboveTwoToTheFiftyThree )
    {
        const int64_t large = 35454616657218897;
        auto source = NgramEmbeddingConfig( 3, 2, 8, 0 ).withHashConstants( { 3, large, 7 }, { 7, 11, 13, 17 }, { 0, 7, 18, 31 }, 48 );

        NgramEmbeddingConfig loaded( 2, 1, 1, 0 );
        loaded.fromMetadata( source.toMetadata() );

        ASSERT_EQ( loaded.getMultipliers().size(), 3u );
        EXPECT_EQ( loaded.getMultipliers()[ 1 ], large );
        EXPECT_EQ( loaded.getHeadOffsets(), ( std::vector<int64_t>{ 0, 7, 18, 31 } ) );
        EXPECT_EQ( loaded.getTableRows(), 48 );
    }

    // ====================================================================
    // B. The hash, by hand
    // ====================================================================

    // Token 4 after 2 and 3: the bigram mixes (4 * 3) ^ (3 * 5) = 12 ^ 15 = 3, the trigram 3 ^ (2 * 7) = 3 ^ 14 = 13.
    TEST( NgramEmbeddingCpuTests, Hash_MatchesHandComputedRows )
    {
        NgramEmbeddingCpu embedding( "ngram", smallConfig(), Device::Cpu() );
        embedding.build( BuildContext( shape_t{ 1, 3 }, RuntimeMode::Inference, false ) );

        auto ids = idTensor( { 2, 3, 4 } );
        (void)embedding.prefill( ids, 0 );

        const auto& rows = embedding.ngramIds();
        const int32_t* last = rows.data() + 2 * 4;

        EXPECT_EQ( last[ 0 ], 3 % 7 + 0 );
        EXPECT_EQ( last[ 1 ], 3 % 11 + 7 );
        EXPECT_EQ( last[ 2 ], 13 % 13 + 18 );
        EXPECT_EQ( last[ 3 ], 13 % 17 + 31 );
    }

    // An EOS ends a segment: the token after it sees only eos behind it, as the first token of a sequence does.
    TEST( NgramEmbeddingCpuTests, Hash_TokenAfterEosHashesLikeTheFirstToken )
    {
        NgramEmbeddingCpu embedding( "ngram", smallConfig(), Device::Cpu() );
        embedding.build( BuildContext( shape_t{ 1, 4 }, RuntimeMode::Inference, false ) );

        auto ids = idTensor( { 4, 9, 0, 4 } );
        (void)embedding.prefill( ids, 0 );

        const auto& rows = embedding.ngramIds();

        for ( int head = 0; head < 4; ++head )
        {
            EXPECT_EQ( rows.data()[ 3 * 4 + head ], rows.data()[ head ] ) << "head " << head;
        }
    }

    // A negative mixed value wraps to a non-negative row: floor mod, not C++'s truncating %.
    TEST( NgramEmbeddingCpuTests, Hash_NegativeMixedValueMapsIntoTheHead )
    {
        auto config = NgramEmbeddingConfig( 2, 1, 4, 0 ).withHashConstants( { -3, 1 }, { 7 }, { 0 }, 8 );
        NgramEmbeddingCpu embedding( "ngram", config, Device::Cpu() );
        embedding.build( BuildContext( shape_t{ 1, 1 }, RuntimeMode::Inference, false ) );

        // ( 1 * -3 ) ^ ( 0 * 1 ) = -3, and floor_mod( -3, 7 ) = 4 where -3 % 7 = -3.
        auto ids = idTensor( { 1 } );
        (void)embedding.prefill( ids, 0 );

        EXPECT_EQ( embedding.ngramIds().data()[ 0 ], 4 );
    }

    TEST( NgramEmbeddingCpuTests, DecodeBeforeAnyPrefillIsRefused )
    {
        NgramEmbeddingCpu embedding( "ngram", smallConfig(), Device::Cpu() );
        embedding.build( BuildContext( shape_t{ 1, 1 }, RuntimeMode::Inference, false ) );

        auto ids = idTensor( { 3 } );

        EXPECT_THROW( (void)embedding.decode( ids, 0 ), std::logic_error );
    }

    // ====================================================================
    // C. The Qwen 4 reference gate (Qwen4.md Phase 2)
    // ====================================================================

    TEST( NgramEmbeddingReferenceTests, EveryIdAndEveryEmbeddingEqualsTheCapture )
    {
        const fs::path capture_path = captureDirectory() / "qwen4_tiny_moe_reference.safetensors";
        const fs::path constants_path = captureDirectory() / "qwen4_tiny_moe_ngram_constants.txt";

        if ( !fs::exists( capture_path ) || !fs::exists( constants_path ) )
        {
            GTEST_SKIP() << "Qwen 4 tiny reference not present at: " << captureDirectory().string();
        }

        auto constants = readConstants( constants_path );
        auto config = NgramEmbeddingConfig( kNgramSize, kHeadsPerNgram, kEmbeddingDim, kEos )
            .withHashConstants(
                NgramEmbeddingConfig::splitIntegers( constants.at( "layer1.multipliers" ) ),
                NgramEmbeddingConfig::splitIntegers( constants.at( "layer1.head_vocab_sizes" ) ),
                NgramEmbeddingConfig::splitIntegers( constants.at( "layer1.head_offsets" ) ),
                static_cast<dim_t>( std::stoll( constants.at( "layer1.table_rows" ) ) ) );

        Serialization::WeightsReader capture( capture_path );

        NgramEmbeddingCpu embedding( "ngram", config, Device::Cpu() );
        embedding.build( BuildContext( shape_t{ 1, kWidestStep }, RuntimeMode::Inference, false ) );
        embedding.loadParameter( "weight", capture.readTensorBlob<CpuMemoryResource>( "weights.layer1.ple.ngram_table" ) );

        const auto token_blob = capture.readTensorBlob<CpuMemoryResource>( "tokens" );
        const auto* tokens = static_cast<const int32_t*>( static_cast<const void*>( token_blob.data() ) );

        struct Step
        {
            std::string name;
            dim_t first;
            dim_t length;
        };

        std::vector<Step> steps{ { "prefill0", 0, 13 }, { "prefill1", 13, 11 } };

        for ( dim_t t = 24; t < 32; ++t )
        {
            steps.push_back( { "decode" + std::to_string( t - 24 ), t, 1 } );
        }

        int64_t id_mismatches = 0;
        int64_t embedding_mismatches = 0;

        for ( const auto& step : steps )
        {
            std::vector<int32_t> chunk( tokens + step.first, tokens + step.first + step.length );
            auto ids = idTensor( chunk );

            const auto& output = step.first == 0 ? embedding.prefill( ids, 0 )
                : step.length > 1 ? embedding.prefill( ids, step.first ) : embedding.decode( ids, step.first );

            const auto expected_ids = capture.readTensorBlob<CpuMemoryResource>( step.name + ".layer1.ple.ngram_ids" );
            const auto expected_embedding = capture.readTensorBlob<CpuMemoryResource>( step.name + ".layer1.ple.ngram_embedding" );

            const auto* want_ids = static_cast<const int32_t*>( static_cast<const void*>( expected_ids.data() ) );
            const auto* want_embedding = static_cast<const float*>( static_cast<const void*>( expected_embedding.data() ) );
            const auto& got_ids = embedding.ngramIds();

            ASSERT_EQ( got_ids.size(), static_cast<dim_t>( expected_ids.sizeBytes() / sizeof( int32_t ) ) ) << step.name;
            ASSERT_EQ( output.size(), static_cast<dim_t>( expected_embedding.sizeBytes() / sizeof( float ) ) ) << step.name;

            for ( dim_t i = 0; i < got_ids.size(); ++i )
            {
                if ( got_ids.data()[ i ] != want_ids[ i ] )
                {
                    ++id_mismatches;
                    ADD_FAILURE() << step.name << " id " << i << ": " << got_ids.data()[ i ] << " vs " << want_ids[ i ];
                }
            }

            embedding_mismatches += std::memcmp( output.data(), want_embedding, expected_embedding.sizeBytes() ) != 0 ? 1 : 0;
        }

        EXPECT_EQ( id_mismatches, 0 );
        EXPECT_EQ( embedding_mismatches, 0 ) << "steps whose embeddings differ from the capture in any bit";
    }
}
