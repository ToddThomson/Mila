/**
 * @file NgramEmbedding.Cuda.cpp
 * @brief NgramEmbedding<DeviceType::Cuda, {FP32, BF16}>, its Qwen 4 reference gate, and its decode under replay.
 *
 * The gate (Specifications/Qwen4.md section 9, Phase 2) is exact: every n-gram id equals the reference's across the
 * EOS tokens and the chunk boundary, and every embedding is the table row those ids name. It reads
 * Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py --variant moe and skips when the capture is absent.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;
import Compute.CudaExecutionContext;

namespace Mila::Tests::Dnn::Components::Embeddings
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostInt32 = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using DeviceInt32 = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;

        // The tiny reference: n-gram size 3, 8 heads per order, 256-wide output, EOS id 5, PLE on layer index 1.
        constexpr dim_t kNgramSize = 3;
        constexpr dim_t kHeadsPerNgram = 8;
        constexpr dim_t kEmbeddingDim = 256;
        constexpr int64_t kEos = 5;
        constexpr dim_t kWidestStep = 13;

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

        using NgramPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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

        /// key=value lines of the reference's constants file.
        std::map<std::string, std::string> readConstants( const fs::path& path )
        {
            std::map<std::string, std::string> values;
            std::ifstream file( path );
            std::string line;

            while ( std::getline( file, line ) )
            {
                const auto equals = line.find( '=' );

                if ( equals != std::string::npos )
                {
                    values[ line.substr( 0, equals ) ] = line.substr( equals + 1 );
                }
            }

            return values;
        }

        /// A small hand-built table: sizes 7, 11, 13, 17 for two orders of two heads.
        NgramEmbeddingConfig smallConfig()
        {
            return NgramEmbeddingConfig( 3, 2, 8, 0 )
                .withHashConstants( { 3, 5, 7 }, { 7, 11, 13, 17 }, { 0, 7, 18, 31 }, 48 );
        }
    }

    template<typename TPrecisionTag>
    class NgramEmbeddingCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using EmbeddingType = Mila::Dnn::NgramEmbedding<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;

        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        std::unique_ptr<EmbeddingType> built( const NgramEmbeddingConfig& config, dim_t length )
        {
            auto embedding = std::make_unique<EmbeddingType>( "ngram", config, Device::Cuda( 0 ) );
            embedding->build( BuildContext( shape_t{ 1, length }, RuntimeMode::Inference, true ) );

            return embedding;
        }

        DeviceInt32 idTensor( const std::vector<int32_t>& ids )
        {
            HostInt32 host( Device::Cpu(), shape_t{ 1, static_cast<dim_t>( ids.size() ) } );

            for ( size_t i = 0; i < ids.size(); ++i )
            {
                host.data()[ i ] = ids[ i ];
            }

            DeviceInt32 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, context_.get() );
            context_->synchronize();

            return device;
        }

        std::vector<int32_t> idsOnHost( EmbeddingType& embedding )
        {
            embedding.synchronize();

            const auto& device = embedding.ngramIds();
            HostInt32 host( Device::Cpu(), device.shape() );
            copy( device, host, context_.get() );
            context_->synchronize();

            return std::vector<int32_t>( host.data(), host.data() + host.size() );
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

    TYPED_TEST_SUITE( NgramEmbeddingCudaTests, NgramPrecisions, PrecisionNames );

    // ====================================================================
    // A. The hash, by hand
    // ====================================================================

    // Token 4 after 2 and 3: the bigram mixes (4 * 3) ^ (3 * 5) = 12 ^ 15 = 3, the trigram 3 ^ (2 * 7) = 3 ^ 14 = 13.
    TYPED_TEST( NgramEmbeddingCudaTests, Hash_MatchesHandComputedRows )
    {
        auto embedding = this->built( smallConfig(), 3 );
        (void)embedding->prefill( this->idTensor( { 2, 3, 4 } ), 0 );

        const auto rows = this->idsOnHost( *embedding );
        const int32_t* last = rows.data() + 2 * 4;

        EXPECT_EQ( last[ 0 ], 3 % 7 + 0 );
        EXPECT_EQ( last[ 1 ], 3 % 11 + 7 );
        EXPECT_EQ( last[ 2 ], 13 % 13 + 18 );
        EXPECT_EQ( last[ 3 ], 13 % 17 + 31 );
    }

    // An EOS ends a segment: the token after it sees only eos behind it, as the first token of a sequence does.
    TYPED_TEST( NgramEmbeddingCudaTests, Hash_TokenAfterEosHashesLikeTheFirstToken )
    {
        auto embedding = this->built( smallConfig(), 4 );
        (void)embedding->prefill( this->idTensor( { 4, 9, 0, 4 } ), 0 );

        const auto rows = this->idsOnHost( *embedding );

        for ( int head = 0; head < 4; ++head )
        {
            EXPECT_EQ( rows[ 3 * 4 + head ], rows[ head ] ) << "head " << head;
        }
    }

    // A negative mixed value wraps to a non-negative row: floor mod, not C++'s truncating %.
    TYPED_TEST( NgramEmbeddingCudaTests, Hash_NegativeMixedValueMapsIntoTheHead )
    {
        auto config = NgramEmbeddingConfig( 2, 1, 4, 0 ).withHashConstants( { -3, 1 }, { 7 }, { 0 }, 8 );
        auto embedding = this->built( config, 1 );

        // ( 1 * -3 ) ^ ( 0 * 1 ) = -3, and floor_mod( -3, 7 ) = 4 where -3 % 7 = -3.
        (void)embedding->prefill( this->idTensor( { 1 } ), 0 );

        EXPECT_EQ( this->idsOnHost( *embedding )[ 0 ], 4 );
    }

    // A chunk shorter than the history keeps the older ids: the decode after [2, 3] then [4] hashes 4 after 3 and 2.
    TYPED_TEST( NgramEmbeddingCudaTests, ChunkShorterThanTheHistoryEqualsOneChunk )
    {
        auto whole = this->built( smallConfig(), 4 );
        (void)whole->prefill( this->idTensor( { 2, 3, 4, 6 } ), 0 );
        const auto expected = this->idsOnHost( *whole );

        auto stepped = this->built( smallConfig(), 4 );
        (void)stepped->prefill( this->idTensor( { 2 } ), 0 );

        for ( int t = 1; t < 4; ++t )
        {
            const int32_t token = std::vector<int32_t>{ 2, 3, 4, 6 }[ t ];
            (void)stepped->decode( this->idTensor( { token } ), t );

            const auto got = this->idsOnHost( *stepped );

            for ( int head = 0; head < 4; ++head )
            {
                EXPECT_EQ( got[ head ], expected[ t * 4 + head ] ) << "position " << t << " head " << head;
            }
        }
    }

    TYPED_TEST( NgramEmbeddingCudaTests, DecodeBeforeAnyPrefillIsRefused )
    {
        auto embedding = this->built( smallConfig(), 1 );

        EXPECT_THROW( (void)embedding->decode( this->idTensor( { 3 } ), 0 ), std::logic_error );
    }

    TYPED_TEST( NgramEmbeddingCudaTests, AConfigBeyondTheKernelLimitsIsRefused )
    {
        std::vector<int64_t> multipliers( 10, 3 );
        std::vector<int64_t> sizes( 9, 7 );
        std::vector<int64_t> offsets( 9 );

        for ( size_t i = 0; i < offsets.size(); ++i )
        {
            offsets[ i ] = static_cast<int64_t>( i ) * 7;
        }

        auto config = NgramEmbeddingConfig( 10, 1, 9, 0 ).withHashConstants( multipliers, sizes, offsets, 63 );

        EXPECT_THROW( ( typename TestFixture::EmbeddingType( "ngram", config, Device::Cuda( 0 ) ) ), std::runtime_error );
    }

    // ====================================================================
    // B. Decode under replay (Qwen4.md Phase 2 gate; DecodeGraph.md)
    // ====================================================================

    // One decode step is recorded as a graph and replayed for every later token, each written into the same input
    // buffer. A step that read a host value, or a launch argument that changes per token, would diverge from a
    // twin that calls decode every time.
    TYPED_TEST( NgramEmbeddingCudaTests, ReplayedDecodeEqualsCalledDecode )
    {
        const std::vector<int32_t> tokens{ 2, 3, 4, 0, 6, 2, 9, 4, 4, 0, 1, 3, 5, 0, 7 };
        constexpr int kPrompt = 4;

        auto called = this->built( smallConfig(), kPrompt );
        auto replayed = this->built( smallConfig(), kPrompt );

        const std::vector<int32_t> prompt( tokens.begin(), tokens.begin() + kPrompt );
        (void)called->prefill( this->idTensor( prompt ), 0 );
        (void)replayed->prefill( this->idTensor( prompt ), 0 );

        // The first decode is called on both: it primes the state and settles the views, which are host work.
        auto input = this->idTensor( { tokens[ kPrompt ] } );
        (void)called->decode( this->idTensor( { tokens[ kPrompt ] } ), kPrompt );
        (void)replayed->decode( input, kPrompt );
        replayed->synchronize();

        auto* cuda_context = dynamic_cast<CudaExecutionContext*>( replayed->getExecutionContext() );
        ASSERT_NE( cuda_context, nullptr );
        cudaStream_t stream = cuda_context->getStream();

        HostInt32 host_token( Device::Cpu(), shape_t{ 1, 1 } );

        auto writeToken = [&]( int32_t token ) {
            host_token.data()[ 0 ] = token;
            ASSERT_EQ( cudaMemcpyAsync( input.rawData(), host_token.data(), sizeof( int32_t ), cudaMemcpyHostToDevice, stream ), cudaSuccess );
        };

        writeToken( tokens[ kPrompt + 1 ] );

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;

        ASSERT_EQ( cudaStreamBeginCapture( stream, cudaStreamCaptureModeThreadLocal ), cudaSuccess );
        (void)replayed->decode( input, kPrompt + 1 );
        ASSERT_EQ( cudaStreamEndCapture( stream, &graph ), cudaSuccess );
        ASSERT_EQ( cudaGraphInstantiate( &graph_exec, graph, 0 ), cudaSuccess );

        for ( int t = kPrompt + 1; t < static_cast<int>( tokens.size() ); ++t )
        {
            if ( t > kPrompt + 1 )
            {
                writeToken( tokens[ t ] );
            }

            ASSERT_EQ( cudaGraphLaunch( graph_exec, stream ), cudaSuccess );
            (void)called->decode( this->idTensor( { tokens[ t ] } ), t );

            EXPECT_EQ( this->idsOnHost( *replayed ), this->idsOnHost( *called ) ) << "position " << t;
        }

        cudaGraphExecDestroy( graph_exec );
        cudaGraphDestroy( graph );
    }

    // ====================================================================
    // C. The Qwen 4 reference gate (Qwen4.md Phase 2)
    // ====================================================================

    TYPED_TEST( NgramEmbeddingCudaTests, EveryIdAndEveryEmbeddingEqualsTheCapture )
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

        typename TestFixture::EmbeddingType embedding( "ngram", config, Device::Cuda( 0 ) );
        embedding.build( BuildContext( shape_t{ 1, kWidestStep }, RuntimeMode::Inference, false ) );
        // The capture's table is FP32, which a BF16 parameter refuses to load; it is converted on the way in instead.
        using DeviceTensor = typename TestFixture::DeviceTensor;
        {
            const auto blob = capture.readTensorBlob<CpuMemoryResource>( "weights.layer1.ple.ngram_table" );
            HostFp32 host_table( Device::Cpu(), blob.getMetadata().shape );
            std::memcpy( host_table.data(), blob.data(), blob.sizeBytes() );

            copy( host_table, *static_cast<DeviceTensor*>( embedding.getParameters()[ 0 ] ), this->context_.get() );
            this->context_->synchronize();
        }

        // The table as the device holds it: the FP32 capture rounded to the precision, for BF16.
        const auto table = this->toFloat( *static_cast<const DeviceTensor*>( embedding.getParameters()[ 0 ] ) );
        const dim_t row_width = config.getRowWidth();

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
            auto ids = this->idTensor( chunk );

            const auto& device_output = step.first == 0 ? embedding.prefill( ids, 0 )
                : step.length > 1 ? embedding.prefill( ids, step.first ) : embedding.decode( ids, step.first );

            const auto output = this->toFloat( device_output );
            const auto got_ids = this->idsOnHost( embedding );

            const auto expected_ids = capture.readTensorBlob<CpuMemoryResource>( step.name + ".layer1.ple.ngram_ids" );
            const auto* want_ids = static_cast<const int32_t*>( static_cast<const void*>( expected_ids.data() ) );

            ASSERT_EQ( static_cast<dim_t>( got_ids.size() ), static_cast<dim_t>( expected_ids.sizeBytes() / sizeof( int32_t ) ) ) << step.name;

            for ( size_t i = 0; i < got_ids.size(); ++i )
            {
                if ( got_ids[ i ] != want_ids[ i ] )
                {
                    ++id_mismatches;
                    ADD_FAILURE() << step.name << " id " << i << ": " << got_ids[ i ] << " vs " << want_ids[ i ];
                }
            }

            // Every embedding is the table row its id names, bit for bit; at FP32 that is also the capture's value.
            for ( size_t i = 0; i < got_ids.size(); ++i )
            {
                const float* expected_row = table.data() + static_cast<dim_t>( want_ids[ i ] ) * row_width;

                if ( std::memcmp( output.data() + static_cast<dim_t>( i ) * row_width, expected_row, static_cast<size_t>( row_width ) * sizeof( float ) ) != 0 )
                {
                    ++embedding_mismatches;
                }
            }

            if constexpr ( TypeParam::value == TensorDataType::FP32 )
            {
                const auto expected_embedding = capture.readTensorBlob<CpuMemoryResource>( step.name + ".layer1.ple.ngram_embedding" );

                EXPECT_EQ( std::memcmp( output.data(), expected_embedding.data(), expected_embedding.sizeBytes() ), 0 )
                    << step.name << ": the embedding differs from the capture";
            }
        }

        EXPECT_EQ( id_mismatches, 0 );
        EXPECT_EQ( embedding_mismatches, 0 ) << "rows that are not the table row their id names";
    }
}
