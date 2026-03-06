// Unit tests for LlamaTransformer: build, forward, toString, and fromPretrained on CUDA.
//
// All tests are skipped when no CUDA devices are present.
// fromPretrained integration tests are skipped when weight binaries are absent.

#include <gtest/gtest.h>
#include <filesystem>
#include <string>

import Mila;

namespace Dnn::Components::Transformers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using LlamaCudaType = LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
    using LlamaConfig = Mila::Dnn::LlamaConfig;

    namespace fs = std::filesystem;

    static fs::path llama32_weights_path()
    {
        fs::path dataDir = TEST_DATA_DIR;
        return dataDir / "models" / "llama" / "llama32_1b_fp32.bin";
    }

    // ---- CUDA: build --------------------------------------------------------

    TEST( LlamaTransformerCudaTests, Build_AllocatesParameters )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 128 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f )
            .withRoPEScalingFactor( 1.0f )
            .withBias( false );

        LlamaCudaType net( "llama_cuda_build", cfg, Device::Cuda( 0 ) );

        EXPECT_NO_THROW( net.build( { 1, 8 } ) );
        EXPECT_GT( net.parameterCount(), 0u );
    }

    // ---- CUDA: forward ------------------------------------------------------

    TEST( LlamaTransformerCudaTests, Forward_ProducesLogitsWithExpectedShape )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 128 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f )
            .withRoPEScalingFactor( 1.0f )
            .withBias( false );

        DeviceId cuda_id = Device::Cuda( 0 );
        LlamaCudaType net( "llama_cuda_forward", cfg, cuda_id );

        const int64_t batch = 2;
        const int64_t seq = 5;

        net.build( { batch, cfg.getMaxSequenceLength() } );

        LlamaCudaType::TokenIndexType input( cuda_id, shape_t{ batch, seq } );

        Tensor<dtype_t::INT32, CpuMemoryResource> host_input( Device::Cpu(), shape_t{ batch, seq } );
        int32_t* data = host_input.data();
        size_t total = static_cast<size_t>(batch * seq);

        for ( size_t i = 0; i < total; ++i )
        {
            data[ i ] = static_cast<int32_t>( i % cfg.getVocabSize() );
        }

        copy( host_input, input );

        EXPECT_NO_THROW(
            {
                auto& logits = net.forward( input );
                auto shape = logits.shape();

                ASSERT_EQ( shape.size(), 3u );
                EXPECT_EQ( shape[ 0 ], batch );
                EXPECT_EQ( shape[ 1 ], seq );
                EXPECT_EQ( shape[ 2 ], cfg.getVocabSize() );
            } );
    }

    // ---- CUDA: toString -----------------------------------------------------

    TEST( LlamaTransformerCudaTests, ToString_ContainsExpectedFields )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 128 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f )
            .withRoPEScalingFactor( 1.0f )
            .withBias( false );

        LlamaCudaType net( "llama_cuda_tostring", cfg, Device::Cuda( 0 ) );

        std::string s = net.toString();
        EXPECT_NE( s.find( "Llama Network" ), std::string::npos );
        EXPECT_NE( s.find( "Number of KV heads" ), std::string::npos );
        EXPECT_NE( s.find( "RoPE theta" ), std::string::npos );
    }

    // ---- CUDA: fromPretrained -----------------------------------------------

    TEST( LlamaTransformerCudaTests, FromPretrained_FileNotFound_Throws )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        fs::path missing = fs::temp_directory_path() / "mila_nonexistent_llama.bin";

        if ( fs::exists( missing ) )
        {
            fs::remove( missing );
        }

        EXPECT_THROW(
            LlamaCudaType::fromPretrained( missing, /*batch=*/1, /*seq=*/1, Device::Cuda( 0 ) ),
            std::runtime_error );
    }

    TEST( LlamaTransformerCudaTests, FromPretrained_LoadsIfWeightsPresent )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto p = llama32_weights_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 1B weights not present at: " << p.string();
        }

        auto llama = LlamaCudaType::fromPretrained( p, /*batch=*/1, /*seq=*/128, Device::Cuda( 0 ) );

        EXPECT_GT( llama->parameterCount(), 0u );

        std::string s = llama->toString();
        EXPECT_NE( s.find( "Llama Network" ), std::string::npos );
    }
}