// Unit tests covering Gpt network build/forward/toString and fromPretrained factory.
//
// Provides fast unit tests that don't require external weights, and an integration
// test that attempts to load a converted GPT-2 .bin when present in the repo.

#include <gtest/gtest.h>
#include <filesystem>
#include <string>

import Mila;

namespace Dnn::Components::Transformers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using GptType = GptTransformer<DeviceType::Cpu, TensorDataType::FP32>;
    using GptConfig = Mila::Dnn::GptConfig;
    namespace fs = std::filesystem;

    // Helper: path to GPT-2 weights under TEST_DATA_DIR (set via CMake target_compile_definitions)
    static fs::path gpt2_weights_path()
    {
        fs::path dataDir = TEST_DATA_DIR;
        return dataDir / "models" / "gpt2" / "gpt2_small_fp32.bin";
    }

    // Helper to create a small, valid GptConfig for fast tests.
    static GptConfig makeSmallConfig()
    {
        GptConfig cfg( /*embedding_size=*/16, /*num_layers=*/2 );
        cfg.withVocabSize( 97 )
            .withNumHeads( 2 )
            .withMaxSequenceLength( 64 )
            .withHiddenSize( 64 )
            .withBias( true );

        return cfg;
    }

    TEST( GptTests, BuildAllocatesParametersAndParameterCount )
    {
        auto cfg = makeSmallConfig();

        GptType net( "gpt_test_build", cfg, Device::Cpu() );

        // Build with a small shape (batch, seq)
        shape_t build_shape = { 1, 8 };
        EXPECT_NO_THROW( net.build( build_shape ) );

        // Parameter count should be positive after build
        auto params = net.parameterCount();
        EXPECT_GT( params, 0u );
    }

    TEST( GptTests, ForwardProducesLogitsWithExpectedShape )
    {
        auto cfg = makeSmallConfig();

        GptType net( "gpt_test_forward", cfg, Device::Cpu() );

        const int64_t batch = 2;
        const int64_t seq = 5;

        // Build network with runtime positional capacity
        shape_t build_shape = { batch, cfg.getMaxSequenceLength() };
        net.build( build_shape );

        // Create compact inference input (batch, seq)
        typename GptType::TokenIndexType input( Device::Cpu(), shape_t{ batch, seq } );

        // Fill input with valid token indices in range [0, vocab)
        int32_t* data_ptr = input.data();
        size_t total = static_cast<size_t>(batch * seq);
        for ( size_t i = 0; i < total; ++i )
        {
            data_ptr[ i ] = static_cast<int32_t>( i % cfg.getVocabSize() );
        }

        // Run forward on the shorter slice (network permits seq <= built max)
        EXPECT_NO_THROW( {
            auto& logits = net.forward( input );

            // Verify output shape: (batch, seq, vocab)
            auto out_shape = logits.shape();
            ASSERT_EQ( out_shape.size(), 3u );
            EXPECT_EQ( out_shape[ 0 ], batch );
            EXPECT_EQ( out_shape[ 1 ], seq );
            EXPECT_EQ( out_shape[ 2 ], cfg.getVocabSize() );
        } );
    }

    TEST( GptTests, ToStringAndConfigToString )
    {
        auto cfg = makeSmallConfig();

        GptType net( "gpt_test_tostring", cfg, Device::Cpu() );

        // toString should contain the network name and some config fields
        std::string s = net.toString();
        EXPECT_NE( s.find( "Gpt Network" ), std::string::npos );
        EXPECT_NE( s.find( "Embedding Size/Dim" ), std::string::npos );

        // Ensure config's toString returns a short human-readable summary
        std::string cfg_str = cfg.toString();
        EXPECT_NE( cfg_str.find( "embedding size" ), std::string::npos );
    }

    TEST( GptTests, FromPretrained_FileNotFound_Throws )
    {
        // Ensure non-existent path triggers a runtime error from the factory.
        std::filesystem::path missing = std::filesystem::temp_directory_path() / "mila_nonexistent_model.bin";

        // Ensure file does not exist
        if ( std::filesystem::exists( missing ) )
            std::filesystem::remove( missing );

        EXPECT_THROW( GptType::fromPretrained( missing, /*batch=*/1, /*seq=*/1, Device::Cpu() ),
            std::runtime_error );
    }

    TEST( GptTests, FromPretrained_LoadsIfWeightsPresent )
    {
        // Check repository-local Data/Models/GPT2 for the converted weights as requested.
        std::filesystem::path model_path = std::filesystem::path( "." ) / "Data" / "Models" / "GPT2" / "gpt2_small_fp32.bin";

        if ( !std::filesystem::exists( model_path ) )
        {
            GTEST_SKIP() << "GPT-2 converted weights not present at: " << model_path.string()
                         << " - skipping integration test.";
        }

        // If file exists, ensure factory returns a built Gpt instance without throwing.
        auto gpt2 = GptType::fromPretrained( model_path, /*batch=*/1, /*seq=*/8, Device::Cpu() );

        // Basic sanity: parameters allocated
        EXPECT_GT( gpt2->parameterCount(), 0u );

        // toString should include model name (metadata provided name may be empty)
        std::string s = gpt2->toString();
        EXPECT_NE( s.find( "Gpt Network" ), std::string::npos );
    }
}
