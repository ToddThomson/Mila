#include <gtest/gtest.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdint>
#include <string>

import Mila;

namespace Mila::Dnn::Data::Tests
{
    using namespace Mila::Dnn::Compute;

    class StreamingDataLoaderTest : public ::testing::Test
    {
    protected:
        std::filesystem::path temp_tokens_file_;
        size_t num_test_tokens_{ 10000 };

        void SetUp() override
        {
            temp_tokens_file_ = std::filesystem::temp_directory_path() / "test_tokens.tokens";
        }

        void TearDown() override
        {
            if ( std::filesystem::exists( temp_tokens_file_ ) )
            {
                std::filesystem::remove( temp_tokens_file_ );
            }
        }

        void createTokensFile( size_t num_tokens, const std::vector<TokenId>& pattern = {} )
        {
            std::ofstream file( temp_tokens_file_, std::ios::binary );
            ASSERT_TRUE( file.is_open() ) << "Failed to create test tokens file";

            if ( pattern.empty() )
            {
                for ( size_t i = 0; i < num_tokens; ++i )
                {
                    TokenId token = static_cast<TokenId>( i % 50000 );
                    file.write( reinterpret_cast<const char*>( &token ), sizeof( TokenId ) );
                }
            }
            else
            {
                for ( size_t i = 0; i < num_tokens; ++i )
                {
                    TokenId token = pattern[ i % pattern.size() ];
                    file.write( reinterpret_cast<const char*>( &token ), sizeof( TokenId ) );
                }
            }

            file.close();
        }

        void verifySequentialTokens( const StreamingDataLoader<CpuMemoryResource>& loader )
        {
            const auto& inputs = loader.inputs();
            const auto& targets = loader.targets();

            for ( int64_t b = 0; b < loader.batchSize(); ++b )
            {
                for ( int64_t t = 0; t < loader.sequenceLength() - 1; ++t )
                {
                    int32_t input_val = inputs.data()[ b * loader.sequenceLength() + t ];
                    int32_t next_input_val = inputs.data()[ b * loader.sequenceLength() + t + 1 ];
                    int32_t target_val = targets.data()[ b * loader.sequenceLength() + t ];

                    // Target should equal the next token in the input sequence (shifted by 1),
                    // not necessarily numeric input+1 (wrapping or custom patterns can occur).
                    EXPECT_EQ( target_val, next_input_val )
                        << "Target should be the next token in the input sequence at batch " << b << ", position " << t;
                }
            }
        }
    };

    TEST_F( StreamingDataLoaderTest, ConstructorValidation_InvalidSequenceLength )
    {
        createTokensFile( num_test_tokens_ );

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    0,
                    false,
                    Device::Cpu() );
            },
            std::invalid_argument );

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    -10,
                    false,
                    Device::Cpu() );
            },
            std::invalid_argument );
    }

    TEST_F( StreamingDataLoaderTest, ConstructorValidation_MissingFile )
    {
        std::filesystem::path non_existent = std::filesystem::temp_directory_path() / "non_existent.tokens";

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    non_existent,
                    4,
                    128,
                    false,
                    Device::Cpu() );
            },
            std::runtime_error );
    }

    TEST_F( StreamingDataLoaderTest, ConstructorValidation_InvalidFileSize )
    {
        std::ofstream file( temp_tokens_file_, std::ios::binary );
        const char invalid_data[] = { 1, 2, 3 };
        file.write( invalid_data, sizeof( invalid_data ) );
        file.close();

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    128,
                    false,
                    Device::Cpu() );
            },
            std::runtime_error );
    }

    TEST_F( StreamingDataLoaderTest, ConstructorValidation_InsufficientTokens )
    {
        createTokensFile( 100 );

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    128,
                    false,
                    Device::Cpu() );
            },
            std::runtime_error );
    }

    TEST_F( StreamingDataLoaderTest, ConstructorValidation_DeviceTypeMismatch )
    {
        createTokensFile( num_test_tokens_ );

        EXPECT_THROW(
            {
                StreamingDataLoader<CpuMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    128,
                    false,
                    Device::Cuda( 0 ) );
            },
            std::runtime_error );
    }

    TEST_F( StreamingDataLoaderTest, BasicFunctionality_CPUMemory )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 4;
        const int64_t seq_length = 128;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        EXPECT_GT( loader.numBatches(), 0 );
        EXPECT_EQ( loader.batchSize(), batch_size );
        EXPECT_EQ( loader.sequenceLength(), seq_length );
        EXPECT_EQ( loader.numTokens(), num_test_tokens_ );
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );

        const auto& inputs = loader.inputs();
        const auto& targets = loader.targets();

        EXPECT_EQ( inputs.shape()[ 0 ], batch_size );
        EXPECT_EQ( inputs.shape()[ 1 ], seq_length );
        EXPECT_EQ( targets.shape()[ 0 ], batch_size );
        EXPECT_EQ( targets.shape()[ 1 ], seq_length );
    }

    TEST_F( StreamingDataLoaderTest, BatchIteration_NonTrainingMode )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 2;
        const int64_t seq_length = 64;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        const int64_t total_batches = loader.numBatches();
        EXPECT_GT( total_batches, 0 );

        for ( int64_t i = 0; i < total_batches; ++i )
        {
            EXPECT_EQ( loader.currentBatch(), i );
            EXPECT_TRUE( loader.hasNext() );

            loader.nextBatch();

            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
        }

        EXPECT_EQ( loader.currentBatch(), total_batches );
        EXPECT_FALSE( loader.hasNext() );
    }

    TEST_F( StreamingDataLoaderTest, BatchIteration_TrainingMode )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 4;
        const int64_t seq_length = 128;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            true,
            Device::Cpu() );

        const int64_t total_batches = loader.numBatches();
        EXPECT_GT( total_batches, 0 );

        std::vector<std::vector<int32_t>> first_epoch_data;

        for ( int64_t i = 0; i < total_batches && i < 5; ++i )
        {
            loader.nextBatch();
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );

            const auto& inputs = loader.inputs();
            std::vector<int32_t> batch_data( inputs.data(), inputs.data() + batch_size * seq_length );
            first_epoch_data.push_back( batch_data );
        }

        loader.reset();

        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );

        for ( int64_t i = 0; i < std::min( total_batches, 5LL ); ++i )
        {
            loader.nextBatch();
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
        }
    }

    TEST_F( StreamingDataLoaderTest, Reset_Functionality )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 2;
        const int64_t seq_length = 64;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        const int64_t batches_to_process = std::min( loader.numBatches(), 3LL );

        for ( int64_t i = 0; i < batches_to_process; ++i )
        {
            loader.nextBatch();
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
        }

        EXPECT_EQ( loader.currentBatch(), batches_to_process );

        loader.reset();

        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }

    TEST_F( StreamingDataLoaderTest, DataCorrectness_TargetIsInputShifted )
    {
        std::vector<TokenId> pattern = { 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
        createTokensFile( 5000, pattern );

        const int64_t batch_size = 2;
        const int64_t seq_length = 8;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        loader.nextBatch();
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

        verifySequentialTokens( loader );
    }

    TEST_F( StreamingDataLoaderTest, TensorShapes_Validation )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 8;
        const int64_t seq_length = 256;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        const auto& inputs = loader.inputs();
        const auto& targets = loader.targets();

        EXPECT_EQ( inputs.shape()[ 0 ], batch_size );
        EXPECT_EQ( inputs.shape()[ 1 ], seq_length );
        EXPECT_EQ( targets.shape()[ 0 ], batch_size );
        EXPECT_EQ( targets.shape()[ 1 ], seq_length );

        EXPECT_EQ( inputs.size(), batch_size * seq_length );
        EXPECT_EQ( targets.size(), batch_size * seq_length );
    }

    TEST_F( StreamingDataLoaderTest, CustomConfig_VerboseLogging )
    {
        createTokensFile( num_test_tokens_ );

        StreamingDataLoaderConfig config;
        config.verbose_logging = true;
        config.max_queue_size = 5;

        testing::internal::CaptureStdout();

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            4,
            128,
            false,
            Device::Cpu(),
            config );

        std::string output = testing::internal::GetCapturedStdout();

        EXPECT_FALSE( output.empty() );
        EXPECT_NE( output.find( "StreamingDataLoader initialized" ), std::string::npos );
    }

    TEST_F( StreamingDataLoaderTest, CustomConfig_TokenWindowSize )
    {
        createTokensFile( 100000 );

        StreamingDataLoaderConfig config;
        config.token_window_size = 10000;
        config.verbose_logging = false;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            4,
            128,
            false,
            Device::Cpu(),
            config );

        EXPECT_GT( loader.numBatches(), 0 );
    }

    TEST_F( StreamingDataLoaderTest, MultipleEpochs_TrainingMode )
    {
        createTokensFile( 5000 );

        const int64_t batch_size = 2;
        const int64_t seq_length = 32;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            true,
            Device::Cpu() );

        const int64_t total_batches = loader.numBatches();

        for ( int epoch = 0; epoch < 3; ++epoch )
        {
            loader.reset();
            EXPECT_EQ( loader.currentBatch(), 0 );

            for ( int64_t i = 0; i < total_batches; ++i )
            {
                loader.nextBatch();
                std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
            }

            EXPECT_EQ( loader.currentBatch(), total_batches );
        }
    }

    TEST_F( StreamingDataLoaderTest, SmallDataset_EdgeCase )
    {
        createTokensFile( 1000 );

        const int64_t batch_size = 2;
        const int64_t seq_length = 16;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        EXPECT_GT( loader.numBatches(), 0 );
        EXPECT_TRUE( loader.hasNext() );

        loader.nextBatch();
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    }

    TEST_F( StreamingDataLoaderTest, LargeSequenceLength )
    {
        createTokensFile( 50000 );

        const int64_t batch_size = 1;
        const int64_t seq_length = 2048;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        EXPECT_GT( loader.numBatches(), 0 );

        loader.nextBatch();
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

        const auto& inputs = loader.inputs();
        EXPECT_EQ( inputs.shape()[ 1 ], seq_length );
    }

    TEST_F( StreamingDataLoaderTest, Threading_StressTest )
    {
        createTokensFile( 20000 );

        const int64_t batch_size = 4;
        const int64_t seq_length = 128;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            true,
            Device::Cpu() );

        const int64_t batches_to_process = std::min( loader.numBatches(), 20LL );

        for ( int64_t i = 0; i < batches_to_process; ++i )
        {
            loader.nextBatch();
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );

            const auto& inputs = loader.inputs();
            EXPECT_GT( inputs.size(), 0 );
        }
    }

    TEST_F( StreamingDataLoaderTest, DataIntegrity_AcrossBatches )
    {
        std::vector<TokenId> pattern;
        for ( size_t i = 0; i < 1000; ++i )
        {
            pattern.push_back( static_cast<TokenId>( i ) );
        }
        createTokensFile( 10000, pattern );

        const int64_t batch_size = 2;
        const int64_t seq_length = 64;

        StreamingDataLoader<CpuMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cpu() );

        for ( int64_t batch_idx = 0; batch_idx < std::min( loader.numBatches(), 5LL ); ++batch_idx )
        {
            loader.nextBatch();
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );

            verifySequentialTokens( loader );
        }
    }

#ifdef MILA_CUDA_AVAILABLE
    TEST_F( StreamingDataLoaderTest, CudaPinnedMemory_BasicFunctionality )
    {
        createTokensFile( num_test_tokens_ );

        const int64_t batch_size = 4;
        const int64_t seq_length = 128;

        StreamingDataLoader<CudaPinnedMemoryResource> loader(
            temp_tokens_file_,
            batch_size,
            seq_length,
            false,
            Device::Cuda( 0 ) );

        EXPECT_GT( loader.numBatches(), 0 );
        EXPECT_EQ( loader.batchSize(), batch_size );
        EXPECT_EQ( loader.sequenceLength(), seq_length );

        const auto& inputs = loader.inputs();
        const auto& targets = loader.targets();

        EXPECT_EQ( inputs.shape()[ 0 ], batch_size );
        EXPECT_EQ( inputs.shape()[ 1 ], seq_length );
        EXPECT_EQ( targets.shape()[ 0 ], batch_size );
        EXPECT_EQ( targets.shape()[ 1 ], seq_length );
    }

    TEST_F( StreamingDataLoaderTest, CudaPinnedMemory_DeviceTypeMismatch )
    {
        createTokensFile( num_test_tokens_ );

        EXPECT_THROW(
            {
                StreamingDataLoader<CudaPinnedMemoryResource> loader(
                    temp_tokens_file_,
                    4,
                    128,
                    false,
                    Device::Cpu() );
            },
            std::runtime_error );
    }
#endif
}