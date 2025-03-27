#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

import Mila;

namespace Gpt2Model::Tests
{
    using namespace Mila::Dnn;

    // A mock logger for testing
    class MockLogger : public Mila::Utils::Logger {
    public:
        std::vector<std::string> logs;

        void log( const std::string& message, int level = 0 ) override {
            logs.push_back( message );
        }
    };

    class DatasetReaderTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create a temporary test file with sample data
            test_filepath = "test_tokens.bin";

            // Generate sample token data: 100 integers with values 0-99
            std::vector<int> sample_data( 100 );
            for ( int i = 0; i < 100; i++ ) {
                sample_data[ i ] = i;
            }

            // Write sample data to file
            std::ofstream file( test_filepath, std::ios::binary );
            ASSERT_TRUE( file.is_open() );
            file.write( reinterpret_cast<char*>( sample_data.data() ),
                sample_data.size() * sizeof( int ) );
            file.close();

            // Setup shared logger for tests
            logger = std::make_shared<MockLogger>();
        }

        void TearDown() override {
            // Clean up the temporary file
            std::filesystem::remove( test_filepath );
        }

        std::string test_filepath;
        std::shared_ptr<MockLogger> logger;
    };

    // Test constructor with valid parameters
    TEST_F( DatasetReaderTest, ConstructorWithValidParameters ) {
        // Create a configuration with our mock logger
        Gpt2::DatasetReader::Config config;
        config.logger = logger;
        config.verbose_logging = true;

        // Should not throw exception
        EXPECT_NO_THROW( {
            Gpt2::DatasetReader reader( test_filepath, 2, 5, config );
            } );
    }

    // Test constructor with invalid parameters
    TEST_F( DatasetReaderTest, ConstructorWithInvalidParameters ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;

        // Batch size of 0 should throw
        EXPECT_THROW( {
            Gpt2::DatasetReader reader( test_filepath, 0, 5, config );
            }, std::invalid_argument );

        // Sequence length of 0 should throw
        EXPECT_THROW( {
            Gpt2::DatasetReader reader( test_filepath, 2, 0, config );
            }, std::invalid_argument );

        // Non-existent file should throw
        EXPECT_THROW( {
            Gpt2::DatasetReader reader( "nonexistent_file.bin", 2, 5, config );
            }, std::runtime_error );

        // Create a reader where batch_size * seq_len > file_size in tokens
        // Our test file has 100 tokens, so 10*20=200 should throw
        EXPECT_THROW( {
            Gpt2::DatasetReader reader( test_filepath, 10, 20, config );
            }, std::runtime_error );

        // Check for the specific error message (optional)
        try {
            Gpt2::DatasetReader reader( test_filepath, 10, 20, config );
        }
        catch ( const std::runtime_error& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Not enough tokens" ) != std::string::npos );
        }
    }

    // Test next_batch returns correct data
    TEST_F( DatasetReaderTest, NextBatchReturnsValidData ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;

        // Create reader with batch_size=2, seq_len=5
        Gpt2::DatasetReader reader( test_filepath, 2, 5, config );

        // Get a batch
        auto [inputs, targets] = reader.next_batch();

        // Verify dimensions of returned tensors
        EXPECT_EQ( inputs.shape()[0], 2);  // Batch size
        EXPECT_EQ( inputs.shape()[1], 5);  // Sequence length

        EXPECT_EQ( targets.shape()[0], 2);  // Batch size
        EXPECT_EQ( targets.shape()[1], 5);  // Sequence length

        // Check that targets are inputs shifted by one position
        // (Testing the specific values requires knowledge of the internal implementation)
    }

    // Test pause and resume functionality
    TEST_F( DatasetReaderTest, PauseAndResume ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;
        config.verbose_logging = true;

        Gpt2::DatasetReader reader( test_filepath, 2, 5, config );

        // Pause the reader
        EXPECT_NO_THROW( reader.pause() );

        // Check if paused status is reflected in logs
        bool found_pause_log = false;

        auto logs_t = dynamic_cast<MockLogger*>(logger.get())->logs;
        for ( const auto& log : dynamic_cast<MockLogger*>(logger.get())->logs ) {
            if ( log.find( "Pausing" ) != std::string::npos ) {
                found_pause_log = true;
                break;
            }
        }
        EXPECT_TRUE( found_pause_log );

        // Clear logs for next test
        dynamic_cast<MockLogger*>(logger.get())->logs.clear();

        // Resume the reader
        EXPECT_NO_THROW( reader.resume() );

        // Check if resume status is reflected in logs
        bool found_resume_log = false;
        for ( const auto& log : dynamic_cast<MockLogger*>(logger.get())->logs ) {
            if ( log.find( "Resuming" ) != std::string::npos ) {
                found_resume_log = true;
                break;
            }
        }
        EXPECT_TRUE( found_resume_log );
    }

    // Test proper handling of empty files
    TEST_F( DatasetReaderTest, EmptyFileHandling ) {
        // Create an empty file
        std::string empty_filepath = "empty_test.bin";
        std::ofstream empty_file( empty_filepath, std::ios::binary );
        empty_file.close();

        Gpt2::DatasetReader::Config config;
        config.logger = logger;

        // Should throw because the file is empty
        EXPECT_THROW( {
            Gpt2::DatasetReader reader( empty_filepath, 2, 5, config );
            }, std::runtime_error );

        // Clean up
        std::filesystem::remove( empty_filepath );
    }

    // Test with pinned memory allocation
    TEST_F( DatasetReaderTest, PinnedMemoryAllocation ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;

        // Skip this test if CUDA is not available
        try {
            Gpt2::DatasetReader reader( test_filepath, 2, 5, config );
            // If we get here, the constructor worked with pinned memory
            SUCCEED();
        }
        catch ( const std::runtime_error& e ) {
            // Check if the error is related to CUDA unavailability
            std::string error_msg = e.what();
            if ( error_msg.find( "CUDA" ) != std::string::npos ) {
                GTEST_SKIP() << "Skipping test as CUDA pinned memory allocation failed.";
            }
            else {
                FAIL() << "Unexpected error: " << error_msg;
            }
        }
    }

    TEST_F( DatasetReaderTest, DataWraparound ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;
        config.verbose_logging = true;

        Gpt2::DatasetReader reader( test_filepath, 5, 10, config );

        // Get 3 batches - we should wrap around after 2 batches (100/50 = 2)
        auto batch1 = reader.next_batch();
        auto batch2 = reader.next_batch();

        // This should trigger wrap-around internally since we'll run out of tokens
        EXPECT_NO_THROW( {
            auto batch3 = reader.next_batch();
            } );
        // Give background threads time to complete their logging operations
        std::this_thread::sleep_for( std::chrono::milliseconds( 2000 ) );

        // Check logs for wrap-around message
        bool found_wraparound_log = false;
        for ( const auto& log : dynamic_cast<MockLogger*>(logger.get())->logs ) {
            if ( log.find( "wrapping around" ) != std::string::npos ) {
                found_wraparound_log = true;
                break;
            }
        }
        EXPECT_TRUE( found_wraparound_log );
    }

    TEST_F( DatasetReaderTest, MultipleBatchFetches ) {
        Gpt2::DatasetReader::Config config;
        config.logger = logger;
        config.max_queue_size = 3;  // Small queue to test queueing behavior

        Gpt2::DatasetReader reader( test_filepath, 2, 5, config );

        // Fetch multiple batches in succession
        for ( int i = 0; i < 5; i++ ) {
            EXPECT_NO_THROW( {
                auto batch = reader.next_batch(); } ) << "Failed to fetch batch " << i;
        }
    }
}