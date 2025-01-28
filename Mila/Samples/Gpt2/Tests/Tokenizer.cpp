#include <gtest/gtest.h>


#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Dnn::Gpt2::Tests
{
    using namespace Mila::Dnn::Gpt2;

    class TokenizerTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Setup code if needed
        }

        void TearDown() override {
            // Cleanup code if needed
        }
    };

    TEST_F( TokenizerTests, Constructor_ValidFile_ShouldInitializeCorrectly ) {
        // Arrange
        std::string filename = "data/models/gpt2/gpt2_tokenizer.bin";

        // Act
        Tokenizer tokenizer( filename );

        // Assert
        EXPECT_EQ( tokenizer.get_eot_token(), 50256 );
    }

    TEST_F( TokenizerTests, Decode_ValidTokenId_ShouldReturnCorrectString ) {
        // Arrange
        std::string filename = "data/models/gpt2/gpt2_tokenizer.bin";
        Tokenizer tokenizer( filename );
        uint32_t token_id = 0; // Assuming 0 is a valid token ID in the file

        // Act
        const char* decoded_string = tokenizer.decode( token_id );

        // Assert
        EXPECT_STREQ( decoded_string, "!" );
    }

    TEST_F( TokenizerTests, Decode_InvalidTokenId_ShouldThrowException ) {
        // Arrange
        std::string filename = "data/models/gpt2/gpt2_tokenizer.bin";
        Tokenizer tokenizer( filename );
        uint32_t invalid_token_id = 99999; // Assuming 99999 is an invalid token ID

        // Act & Assert
        EXPECT_THROW( tokenizer.decode( invalid_token_id ), std::runtime_error );
    }

    TEST_F( TokenizerTests, Constructor_InvalidFile_ShouldThrowException ) {
        // Arrange
        std::string filename = "data/models/gpt2/gpt2_124M.bin";

        // Act & Assert
        EXPECT_THROW( Tokenizer tokenizer( filename ), std::runtime_error );
    }
}