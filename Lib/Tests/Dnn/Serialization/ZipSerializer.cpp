#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <format>
#include <system_error>
#include <algorithm>

import Mila;

namespace Dnn::Serialization::Tests
{
    using namespace std::chrono_literals;
	using namespace Mila::Dnn::Serialization;

    static std::filesystem::path makeTempZipPath()
    {
        auto tmp = std::filesystem::temp_directory_path();
        auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        return tmp / std::format( "mila_test_zip_{}.mila", ts );
    }

    TEST( ZipSerializerTest, WriteReadRoundTrip )
    {
        auto path = makeTempZipPath();

        // Ensure no prior file
        std::error_code ec;
        std::filesystem::remove( path, ec );

        Mila::Dnn::Serialization::ZipSerializer writer;

        std::string file_data = "hello-mila";
        std::string other_data = "other-bytes";
        std::string version = "1.2.3";

        // Open for write
        ASSERT_TRUE( writer.open( path.string(), OpenMode::Write ) );

        // Add regular file and metadata
        EXPECT_TRUE( writer.addData( "files/data.bin", file_data.data(), file_data.size() ) );
        EXPECT_TRUE( writer.addData( "files/other.bin", other_data.data(), other_data.size() ) );
        EXPECT_TRUE( writer.addMetadata( "version", version ) );

        // Close writer
        EXPECT_TRUE( writer.close() );

        // Re-open for read
        ZipSerializer reader;
        ASSERT_TRUE( reader.open( path.string(), OpenMode::Read ) );

        // Listing files should include our paths
        auto files = reader.listFiles();

        EXPECT_NE( std::find( files.begin(), files.end(), "files/data.bin" ), files.end() );
        EXPECT_NE( std::find( files.begin(), files.end(), "files/other.bin" ), files.end() );
        EXPECT_NE( std::find( files.begin(), files.end(), "metadata/version" ), files.end() );

        // File sizes
        EXPECT_EQ( reader.getFileSize( "files/data.bin" ), file_data.size() );
        EXPECT_EQ( reader.getFileSize( "files/other.bin" ), other_data.size() );
        EXPECT_EQ( reader.getFileSize( "metadata/version" ), version.size() );

        // Extract data
        std::vector<char> buf( reader.getFileSize( "files/data.bin" ) );
        EXPECT_EQ( reader.extractData( "files/data.bin", buf.data(), buf.size() ), buf.size() );
        EXPECT_EQ( std::string( buf.data(), buf.size() ), file_data );

        std::vector<char> buf2( reader.getFileSize( "files/other.bin" ) );
        EXPECT_EQ( reader.extractData( "files/other.bin", buf2.data(), buf2.size() ), buf2.size() );
        EXPECT_EQ( std::string( buf2.data(), buf2.size() ), other_data );

        // Metadata read
        EXPECT_EQ( reader.getMetadata( "version" ), version );

        // hasFile positive/negative
        EXPECT_TRUE( reader.hasFile( "files/data.bin" ) );
        EXPECT_FALSE( reader.hasFile( "does/not/exist" ) );

        // addData should fail when opened for read
        EXPECT_FALSE( reader.addData( "should/fail", "x", 1 ) );

        EXPECT_TRUE( reader.close() );

        // Clean up
        std::filesystem::remove( path, ec );
        EXPECT_FALSE( std::filesystem::exists( path ) );
    }

    TEST( ZipSerializerTest, AddDataWithoutOpenReturnsFalse )
    {
        ZipSerializer s;
        const char payload[] = "x";
        EXPECT_FALSE( s.addData( "a/b", payload, sizeof( payload ) ) );
    }

    TEST( ZipSerializerTest, ExtractNonexistentFileReturnsZero )
    {
        auto path = makeTempZipPath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        ZipSerializer writer;
        ASSERT_TRUE( writer.open( path.string(), OpenMode::Write ) );
        EXPECT_TRUE( writer.addData( "only/one", "data", 4 ) );
        EXPECT_TRUE( writer.close() );

        ZipSerializer reader;
        ASSERT_TRUE( reader.open( path.string(), OpenMode::Read ) );

        // Non-existent file
        std::vector<char> buf( 16 );
        EXPECT_EQ( reader.extractData( "no/such/file", buf.data(), buf.size() ), 0u );
        EXPECT_EQ( reader.getFileSize( "no/such/file" ), 0u );
        EXPECT_FALSE( reader.hasFile( "no/such/file" ) );

        EXPECT_TRUE( reader.close() );

        std::filesystem::remove( path, ec );
    }
}