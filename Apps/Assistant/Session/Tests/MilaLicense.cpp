/**
 * @file MilaLicense.cpp
 * @brief The licence the package carries is read with the digest an acceptance keeps, and a package without one says
 *        so (AssistantOnboarding.md 4.1).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

import Mila;
import Assistant.MilaLicense;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        class MilaLicenseTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_license_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
                std::filesystem::create_directories( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            std::filesystem::path path() const
            {
                return folder_ / "License.md";
            }

            void store( std::string_view text ) const
            {
                std::ofstream( path(), std::ios::binary ) << text;
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( MilaLicenseTest, TheTextIsReadWithTheDigestOfItsBytes )
    {
        store( "abc" );

        const std::expected<MilaLicense, std::string> license = MilaLicense::read( path() );

        ASSERT_TRUE( license.has_value() ) << license.error();
        EXPECT_EQ( license->text, "abc" );
        EXPECT_EQ( license->sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );
    }

    TEST_F( MilaLicenseTest, AnotherTextHasAnotherDigest )
    {
        store( "abc" );
        const std::string first = MilaLicense::read( path() )->sha256;

        store( "abd" );
        const std::string second = MilaLicense::read( path() )->sha256;

        EXPECT_NE( first, second );
    }

    TEST_F( MilaLicenseTest, AMissingLicenceIsSaid )
    {
        const std::expected<MilaLicense, std::string> license = MilaLicense::read( folder_ / "absent.md" );

        ASSERT_FALSE( license.has_value() );
        EXPECT_NE( license.error().find( "license" ), std::string::npos ) << license.error();
    }

    TEST_F( MilaLicenseTest, AnEmptyLicenceIsNoLicence )
    {
        store( "" );

        EXPECT_FALSE( MilaLicense::read( path() ).has_value() );
    }

    TEST_F( MilaLicenseTest, TheRepositorysLicenceIsTheOneShipped )
    {
        // The file the build copies beside the session; the test runs from the repository's root.
        const std::expected<MilaLicense, std::string> license = MilaLicense::read( "License.md" );

        ASSERT_TRUE( license.has_value() ) << license.error();
        EXPECT_NE( license->text.find( "AS IS" ), std::string::npos );
    }
}
