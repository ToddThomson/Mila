/**
 * @file GrantPath.cpp
 * @brief The one form of path a grant accepts: each path refused beside the same kind of path, accepted
 *        (ToolCalling.md 3.4).
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    using Names = std::vector<std::string>;

    TEST( GrantPath, AStepUpOutOfAFolderIsRefused )
    {
        EXPECT_THROW( grantPathNames( "../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "src/../../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "D:\\project\\..\\secret.txt" ), std::invalid_argument );
    }

    TEST( GrantPath, ANameWithDotsInItIsAccepted )
    {
        EXPECT_EQ( grantPathNames( "a..b.txt" ), ( Names{ "a..b.txt" } ) );
        EXPECT_EQ( grantPathNames( "./src/./main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( grantPathNames( "." ), Names{} ) << "the folder itself";
    }

    TEST( GrantPath, AColonAfterTheDriveIsRefused )
    {
        // An alternate data stream, or a drive letter where a name belongs.
        EXPECT_THROW( grantPathNames( "notes.txt:hidden" ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "D:\\project\\notes.txt:hidden" ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "src\\C:\\Windows" ), std::invalid_argument );
    }

    TEST( GrantPath, ADriveLetterPathIsAccepted )
    {
        EXPECT_EQ( grantPathNames( "D:\\project\\notes.txt" ), ( Names{ "D:", "project", "notes.txt" } ) );
        EXPECT_EQ( grantPathNames( "d:/project/notes.txt" ), ( Names{ "D:", "project", "notes.txt" } ) );
        EXPECT_EQ( grantPathNames( "D:\\" ), ( Names{ "D:" } ) );
    }

    TEST( GrantPath, ADeviceNameIsRefusedWithOrWithoutAnExtension )
    {
        for ( const char* path : { "NUL", "nul", "nul.txt", "src/CON", "aux.tar.gz", "COM1", "lpt9.log", "COM\xC2\xB9",
                  "LPT\xC2\xB3.txt", "NUL .txt", "CONIN$", "conout$" } )
        {
            EXPECT_THROW( grantPathNames( path ), std::invalid_argument ) << path;
        }
    }

    TEST( GrantPath, ANameThatBeginsLikeADeviceIsAccepted )
    {
        for ( const char* path : { "CONFIG.txt", "nullable.h", "auxiliary/notes.md", "COM10", "COM0", "LPT", "console.log" } )
        {
            EXPECT_NO_THROW( grantPathNames( path ) ) << path;
        }
    }

    TEST( GrantPath, ANameEndingInASpaceOrPeriodIsRefused )
    {
        EXPECT_THROW( grantPathNames( "notes.txt." ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "notes.txt " ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "src /main.cpp" ), std::invalid_argument );
    }

    TEST( GrantPath, ANameBeginningWithAPeriodIsAccepted )
    {
        EXPECT_EQ( grantPathNames( ".gitignore" ), ( Names{ ".gitignore" } ) );
        EXPECT_EQ( grantPathNames( ".github/workflows/ci.yml" ), ( Names{ ".github", "workflows", "ci.yml" } ) );
    }

    TEST( GrantPath, ANetworkDevicePrefixedOrRootlessPathIsRefused )
    {
        for ( const char* path : { "\\\\server\\share\\notes.txt", "//server/share", "\\\\?\\D:\\project\\notes.txt",
                  "\\\\.\\PhysicalDrive0", "\\\\?\\GLOBALROOT\\Device", "\\Windows\\win.ini", "/etc/passwd", "C:notes.txt", "C:" } )
        {
            EXPECT_THROW( grantPathNames( path ), std::invalid_argument ) << path;
        }
    }

    TEST( GrantPath, ARelativePathIsAccepted )
    {
        EXPECT_EQ( grantPathNames( "src\\main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( grantPathNames( "src/main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( grantPathNames( "src//main.cpp/" ), ( Names{ "src", "main.cpp" } ) );
    }

    TEST( GrantPath, AReservedOrControlCharacterIsRefused )
    {
        for ( const char* path : { "a<b.txt", "a>b.txt", "a\"b.txt", "a|b.txt", "what?.txt", "*.cpp", "tab\there.txt" } )
        {
            EXPECT_THROW( grantPathNames( path ), std::invalid_argument ) << path;
        }

        EXPECT_THROW( grantPathNames( std::string( "nul\0hidden", 10 ) ), std::invalid_argument );
        EXPECT_THROW( grantPathNames( "" ), std::invalid_argument );
    }

    TEST( GrantPath, AUnicodeNameIsAccepted )
    {
        EXPECT_EQ( grantPathNames( "notizen/\xC3\xBC" "bersicht.md" ), ( Names{ "notizen", "\xC3\xBC" "bersicht.md" } ) );
        EXPECT_EQ( grantPathNames( "\xE6\x96\x87\xE6\xA1\xA3.txt" ), ( Names{ "\xE6\x96\x87\xE6\xA1\xA3.txt" } ) );
    }
}
