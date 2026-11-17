/**
 * @file GrantPath.cpp
 * @brief The one form of path a grant accepts, each platform's: each path refused beside the same kind of path,
 *        accepted (ToolCalling.md 3.4). Both forms are tested on every platform.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::Agents;

    using Names = std::vector<std::string>;

    namespace
    {
        Names windows( std::string_view path )
        {
            return grantPathNames( path, GrantPathForm::Windows );
        }

        Names posix( std::string_view path )
        {
            return grantPathNames( path, GrantPathForm::Posix );
        }
    }

    // ================================================================
    // Windows
    // ================================================================

    TEST( GrantPathWindows, AStepUpOutOfAFolderIsRefused )
    {
        EXPECT_THROW( windows( "../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( windows( "src/../../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( windows( "D:\\project\\..\\secret.txt" ), std::invalid_argument );
    }

    TEST( GrantPathWindows, ANameWithDotsInItIsAccepted )
    {
        EXPECT_EQ( windows( "a..b.txt" ), ( Names{ "a..b.txt" } ) );
        EXPECT_EQ( windows( "./src/./main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( windows( "." ), Names{} ) << "the folder itself";
    }

    TEST( GrantPathWindows, AColonAfterTheDriveIsRefused )
    {
        // An alternate data stream, or a drive letter where a name belongs.
        EXPECT_THROW( windows( "notes.txt:hidden" ), std::invalid_argument );
        EXPECT_THROW( windows( "D:\\project\\notes.txt:hidden" ), std::invalid_argument );
        EXPECT_THROW( windows( "src\\C:\\Windows" ), std::invalid_argument );
    }

    TEST( GrantPathWindows, ADriveLetterPathIsAccepted )
    {
        EXPECT_EQ( windows( "D:\\project\\notes.txt" ), ( Names{ "D:", "project", "notes.txt" } ) );
        EXPECT_EQ( windows( "d:/project/notes.txt" ), ( Names{ "D:", "project", "notes.txt" } ) );
        EXPECT_EQ( windows( "D:\\" ), ( Names{ "D:" } ) );
    }

    TEST( GrantPathWindows, ADeviceNameIsRefusedWithOrWithoutAnExtension )
    {
        for ( const char* path : { "NUL", "nul", "nul.txt", "src/CON", "aux.tar.gz", "COM1", "lpt9.log", "COM\xC2\xB9",
                  "LPT\xC2\xB3.txt", "NUL .txt", "CONIN$", "conout$" } )
        {
            EXPECT_THROW( windows( path ), std::invalid_argument ) << path;
        }
    }

    TEST( GrantPathWindows, ANameThatBeginsLikeADeviceIsAccepted )
    {
        for ( const char* path : { "CONFIG.txt", "nullable.h", "auxiliary/notes.md", "COM10", "COM0", "LPT", "console.log" } )
        {
            EXPECT_NO_THROW( windows( path ) ) << path;
        }
    }

    TEST( GrantPathWindows, ANameEndingInASpaceOrPeriodIsRefused )
    {
        EXPECT_THROW( windows( "notes.txt." ), std::invalid_argument );
        EXPECT_THROW( windows( "notes.txt " ), std::invalid_argument );
        EXPECT_THROW( windows( "src /main.cpp" ), std::invalid_argument );
    }

    TEST( GrantPathWindows, ANameBeginningWithAPeriodIsAccepted )
    {
        EXPECT_EQ( windows( ".gitignore" ), ( Names{ ".gitignore" } ) );
        EXPECT_EQ( windows( ".github/workflows/ci.yml" ), ( Names{ ".github", "workflows", "ci.yml" } ) );
    }

    TEST( GrantPathWindows, ANetworkDevicePrefixedOrRootlessPathIsRefused )
    {
        for ( const char* path : { "\\\\server\\share\\notes.txt", "//server/share", "\\\\?\\D:\\project\\notes.txt",
                  "\\\\.\\PhysicalDrive0", "\\\\?\\GLOBALROOT\\Device", "\\Windows\\win.ini", "/etc/passwd", "C:notes.txt", "C:" } )
        {
            EXPECT_THROW( windows( path ), std::invalid_argument ) << path;
        }
    }

    TEST( GrantPathWindows, ARelativePathIsAccepted )
    {
        EXPECT_EQ( windows( "src\\main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( windows( "src/main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( windows( "src//main.cpp/" ), ( Names{ "src", "main.cpp" } ) );
    }

    TEST( GrantPathWindows, AReservedOrControlCharacterIsRefused )
    {
        for ( const char* path : { "a<b.txt", "a>b.txt", "a\"b.txt", "a|b.txt", "what?.txt", "*.cpp", "tab\there.txt" } )
        {
            EXPECT_THROW( windows( path ), std::invalid_argument ) << path;
        }

        EXPECT_THROW( windows( std::string( "nul\0hidden", 10 ) ), std::invalid_argument );
        EXPECT_THROW( windows( "" ), std::invalid_argument );
    }

    TEST( GrantPathWindows, AUnicodeNameIsAccepted )
    {
        EXPECT_EQ( windows( "notizen/\xC3\xBC" "bersicht.md" ), ( Names{ "notizen", "\xC3\xBC" "bersicht.md" } ) );
        EXPECT_EQ( windows( "\xE6\x96\x87\xE6\xA1\xA3.txt" ), ( Names{ "\xE6\x96\x87\xE6\xA1\xA3.txt" } ) );
    }

    // ================================================================
    // Posix
    // ================================================================

    TEST( GrantPathPosix, AStepUpOutOfAFolderIsRefused )
    {
        EXPECT_THROW( posix( "../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( posix( "src/../../secret.txt" ), std::invalid_argument );
        EXPECT_THROW( posix( "/Users/someone/project/../secret.txt" ), std::invalid_argument );
    }

    TEST( GrantPathPosix, ANameWithDotsInItIsAccepted )
    {
        EXPECT_EQ( posix( "a..b.txt" ), ( Names{ "a..b.txt" } ) );
        EXPECT_EQ( posix( "./src/./main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( posix( "." ), Names{} ) << "the folder itself";
    }

    TEST( GrantPathPosix, AResourceForkIsRefused )
    {
        for ( const char* path : { "notes.txt/..namedfork/rsrc", "/Users/someone/notes.txt/..namedfork/data", "..namedfork",
                  "notes.txt/..NAMEDFORK/rsrc" } )
        {
            EXPECT_THROW( posix( path ), std::invalid_argument ) << path;
        }
    }

    TEST( GrantPathPosix, ANameLikeTheForksIsAccepted )
    {
        EXPECT_EQ( posix( "..namedforks/rsrc" ), ( Names{ "..namedforks", "rsrc" } ) );
        EXPECT_EQ( posix( "notes/.namedfork" ), ( Names{ "notes", ".namedfork" } ) );
    }

    TEST( GrantPathPosix, APathFromTheRootIsAccepted )
    {
        EXPECT_EQ( posix( "/Users/someone/notes.txt" ), ( Names{ "/", "Users", "someone", "notes.txt" } ) );
        EXPECT_EQ( posix( "//private///tmp/" ), ( Names{ "/", "private", "tmp" } ) );
        EXPECT_EQ( posix( "/" ), ( Names{ "/" } ) );
    }

    TEST( GrantPathPosix, AControlCharacterIsRefused )
    {
        EXPECT_THROW( posix( "tab\there.txt" ), std::invalid_argument );
        EXPECT_THROW( posix( "line\nbreak.txt" ), std::invalid_argument );
        EXPECT_THROW( posix( std::string( "notes\0hidden", 12 ) ), std::invalid_argument );
        EXPECT_THROW( posix( "" ), std::invalid_argument );
    }

    TEST( GrantPathPosix, WhatWindowsReservesIsAnOrdinaryName )
    {
        // A device name, a colon, a backslash, a trailing space or period: each a character of a name on macOS.
        EXPECT_EQ( posix( "NUL" ), ( Names{ "NUL" } ) );
        EXPECT_EQ( posix( "notes.txt:hidden" ), ( Names{ "notes.txt:hidden" } ) );
        EXPECT_EQ( posix( "D:/project" ), ( Names{ "D:", "project" } ) ) << "a folder named D: in the project, not a drive";
        EXPECT_EQ( posix( "src\\main.cpp" ), ( Names{ "src\\main.cpp" } ) );
        EXPECT_EQ( posix( "notes.txt. " ), ( Names{ "notes.txt. " } ) );
        EXPECT_EQ( posix( "what?.txt" ), ( Names{ "what?.txt" } ) );
    }

    TEST( GrantPathPosix, ARelativePathIsAccepted )
    {
        EXPECT_EQ( posix( "src/main.cpp" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( posix( "src//main.cpp/" ), ( Names{ "src", "main.cpp" } ) );
        EXPECT_EQ( posix( ".github/workflows/ci.yml" ), ( Names{ ".github", "workflows", "ci.yml" } ) );
    }

    TEST( GrantPathPosix, AUnicodeNameIsAccepted )
    {
        EXPECT_EQ( posix( "notizen/\xC3\xBC" "bersicht.md" ), ( Names{ "notizen", "\xC3\xBC" "bersicht.md" } ) );
        EXPECT_EQ( posix( "\xE6\x96\x87\xE6\xA1\xA3.txt" ), ( Names{ "\xE6\x96\x87\xE6\xA1\xA3.txt" } ) );
    }
}
