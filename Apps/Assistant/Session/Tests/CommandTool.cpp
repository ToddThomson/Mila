/**
 * @file CommandTool.cpp
 * @brief The command tool's script, run by Windows PowerShell 5.1: a failed program stops the command with its exit
 *        code, and nothing else a command does is changed (ContainedPrograms.md section 5).
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>

import Assistant.CommandTool;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        struct Ran
        {
            int exit_code = -1;
            std::string output;
            std::string errors;
        };

        class CommandToolScript : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
#ifndef _WIN32
                GTEST_SKIP() << "the command tool runs Windows PowerShell";
#endif
                static int counter = 0;
                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_command_tool_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
                std::filesystem::create_directories( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            /// The command run as the tool runs it, outside the container: the same script, the same PowerShell.
            Ran run( const std::string& command ) const
            {
                const auto script = folder_ / "command.ps1";
                const auto output = folder_ / "output.txt";
                const auto errors = folder_ / "errors.txt";

                std::ofstream( script, std::ios::binary ) << CommandTool::script( command );

                // cmd /c drops the first and last quote of a line that starts with one, hence the outer pair.
                const std::string line = std::format( "\"powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass "
                    "-File \"{}\" > \"{}\" 2> \"{}\"\"", script.string(), output.string(), errors.string() );

                Ran ran;
                ran.exit_code = std::system( line.c_str() );
                ran.output = contents( output );
                ran.errors = contents( errors );

                return ran;
            }

            static std::string contents( const std::filesystem::path& path )
            {
                std::ifstream file( path, std::ios::binary );
                std::ostringstream text;
                text << file.rdbuf();

                return text.str();
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( CommandToolScript, AFailedProgramStopsTheCommandWithItsExitCode )
    {
        const Ran ran = run( "cmd /c exit 3; Write-Output after" );

        EXPECT_EQ( ran.exit_code, 3 );
        EXPECT_EQ( ran.output.find( "after" ), std::string::npos ) << ran.output;
        EXPECT_NE( ran.errors.find( "Stopped: `cmd /c exit 3` exited with code 3" ), std::string::npos ) << ran.errors;
    }

    TEST_F( CommandToolScript, AProgramFailingInsideABlockStopsTheCommand )
    {
        const Ran ran = run( "if ( $true )\n{\n    cmd /c exit 4\n}\nWrite-Output after" );

        EXPECT_EQ( ran.exit_code, 4 );
        EXPECT_EQ( ran.output.find( "after" ), std::string::npos ) << ran.output;
    }

    TEST_F( CommandToolScript, StatementsShareTheirVariablesAndFunctions )
    {
        const Ran ran = run( "$greeting = 'hello'; function Get-Name { 'Mila' }; Write-Output \"$greeting $(Get-Name)\"" );

        EXPECT_EQ( ran.exit_code, 0 ) << ran.errors;
        EXPECT_NE( ran.output.find( "hello Mila" ), std::string::npos ) << ran.output;
    }

    TEST_F( CommandToolScript, AProgramWritingToStandardErrorThatExitsZeroSucceeds )
    {
        const Ran ran = run( "cmd /c \"echo banner 1>&2\"; Write-Output after" );

        EXPECT_EQ( ran.exit_code, 0 ) << ran.errors;
        EXPECT_NE( ran.output.find( "after" ), std::string::npos ) << ran.output;
    }

    TEST_F( CommandToolScript, ASilencedCmdletErrorDoesNotStopTheCommand )
    {
        const Ran ran = run( "Remove-Item no-such-file -ErrorAction SilentlyContinue; Write-Output after" );

        EXPECT_EQ( ran.exit_code, 0 ) << ran.errors;
        EXPECT_NE( ran.output.find( "after" ), std::string::npos ) << ran.output;
    }

    TEST_F( CommandToolScript, ALastCmdletThatFailsReadsAsFailed )
    {
        const Ran ran = run( "Write-Output before; Get-Item no-such-file" );

        EXPECT_EQ( ran.exit_code, 1 );
        EXPECT_NE( ran.output.find( "before" ), std::string::npos ) << ran.output;
    }

    TEST_F( CommandToolScript, AnExitEndsTheCommandWithItsCode )
    {
        EXPECT_EQ( run( "Write-Output before; exit 7; Write-Output after" ).exit_code, 7 );
    }
}
