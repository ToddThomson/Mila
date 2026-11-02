/**
 * @file FileTools.cpp
 * @brief The six file tools, run as an agent runs them, through a grant: what each returns, what each refuses, and
 *        that none reaches past its grant (ToolCalling.md section 5). Windows only, where a folder can be granted.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import Mila;
import Milo;

namespace Mila::Tests::AI
{
    using namespace Milo;

    namespace
    {
        void write( const std::filesystem::path& file, std::string_view content )
        {
            std::filesystem::create_directories( file.parent_path() );
            std::ofstream stream( file, std::ios::binary | std::ios::trunc );
            stream.write( content.data(), static_cast<std::streamsize>( content.size() ) );
        }

        std::string contentOf( const std::filesystem::path& file )
        {
            std::ifstream stream( file, std::ios::binary );

            return std::string( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() );
        }

        Tool toolNamed( std::string_view name )
        {
            for ( Tool& tool : FileTools::tools() )
            {
                if ( tool.name == name )
                {
                    return std::move( tool );
                }
            }

            throw std::invalid_argument( std::format( "no file tool {}", name ) );
        }

        std::string jsonText( const std::filesystem::path& path )
        {
            const std::u8string utf8 = path.generic_u8string();

            return std::string( utf8.begin(), utf8.end() );
        }

        class FileToolsTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                std::random_device seed;
                root_ = std::filesystem::temp_directory_path() / std::format( "MilaFileTools-{:08x}", seed() );
                project_ = root_ / "project";
                outside_ = root_ / "outside";

                write( project_ / "notes.txt", "one\ntwo\nthree\n" );
                write( project_ / "src" / "main.cpp", "int main() { return 41; }\n" );
                write( project_ / "src" / "deep" / "other.cpp", "// nothing here\n" );
                write( project_ / ".git" / "config.cpp", "kept out\n" );
                write( outside_ / "secret.txt", "secret\n" );
            }

            void TearDown() override
            {
                std::error_code error;
                std::filesystem::remove_all( root_, error );
            }

            std::string run( std::string_view name, const std::string& arguments, FolderAccess access = FolderAccess::ReadWrite ) const
            {
                return toolNamed( name ).run( arguments, ToolGrant( { GrantFolder{ project_, access } }, false ), {} );
            }

            void expectRefused( std::string_view name, const std::string& arguments, std::string_view reason,
                FolderAccess access = FolderAccess::ReadWrite ) const
            {
                try
                {
                    const std::string result = run( name, arguments, access );
                    ADD_FAILURE() << name << " returned, where a refusal naming '" << reason << "' was expected: " << result;
                }
                catch ( const std::exception& error )
                {
                    EXPECT_NE( std::string_view( error.what() ).find( reason ), std::string_view::npos ) << error.what();
                }
            }

            std::filesystem::path root_;
            std::filesystem::path project_;
            std::filesystem::path outside_;
        };
    }

    TEST( FileToolsList, EachCarriesWhatTheModelAndThePersonAreGivenAndWhatItNeeds )
    {
        const std::vector<Tool> tools = FileTools::tools();

        ASSERT_EQ( tools.size(), 6u );

        const std::vector<std::string_view> names{ "list_folder", "read_file", "find_files", "search_text", "write_file", "edit_file" };

        for ( std::size_t index = 0; index < tools.size(); ++index )
        {
            const Tool& tool = tools[ index ];
            const bool writes = tool.name == "write_file" || tool.name == "edit_file";

            EXPECT_EQ( tool.name, names[ index ] );
            EXPECT_FALSE( tool.description.empty() ) << tool.name;
            EXPECT_FALSE( tool.summary.empty() ) << tool.name;
            EXPECT_EQ( tool.source, "Milo" ) << tool.name;
            EXPECT_TRUE( tool.parameters.starts_with( R"({"type":"object","properties":{)" ) ) << tool.parameters;
            EXPECT_TRUE( tool.needs.reads_files ) << tool.name;
            EXPECT_EQ( tool.needs.writes_files, writes ) << tool.name;
            EXPECT_FALSE( tool.needs.uses_network ) << tool.name;
            EXPECT_FALSE( tool.needs.runs_programs ) << tool.name;
        }
    }

    TEST( FileToolsList, ASchemaKeepsItsKeysInTheOrderWrittenAndTheGateChecksACallAgainstIt )
    {
        const Tool edit = toolNamed( "edit_file" );

        EXPECT_NE( edit.parameters.find( R"("required":["path","old_text","new_text"])" ), std::string::npos ) << edit.parameters;

        const ArgumentSchema schema( edit.parameters );

        EXPECT_FALSE( schema.check( R"({"path":"a","old_text":"b","new_text":"c"})" ).has_value() );
        EXPECT_TRUE( schema.check( R"({"path":"a"})" ).has_value() );

        for ( const Tool& tool : FileTools::tools() )
        {
            EXPECT_NO_THROW( ArgumentSchema{ tool.parameters } ) << tool.name;
        }

        EXPECT_NO_THROW( ArgumentSchema{ WebPageReader::tool().parameters } );
    }

    TEST_F( FileToolsTest, ReadFileGivesTheLinesAskedForAndSaysWhereToReadOn )
    {
        const std::string whole = run( "read_file", R"({"path":"notes.txt"})" );

        EXPECT_NE( whole.find( "lines 1 to 3 of 3\none\ntwo\nthree\n" ), std::string::npos ) << whole;

        const std::string part = run( "read_file", R"({"path":"notes.txt","start_line":2,"line_count":1})" );

        EXPECT_NE( part.find( "lines 2 to 2 of 3\ntwo\n" ), std::string::npos ) << part;
        EXPECT_NE( part.find( "read on with start_line 3" ), std::string::npos ) << part;
    }

    TEST_F( FileToolsTest, ReadFileSaysWhatItCannotRead )
    {
        expectRefused( "read_file", R"({"path":"notes.txt","start_line":9})", "has 3 lines" );
        expectRefused( "read_file", R"({"path":"absent.txt"})", "absent.txt" );

        write( project_ / "image.bin", std::string( "ab\0cd", 5 ) );
        EXPECT_NE( run( "read_file", R"({"path":"image.bin"})" ).find( "binary file" ), std::string::npos );

        write( project_ / "large.txt", std::string( 8u * 1024u * 1024u + 1u, 'x' ) );
        expectRefused( "read_file", R"({"path":"large.txt"})", "more than can be read at once" );
    }

    TEST_F( FileToolsTest, ListFolderShowsFoldersThenFiles )
    {
        const std::string listed = run( "list_folder", R"({"path":"."})" );
        const auto folder = listed.find( "src/" );
        const auto file = listed.find( "notes.txt" );

        ASSERT_NE( folder, std::string::npos ) << listed;
        ASSERT_NE( file, std::string::npos ) << listed;
        EXPECT_LT( folder, file );
    }

    TEST_F( FileToolsTest, FindFilesMatchesANameAnywhereOrAPathAndNeverEntersGit )
    {
        const std::string anywhere = run( "find_files", R"({"pattern":"*.cpp"})" );

        EXPECT_NE( anywhere.find( "src/main.cpp" ), std::string::npos ) << anywhere;
        EXPECT_NE( anywhere.find( "src/deep/other.cpp" ), std::string::npos ) << anywhere;
        EXPECT_EQ( anywhere.find( "config.cpp" ), std::string::npos ) << anywhere;

        const std::string one_folder = run( "find_files", R"({"pattern":"src/*.cpp"})" );

        EXPECT_NE( one_folder.find( "src/main.cpp" ), std::string::npos ) << one_folder;
        EXPECT_EQ( one_folder.find( "other.cpp" ), std::string::npos ) << one_folder;

        const std::string deep = run( "find_files", R"({"pattern":"src/**/other.cpp"})" );

        EXPECT_NE( deep.find( "src/deep/other.cpp" ), std::string::npos ) << deep;
        EXPECT_NE( run( "find_files", R"({"pattern":"*.none"})" ).find( "No file matches" ), std::string::npos );
    }

    TEST_F( FileToolsTest, SearchTextGivesEachLineItAppearsOnAndNeverEntersGit )
    {
        const std::string found = run( "search_text", R"({"text":"return 41"})" );

        EXPECT_NE( found.find( "src/main.cpp:1: int main() { return 41; }" ), std::string::npos ) << found;
        EXPECT_NE( run( "search_text", R"({"text":"two","path":"notes.txt"})" ).find( "notes.txt:2: two" ), std::string::npos );
        EXPECT_NE( run( "search_text", R"({"text":"absent words"})" ).find( "was not found" ), std::string::npos );
        EXPECT_NE( run( "search_text", R"({"text":"kept out"})" ).find( "was not found" ), std::string::npos );

        expectRefused( "search_text", R"({"text":""})", "is empty" );
        expectRefused( "search_text", R"({"text":"x","path":"absent"})", "no file or folder" );
    }

    TEST_F( FileToolsTest, WriteFileCreatesWithItsFoldersOrReplacesWhole )
    {
        EXPECT_NE( run( "write_file", R"({"path":"made/new.txt","content":"hello"})" ).find( "Created" ), std::string::npos );
        EXPECT_EQ( contentOf( project_ / "made" / "new.txt" ), "hello" );

        EXPECT_NE( run( "write_file", R"({"path":"notes.txt","content":"replaced"})" ).find( "Replaced" ), std::string::npos );
        EXPECT_EQ( contentOf( project_ / "notes.txt" ), "replaced" );

        expectRefused( "write_file", R"({"path":"src","content":"x"})", "is a folder" );
    }

    TEST_F( FileToolsTest, EditFileChangesTheOnePlaceAndShowsItsLinesAsTheyNowRead )
    {
        const std::string result = run( "edit_file", R"({"path":"notes.txt","old_text":"two","new_text":"TWO"})" );

        EXPECT_EQ( contentOf( project_ / "notes.txt" ), "one\nTWO\nthree\n" );
        EXPECT_NE( result.find( "Lines 1 to 3 now read:\none\nTWO\nthree\n" ), std::string::npos ) << result;
    }

    TEST_F( FileToolsTest, EditFileRefusesAPassageThatIsNotExactlyOnePlaceAndChangesNothing )
    {
        write( project_ / "twice.txt", "same\nsame\n" );

        expectRefused( "edit_file", R"({"path":"twice.txt","old_text":"same","new_text":"other"})", "appears 2 times" );
        expectRefused( "edit_file", R"({"path":"notes.txt","old_text":"absent","new_text":"other"})", "does not appear" );
        expectRefused( "edit_file", R"({"path":"notes.txt","old_text":"two","new_text":"two"})", "nothing to change" );
        expectRefused( "edit_file", R"({"path":"notes.txt","old_text":"","new_text":"x"})", "old_text is empty" );

        EXPECT_EQ( contentOf( project_ / "twice.txt" ), "same\nsame\n" );
        EXPECT_EQ( contentOf( project_ / "notes.txt" ), "one\ntwo\nthree\n" );
    }

    TEST_F( FileToolsTest, EditFileKeepsAFilesOwnLineEndings )
    {
        write( project_ / "windows.txt", "one\r\ntwo\r\nthree\r\n" );

        (void)run( "edit_file", R"({"path":"windows.txt","old_text":"one\ntwo","new_text":"ONE\nTWO"})" );

        EXPECT_EQ( contentOf( project_ / "windows.txt" ), "ONE\r\nTWO\r\nthree\r\n" );
    }

    TEST_F( FileToolsTest, NothingReachesPastTheGrant )
    {
        const std::string secret = jsonText( outside_ / "secret.txt" );

        expectRefused( "read_file", std::format( R"({{"path":"{}"}})", secret ), "lies outside" );
        expectRefused( "edit_file", std::format( R"({{"path":"{}","old_text":"secret","new_text":"x"}})", secret ), "lies outside" );
        expectRefused( "write_file", std::format( R"({{"path":"{}","content":"x"}})", secret ), "lies outside" );
        expectRefused( "read_file", R"({"path":"../outside/secret.txt"})", "leads up out" );
        expectRefused( "list_folder", std::format( R"({{"path":"{}"}})", jsonText( outside_ ) ), "lies outside" );

        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret\n" );
    }

    TEST_F( FileToolsTest, AReadOnlyFolderIsReadAndNotWritten )
    {
        EXPECT_NE( run( "read_file", R"({"path":"notes.txt"})", FolderAccess::Read ).find( "one" ), std::string::npos );

        expectRefused( "write_file", R"({"path":"notes.txt","content":"x"})", "only to read", FolderAccess::Read );
        expectRefused( "edit_file", R"({"path":"notes.txt","old_text":"one","new_text":"x"})", "only to read", FolderAccess::Read );

        EXPECT_EQ( contentOf( project_ / "notes.txt" ), "one\ntwo\nthree\n" );
    }
}
