/**
 * @file Assistant.ProgramSetup.ixx
 * @brief What the Mila app gives the commands it runs: the folders of the build tools found on this machine and the
 *        variables they need inside the container (ContainedPrograms.md 4.6, 4.7; MxcConformance.md 7.1).
 */

module;

#include <windows.h>

export module Assistant.ProgramSetup;

import std;

import Mila;

namespace Mila::Assistant
{
    // Not exported, and of module linkage rather than internal, since the exported class's members use them.
    namespace ProgramSetupDetail
    {
        inline std::wstring wide( std::string_view text )
        {
            if ( text.empty() )
                return {};

            const int length = MultiByteToWideChar( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), nullptr, 0 );
            std::wstring result( static_cast<std::size_t>( length ), L'\0' );
            MultiByteToWideChar( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), result.data(), length );

            return result;
        }

        inline std::string environmentValue( const wchar_t* name )
        {
            std::wstring value( 32767, L'\0' );
            const DWORD length = GetEnvironmentVariableW( name, value.data(), static_cast<DWORD>( value.size() ) );
            value.resize( length );
            const std::u8string text = std::filesystem::path( value ).u8string();

            return std::string( text.begin(), text.end() );
        }

        /// A host program's standard output, run with no window, or nothing when it does not start or fails.
        inline std::optional<std::string> output( const std::string& command_line )
        {
            SECURITY_ATTRIBUTES inherit{ sizeof( SECURITY_ATTRIBUTES ), nullptr, TRUE };
            HANDLE read_end = nullptr;
            HANDLE write_end = nullptr;

            if ( !CreatePipe( &read_end, &write_end, &inherit, 0 ) )
                return std::nullopt;

            SetHandleInformation( read_end, HANDLE_FLAG_INHERIT, 0 );

            STARTUPINFOW startup{};
            startup.cb = sizeof( startup );
            startup.dwFlags = STARTF_USESTDHANDLES;
            startup.hStdOutput = write_end;
            startup.hStdError = write_end;
            startup.hStdInput = nullptr;

            PROCESS_INFORMATION process{};
            std::wstring line = wide( command_line );
            const BOOL started = CreateProcessW( nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                nullptr, &startup, &process );
            CloseHandle( write_end );

            if ( !started )
            {
                CloseHandle( read_end );

                return std::nullopt;
            }

            std::string text;
            std::array<char, 4096> buffer{};
            DWORD read = 0;

            while ( ReadFile( read_end, buffer.data(), static_cast<DWORD>( buffer.size() ), &read, nullptr ) && read > 0 )
                text.append( buffer.data(), read );

            WaitForSingleObject( process.hProcess, INFINITE );
            DWORD exit_code = 1;
            GetExitCodeProcess( process.hProcess, &exit_code );
            CloseHandle( process.hProcess );
            CloseHandle( process.hThread );
            CloseHandle( read_end );

            if ( exit_code != 0 )
                return std::nullopt;

            return text;
        }

        inline std::vector<std::string> splitPath( std::string_view path )
        {
            std::vector<std::string> folders;

            for ( const auto part : std::views::split( path, ';' ) )
            {
                std::string folder( part.begin(), part.end() );

                while ( !folder.empty() && ( folder.back() == '\\' || folder.back() == '/' ) )
                    folder.pop_back();

                if ( !folder.empty() )
                    folders.push_back( std::move( folder ) );
            }

            return folders;
        }

        inline std::string lower( std::string text )
        {
            for ( char& character : text )
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );

            return text;
        }

        inline std::filesystem::path pathFromUtf8( std::string_view text )
        {
            return std::filesystem::path( std::u8string( reinterpret_cast<const char8_t*>( text.data() ), text.size() ) );
        }
    }

    /**
     * @brief The build tools a command may use, found once on this machine: Git, CMake, Ninja, Node and .NET on the
     *        person's PATH, and Visual Studio's compiler with the Windows SDK, through Visual Studio's own setup.
     *
     * Only the folders of those tools are given, never the person's whole PATH. The variables are what the tools
     * were measured to need in the container: Visual Studio's INCLUDE, LIB and LIBPATH; Node told not to resolve
     * its scripts from the drive's root, which the container refuses; .NET's first-run work and telemetry off.
     */
    export class ProgramSetup
    {
    public:

        [[nodiscard]] static Mila::AI::ProgramFolders find()
        {
            using namespace ProgramSetupDetail;

            Mila::AI::ProgramFolders programs;
            std::set<std::string> seen;

            const auto add = [&]( const std::filesystem::path& folder )
                {
                    std::error_code error;

                    if ( !std::filesystem::is_directory( folder, error ) )
                        return;

                    const std::u8string text = folder.lexically_normal().u8string();

                    if ( seen.insert( lower( std::string( text.begin(), text.end() ) ) ).second )
                        programs.folders.push_back( folder.lexically_normal() );
                };

            const std::vector<std::string> path = splitPath( environmentValue( L"PATH" ) );

            for ( const std::string_view tool : { "git.exe", "cmake.exe", "ninja.exe", "node.exe", "dotnet.exe" } )
            {
                for ( const std::string& folder : path )
                {
                    std::error_code error;
                    const std::filesystem::path candidate = pathFromUtf8( folder );

                    if ( !std::filesystem::exists( candidate / tool, error ) )
                        continue;

                    add( candidate );

                    // Git's cmd folder starts the git under the install's own folders.
                    if ( tool == "git.exe" && lower( candidate.filename().string() ) == "cmd" )
                        add( candidate.parent_path() );

                    break;
                }
            }

            addVisualStudio( programs, add, path );

            programs.environment.push_back( "NODE_OPTIONS=--preserve-symlinks --preserve-symlinks-main" );

            // The compiler's separate PDB process cannot be reached from the container, so /Zi fails; _CL_ is added
            // after the command line, so /Z7 wins whatever the project asks for.
            programs.environment.push_back( "_CL_=/Z7" );

            // MSBuild reads a variable as a property: .NET's Git query reads the person's own .gitconfig, refused.
            programs.environment.push_back( "EnableSourceControlManagerQueries=false" );
            programs.environment.push_back( "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1" );
            programs.environment.push_back( "DOTNET_NOLOGO=1" );
            programs.environment.push_back( "DOTNET_CLI_TELEMETRY_OPTOUT=1" );

            return programs;
        }

        /// The setup for commands in one project: .NET's home there too, since it does not read HOME.
        [[nodiscard]] static Mila::AI::ProgramFolders forProject( Mila::AI::ProgramFolders programs, const std::filesystem::path& project )
        {
            const std::u8string home = ( project / ".mila" / "Home" ).u8string();
            programs.environment.push_back( "DOTNET_CLI_HOME=" + std::string( home.begin(), home.end() ) );

            return programs;
        }

        /**
         * @brief The person's Git name and email, and nothing else of their Git settings, in the home a command
         *        is given in the project, so a commit it makes is theirs; left as it is when it is already there.
         *        A .gitignore in .mila keeps the tools' home, caches and the command's script out of the project's
         *        history.
         */
        static void writeGitIdentity( const std::filesystem::path& project )
        {
            using namespace ProgramSetupDetail;

            const std::filesystem::path config = project / ".mila" / "Home" / ".gitconfig";
            std::error_code error;
            std::filesystem::create_directories( config.parent_path(), error );

            // The model's own notes in .mila stay the project's to keep.
            if ( const std::filesystem::path ignore = project / ".mila" / ".gitignore"; !std::filesystem::exists( ignore, error ) )
                std::ofstream( ignore, std::ios::binary ) << "Home/\nRoaming/\ncommand.ps1\n";

            if ( std::filesystem::exists( config, error ) )
                return;

            const auto value = []( const char* key ) -> std::string
                {
                    std::string text = output( std::format( "git.exe config --global {}", key ) ).value_or( "" );

                    while ( !text.empty() && ( text.back() == '\n' || text.back() == '\r' ) )
                        text.pop_back();

                    return text;
                };

            const std::string name = value( "user.name" );
            const std::string email = value( "user.email" );

            if ( name.empty() && email.empty() )
                return;

            std::ofstream stream( config, std::ios::binary | std::ios::trunc );
            stream << "[user]\n";

            if ( !name.empty() )
                stream << "\tname = " << name << "\n";

            if ( !email.empty() )
                stream << "\temail = " << email << "\n";
        }

    private:

        /// The folders Visual Studio's x64 setup adds to PATH under Visual Studio or the Windows SDK, and its three
        /// library variables with the folders they name; nothing when Visual Studio has no C++ tools here.
        template <typename TAdd>
        static void addVisualStudio( Mila::AI::ProgramFolders& programs, TAdd& add, const std::vector<std::string>& host_path )
        {
            using namespace ProgramSetupDetail;

            const std::filesystem::path vswhere = pathFromUtf8( environmentValue( L"ProgramFiles(x86)" ) )
                / "Microsoft Visual Studio" / "Installer" / "vswhere.exe";
            std::error_code error;

            if ( !std::filesystem::exists( vswhere, error ) )
                return;

            std::string install = output( std::format( "\"{}\" -latest -products * -requires "
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath",
                vswhere.string() ) ).value_or( "" );

            while ( !install.empty() && ( install.back() == '\n' || install.back() == '\r' ) )
                install.pop_back();

            if ( install.empty() )
                return;

            const std::filesystem::path vcvars = pathFromUtf8( install ) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat";
            const std::optional<std::string> set = output( std::format( "cmd.exe /d /c \"\"{}\" >nul 2>&1 && set\"", vcvars.string() ) );

            if ( !set )
                return;

            std::set<std::string> before;

            for ( const std::string& folder : host_path )
                before.insert( lower( folder ) );

            const std::string kits = lower( environmentValue( L"ProgramFiles(x86)" ) + "\\Windows Kits" );
            const std::string studio = lower( install );

            for ( const auto part : std::views::split( std::string_view( *set ), '\n' ) )
            {
                std::string line( part.begin(), part.end() );

                if ( !line.empty() && line.back() == '\r' )
                    line.pop_back();

                const std::size_t equals = line.find( '=' );

                if ( equals == std::string::npos )
                    continue;

                const std::string name = lower( line.substr( 0, equals ) );

                // The folders they name are given to read too, so the compiler and linker can list their own.
                if ( name == "include" || name == "lib" || name == "libpath" )
                {
                    programs.environment.push_back( line );

                    for ( const std::string& folder : splitPath( line.substr( equals + 1 ) ) )
                        add( pathFromUtf8( folder ) );
                }
                else if ( name == "path" )
                {
                    for ( const std::string& folder : splitPath( line.substr( equals + 1 ) ) )
                    {
                        const std::string key = lower( folder );

                        if ( !before.contains( key ) && ( key.starts_with( studio ) || key.starts_with( kits ) ) )
                            add( pathFromUtf8( folder ) );
                    }
                }
            }
        }
    };
}
