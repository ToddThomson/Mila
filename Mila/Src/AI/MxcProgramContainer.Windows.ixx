/**
 * @file MxcProgramContainer.Windows.ixx
 * @brief A grant's program container on Windows, through Microsoft eXecution Container's C library, loaded at run
 *        time (ContainedPrograms.md 4).
 *
 * One of two candidate files for AI.MxcProgramContainer; CMake compiles this one on Windows when MILA_ENABLE_MXC is
 * on, and the other, whose container is always absent, otherwise.
 */

module;

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>

export module AI.MxcProgramContainer;

import std;
import nlohmann.json;
import AI.ContainedRun;
import AI.FolderAccess;
import AI.GrantFolder;
import AI.ProgramContainer;
import AI.ProgramResult;
import AI.PublicAddress;
import AI.SecondNameWatch;
import AI.ToolDesktop;

namespace Mila::AI
{
    // Of module linkage rather than internal, since the exported class's members use them.
    namespace MxcDetail
    {
        // The only version whose layouts are mirrored here: MXC publishes no C header (microsoft/mxc#1449).
        inline constexpr std::string_view kVersion = "1.0.1";

        // Layouts mirror src/ffi/mxc_ffi/src/error_detail.rs and streaming.rs at microsoft/mxc v1.0.1 (#[repr(C)]).
        struct ErrorDetail
        {
            char* message_utf8;
            char* operation_utf8;
            char* native_code_utf8;
            char* remediation_utf8;
        };

        struct Sandbox;
        struct ReadStream;

        using VersionFunction = const char* (*)();
        using PlatformSupportFunction = char* (*)();
        using StringFreeFunction = void (*)( char* );
        using SpawnFunction = std::int32_t (*)( const char*, std::int32_t, Sandbox**, ErrorDetail* );
        using TakeStreamFunction = ReadStream* (*)( Sandbox* );
        using StreamReadFunction = std::int32_t (*)( ReadStream*, std::uint8_t*, std::size_t, std::size_t* );
        using TryWaitFunction = std::int32_t (*)( Sandbox*, std::int32_t*, std::int32_t*, std::int32_t* );
        using KillFunction = std::int32_t (*)( Sandbox* );
        using SandboxFreeFunction = void (*)( Sandbox* );
        using ReadStreamFreeFunction = void (*)( ReadStream* );
        using ErrorDetailFreeFunction = void (*)( ErrorDetail* );

        struct Library
        {
            HMODULE module{ nullptr };
            SpawnFunction spawn{};
            TakeStreamFunction take_stdout{};
            TakeStreamFunction take_stderr{};
            StreamReadFunction stream_read{};
            TryWaitFunction try_wait{};
            KillFunction kill{};
            KillFunction kill_for_timeout{};
            SandboxFreeFunction sandbox_free{};
            ReadStreamFreeFunction stream_free{};
            ErrorDetailFreeFunction error_free{};
        };

        template <typename TFunction>
        TFunction resolve( HMODULE module, const char* name )
        {
            const auto address = GetProcAddress( module, name );

            if ( address == nullptr )
            {
                throw std::runtime_error( std::format( "MXC's library has no function {}", name ) );
            }

            return reinterpret_cast<TFunction>( address );
        }

        inline std::string utf8( const std::filesystem::path& path )
        {
            const std::u8string text = path.u8string();

            return std::string( text.begin(), text.end() );
        }

        /// A final path in the form MXC's policy takes: a drive-letter or UNC path, never the \\?\ form.
        inline std::wstring plain( const std::filesystem::path& path )
        {
            std::wstring text = path.native();

            if ( text.starts_with( LR"(\\?\UNC\)" ) )
            {
                return LR"(\\)" + text.substr( 8 );
            }

            if ( text.starts_with( LR"(\\?\)" ) )
            {
                return text.substr( 4 );
            }

            return text;
        }

        inline bool holds( const std::wstring& folder, const std::wstring& path )
        {
            return path == folder || ( path.size() > folder.size() && path.starts_with( folder )
                && ( folder.back() == L'\\' || path[ folder.size() ] == L'\\' ) );
        }

        /// Whether the person can make a name in a folder, and so a second name for a file inside it.
        inline bool personCanWrite( const std::wstring& folder )
        {
            const HANDLE handle = CreateFileW( ( LR"(\\?\)" + folder ).c_str(), FILE_ADD_FILE | FILE_ADD_SUBDIRECTORY,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr );

            if ( handle == INVALID_HANDLE_VALUE )
            {
                return false;
            }

            CloseHandle( handle );

            return true;
        }

        inline std::string environmentValue( const wchar_t* name )
        {
            std::wstring value( 32767, L'\0' );
            const DWORD length = GetEnvironmentVariableW( name, value.data(), static_cast<DWORD>( value.size() ) );
            value.resize( length );

            return utf8( std::filesystem::path( value ) );
        }

        /// Standard output or error read to the end, kept up to a cap, the rest read and dropped so the program never
        /// blocks on a full pipe.
        struct Drained
        {
            std::string kept;
            std::uint64_t dropped{ 0 };
        };

        inline void drain( const Library& library, ReadStream* stream, std::size_t cap, Drained& drained )
        {
            std::array<std::uint8_t, 16 * 1024> buffer{};

            for ( ;; )
            {
                std::size_t read = 0;

                if ( library.stream_read( stream, buffer.data(), buffer.size(), &read ) != 0 || read == 0 )
                {
                    return;
                }

                const std::size_t room = cap > drained.kept.size() ? cap - drained.kept.size() : 0;
                const std::size_t kept = std::min( room, read );
                drained.kept.append( reinterpret_cast<const char*>( buffer.data() ), kept );
                drained.dropped += read - kept;
            }
        }
    }

    /**
     * @brief Runs a program in an MXC process container held to the run's folders and network
     *        (ContainedPrograms.md 4.2), on a desktop of Mila's own (4.4).
     *
     * Before each run the granted folders, and any program folder the person can write to, are brought up to date
     * by their SecondNameWatch: a file with a name outside them, or in a folder granted less, is denied to that run
     * by its name inside and named in the result (4.3). That check counts against the run's time, and a run whose
     * check does not finish within it is refused.
     *
     * The run is given an environment of Mila's writing (4.7), never the person's: its temporary folders lie in the
     * first folder granted to write, under `.mila`. Telemetry is off in every request (4.10).
     */
    export class MxcProgramContainer final : public ProgramContainer
    {
    public:

        /**
         * @brief Loads MXC's library from a file and checks it is the version this build mirrors and that this
         *        machine runs its process container.
         *
         * @return The container, or why there is none.
         */
        static std::expected<std::shared_ptr<const MxcProgramContainer>, std::string> open( const std::filesystem::path& library )
        {
            const HMODULE module = LoadLibraryExW( std::filesystem::absolute( library ).c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS );

            if ( module == nullptr )
            {
                return std::unexpected( std::format( "MXC's library '{}' could not be loaded (error {})", MxcDetail::utf8( library ), GetLastError() ) );
            }

            try
            {
                const std::string_view version = MxcDetail::resolve<MxcDetail::VersionFunction>( module, "mxc_version" )();

                if ( version != MxcDetail::kVersion )
                {
                    throw std::runtime_error( std::format( "MXC's library is version {}, and this build runs only {}", version, MxcDetail::kVersion ) );
                }

                const auto string_free = MxcDetail::resolve<MxcDetail::StringFreeFunction>( module, "mxc_string_free" );
                char* support = MxcDetail::resolve<MxcDetail::PlatformSupportFunction>( module, "mxc_platform_support_json" )();
                const auto report = nlohmann::json::parse( support == nullptr ? "{}" : support, nullptr, false );
                string_free( support );

                const bool has_container = report.is_object() && report.value( "isSupported", false ) && report.contains( "availableMethods" )
                    && report[ "availableMethods" ].is_array()
                    && std::ranges::count( report[ "availableMethods" ], nlohmann::json( "processcontainer" ) ) > 0;

                if ( !has_container )
                {
                    const auto reason = report.is_object() && report.contains( "reason" ) && report[ "reason" ].is_string()
                        ? report[ "reason" ].get<std::string>() : std::string( "it names no reason" );

                    throw std::runtime_error( std::format( "this machine does not run MXC's process container: {}", reason ) );
                }

                MxcDetail::Library functions;
                functions.module = module;
                functions.spawn = MxcDetail::resolve<MxcDetail::SpawnFunction>( module, "mxc_spawn_json" );
                functions.take_stdout = MxcDetail::resolve<MxcDetail::TakeStreamFunction>( module, "mxc_sandbox_take_stdout" );
                functions.take_stderr = MxcDetail::resolve<MxcDetail::TakeStreamFunction>( module, "mxc_sandbox_take_stderr" );
                functions.stream_read = MxcDetail::resolve<MxcDetail::StreamReadFunction>( module, "mxc_stream_read" );
                functions.try_wait = MxcDetail::resolve<MxcDetail::TryWaitFunction>( module, "mxc_sandbox_try_wait" );
                functions.kill = MxcDetail::resolve<MxcDetail::KillFunction>( module, "mxc_sandbox_kill" );
                functions.kill_for_timeout = MxcDetail::resolve<MxcDetail::KillFunction>( module, "mxc_sandbox_kill_for_timeout" );
                functions.sandbox_free = MxcDetail::resolve<MxcDetail::SandboxFreeFunction>( module, "mxc_sandbox_free" );
                functions.stream_free = MxcDetail::resolve<MxcDetail::ReadStreamFreeFunction>( module, "mxc_read_stream_free" );
                functions.error_free = MxcDetail::resolve<MxcDetail::ErrorDetailFreeFunction>( module, "mxc_error_detail_free" );

                return std::shared_ptr<const MxcProgramContainer>( new MxcProgramContainer( functions ) );
            } catch ( const std::exception& error ) {
                FreeLibrary( module );

                return std::unexpected( std::string( error.what() ) );
            }
        }

        MxcProgramContainer( const MxcProgramContainer& ) = delete;
        MxcProgramContainer& operator=( const MxcProgramContainer& ) = delete;

        ~MxcProgramContainer() override
        {
            watches_.clear();
            desktop_.reset();
            FreeLibrary( library_.module );
        }

        ProgramResult run( const ContainedRun& run, std::stop_token stop ) const override
        {
            const auto deadline = std::chrono::steady_clock::now() + run.time;

            ProgramResult result;
            std::vector<std::wstring> denied;

            if ( !withhold( run, deadline, stop, denied, result.withheld ) )
            {
                if ( stop.stop_requested() )
                {
                    result.stopped = true;

                    return result;
                }

                throw std::runtime_error( "the folders granted are too large to check for files with a second name in the time "
                    "the run was given; give it more time" );
            }

            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>( deadline - std::chrono::steady_clock::now() );

            if ( left.count() <= 0 )
            {
                throw std::runtime_error( "checking the folders took all the time the run was given; give it more time" );
            }

            const std::string request = makeRequest( run, denied, left ).dump();

            std::exception_ptr failure;
            std::thread runner( [&]
                {
                    try
                    {
                        // A thread of its own, which owns no window, so it can take the desktop the program lands on.
                        desktop_->useOnThisThread();
                        runOnThisThread( request, run.output_bytes, deadline, stop, result );
                    } catch ( ... ) {
                        failure = std::current_exception();
                    }
                } );

            runner.join();

            if ( failure )
            {
                std::rethrow_exception( failure );
            }

            return result;
        }

    private:

        explicit MxcProgramContainer( const MxcDetail::Library& library )
            : library_( library ), desktop_( std::make_unique<ToolDesktop>() )
        {
        }

        /// Brings each folder's watch up to date and lists the names to deny this run, as MXC's policy and as the
        /// result shows them. @return False when the deadline passed or stop was requested first.
        bool withhold( const ContainedRun& run, std::chrono::steady_clock::time_point deadline, std::stop_token stop,
            std::vector<std::wstring>& denied, std::vector<std::string>& withheld ) const
        {
            struct Held
            {
                std::wstring folder;
                FolderAccess access;
            };

            std::vector<Held> held;

            for ( const GrantFolder& granted : run.folders )
            {
                held.push_back( { MxcDetail::plain( granted.folder ), granted.access } );
            }

            const std::size_t granted_count = held.size();

            for ( const auto& folder : run.program_folders )
            {
                held.push_back( { MxcDetail::plain( folder ), FolderAccess::Read } );
            }

            // The most a name is given: ReadWrite when a folder granted to write holds it.
            const auto access = [&held]( const std::wstring& path ) -> std::optional<FolderAccess>
                {
                    std::optional<FolderAccess> best;

                    for ( const Held& folder : held )
                    {
                        if ( MxcDetail::holds( folder.folder, path ) && ( !best || folder.access == FolderAccess::ReadWrite ) )
                        {
                            best = folder.access;
                        }
                    }

                    return best;
                };

            const std::lock_guard lock( mutex_ );

            for ( std::size_t index = 0; index < held.size(); ++index )
            {
                const std::wstring& folder = held[ index ].folder;

                // A program folder only an administrator can write to cannot be given a second name inside.
                if ( index >= granted_count && !MxcDetail::personCanWrite( folder ) )
                {
                    continue;
                }

                auto found = watches_.find( folder );

                if ( found == watches_.end() )
                {
                    found = watches_.emplace( folder, std::make_unique<SecondNameWatch>( std::filesystem::path( folder ) ) ).first;
                }

                if ( !found->second->refresh( deadline, stop ) )
                {
                    return false;
                }

                for ( const auto& [name, names] : found->second->files() )
                {
                    const auto given = access( name.native() );

                    if ( !given )
                    {
                        continue;
                    }

                    const bool every_name_held = std::ranges::all_of( names, [&]( const std::filesystem::path& other )
                        {
                            const auto other_given = access( other.native() );

                            return other_given && ( *given == FolderAccess::Read || *other_given == FolderAccess::ReadWrite );
                        } );

                    if ( !every_name_held && std::ranges::find( denied, name.native() ) == denied.end() )
                    {
                        denied.push_back( name.native() );
                        withheld.push_back( MxcDetail::utf8( name ) );
                    }
                }
            }

            return true;
        }

        /// The MXC 1.0.0 request for one run (ContainedPrograms.md 4.2).
        static nlohmann::json makeRequest( const ContainedRun& run, const std::vector<std::wstring>& denied, std::chrono::milliseconds left )
        {
            nlohmann::json read_write = nlohmann::json::array();
            nlohmann::json read_only = nlohmann::json::array();
            std::optional<std::filesystem::path> own_folder;

            for ( const GrantFolder& granted : run.folders )
            {
                const std::filesystem::path folder( MxcDetail::plain( granted.folder ) );

                if ( granted.access == FolderAccess::ReadWrite )
                {
                    read_write.push_back( MxcDetail::utf8( folder ) );

                    if ( !own_folder )
                    {
                        own_folder = folder / ".mila";
                    }
                }
                else
                {
                    read_only.push_back( MxcDetail::utf8( folder ) );
                }
            }

            if ( !own_folder )
            {
                throw std::runtime_error( "a program runs only with a folder granted to write, where its temporary files go" );
            }

            std::string path = MxcDetail::environmentValue( L"SystemRoot" ) + "\\System32;"
                + MxcDetail::environmentValue( L"SystemRoot" ) + "\\System32\\WindowsPowerShell\\v1.0";

            for ( const auto& folder : run.program_folders )
            {
                const std::string text = MxcDetail::utf8( std::filesystem::path( MxcDetail::plain( folder ) ) );
                read_only.push_back( text );
                path += ";" + text;
            }

            nlohmann::json denied_paths = nlohmann::json::array();

            for ( const std::wstring& name : denied )
            {
                denied_paths.push_back( MxcDetail::utf8( std::filesystem::path( name ) ) );
            }

            const std::filesystem::path temp = *own_folder / "Temp";
            const std::filesystem::path local = *own_folder / "Local";
            std::filesystem::create_directories( temp );
            std::filesystem::create_directories( local );

            const std::string system_root = MxcDetail::environmentValue( L"SystemRoot" );

            nlohmann::json egress = { { "default", run.network ? "allow" : "deny" } };

            if ( run.network )
            {
                nlohmann::json to = nlohmann::json::array();

                for ( const std::string& range : nonPublicRanges() )
                {
                    to.push_back( { { "cidr", range } } );
                }

                egress[ "deny" ] = nlohmann::json::array( { { { "to", to } } } );
            }

            return {
                { "version", "1.0.0" },
                { "containment", "processcontainer" },
                { "lifecycle", { { "destroyOnExit", true } } },
                { "process", {
                    { "commandLine", run.command_line },
                    { "cwd", MxcDetail::utf8( std::filesystem::path( MxcDetail::plain( run.start_folder ) ) ) },
                    { "timeout", left.count() },
                    { "env", nlohmann::json::array( {
                        "SystemRoot=" + system_root,
                        "windir=" + system_root,
                        "ComSpec=" + system_root + "\\System32\\cmd.exe",
                        "PATHEXT=.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC",
                        "PATH=" + path,
                        "TEMP=" + MxcDetail::utf8( temp ),
                        "TMP=" + MxcDetail::utf8( temp ),
                        "LOCALAPPDATA=" + MxcDetail::utf8( local ) } ) } } },
                { "filesystem", { { "readwritePaths", read_write }, { "readonlyPaths", read_only }, { "deniedPaths", denied_paths } } },
                { "network", { { "egress", egress }, { "ingress", { { "default", "deny" }, { "hostLoopback", "deny" } } } } },
                { "ui", { { "disable", false }, { "clipboard", "none" }, { "injection", false } } },
                { "telemetry", { { "enabled", false } } },
            };
        }

        /// Starts the run, reads its output on two threads, and ends it on stop or its deadline -- polling and ending
        /// from this one thread, as MXC's concurrency contract asks (streaming.rs).
        void runOnThisThread( const std::string& request, std::size_t cap, std::chrono::steady_clock::time_point deadline,
            std::stop_token stop, ProgramResult& result ) const
        {
            MxcDetail::Sandbox* sandbox = nullptr;
            MxcDetail::ErrorDetail error{};

            if ( library_.spawn( request.c_str(), 0, &sandbox, &error ) != 0 || sandbox == nullptr )
            {
                const std::string message = error.message_utf8 != nullptr ? error.message_utf8 : "MXC gave no reason";
                library_.error_free( &error );

                throw std::runtime_error( std::format( "the program could not be started: {}", message ) );
            }

            MxcDetail::ReadStream* out_stream = library_.take_stdout( sandbox );
            MxcDetail::ReadStream* err_stream = library_.take_stderr( sandbox );
            MxcDetail::Drained out;
            MxcDetail::Drained err;
            std::thread out_reader( [&] { if ( out_stream != nullptr ) MxcDetail::drain( library_, out_stream, cap, out ); } );
            std::thread err_reader( [&] { if ( err_stream != nullptr ) MxcDetail::drain( library_, err_stream, cap, err ); } );

            std::int32_t exit_code = 0;
            std::int32_t running = 1;
            std::int32_t timed_out = 0;
            bool ended = false;

            while ( running != 0 )
            {
                if ( library_.try_wait( sandbox, &exit_code, &running, &timed_out ) != 0 )
                {
                    // Its state cannot be read: end it rather than wait on what cannot be seen.
                    library_.kill( sandbox );
                    running = 0;
                }

                if ( running != 0 && !ended )
                {
                    if ( stop.stop_requested() )
                    {
                        library_.kill( sandbox );
                        result.stopped = true;
                        ended = true;
                    }
                    else if ( std::chrono::steady_clock::now() >= deadline )
                    {
                        library_.kill_for_timeout( sandbox );
                        result.timed_out = true;
                        ended = true;
                    }
                }

                if ( running != 0 )
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
                }
            }

            out_reader.join();
            err_reader.join();

            if ( out_stream != nullptr )
            {
                library_.stream_free( out_stream );
            }

            if ( err_stream != nullptr )
            {
                library_.stream_free( err_stream );
            }

            library_.sandbox_free( sandbox );

            result.exit_code = exit_code;
            result.timed_out = result.timed_out || timed_out != 0;
            result.output = std::move( out.kept );
            result.output_dropped = out.dropped;
            result.errors = std::move( err.kept );
            result.errors_dropped = err.dropped;
        }

        MxcDetail::Library library_;
        std::unique_ptr<ToolDesktop> desktop_;

        // One watch per folder, kept across runs; refreshed one run at a time.
        mutable std::mutex mutex_;
        mutable std::map<std::wstring, std::unique_ptr<SecondNameWatch>> watches_;
    };
}
