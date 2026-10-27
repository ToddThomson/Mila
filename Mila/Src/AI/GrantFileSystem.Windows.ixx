/**
 * @file GrantFileSystem.Windows.ixx
 * @brief The operating system's file calls a grant makes on Windows: open, create, list, and what an open handle
 *        reports of itself.
 *
 * One of two candidate files for AI.GrantFileSystem; CMake compiles this one on Windows. Every function works on
 * an open handle or opens one, so a check is made on what was opened rather than on a name.
 */

module;

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>

export module AI.GrantFileSystem;

import std;
import AI.FolderEntry;

namespace Mila::AI::GrantFileSystem
{
    /// An open file or folder: a Windows HANDLE, or kNoHandle.
    export using NativeHandle = std::uintptr_t;

    export inline constexpr NativeHandle kNoHandle = 0;

    /// Whether this build can hold a tool to a folder.
    export inline constexpr bool kEnforcesFolders = true;

    namespace
    {
        // ntstatus.h cannot be included beside windows.h without redefining its own values.
        constexpr NTSTATUS kStatusObjectNameNotFound = static_cast<NTSTATUS>( 0xC0000034L );
        constexpr NTSTATUS kStatusObjectPathNotFound = static_cast<NTSTATUS>( 0xC000003AL );
        constexpr NTSTATUS kStatusObjectNameCollision = static_cast<NTSTATUS>( 0xC0000035L );
        constexpr NTSTATUS kStatusNotADirectory = static_cast<NTSTATUS>( 0xC0000103L );
        constexpr NTSTATUS kStatusFileIsADirectory = static_cast<NTSTATUS>( 0xC00000BAL );

        HANDLE native( NativeHandle handle ) noexcept
        {
            return reinterpret_cast<HANDLE>( handle );
        }

        NativeHandle wrap( HANDLE handle ) noexcept
        {
            return handle == INVALID_HANDLE_VALUE || handle == nullptr ? kNoHandle : reinterpret_cast<NativeHandle>( handle );
        }

        std::string describe( DWORD error )
        {
            wchar_t* buffer = nullptr;
            const DWORD length = FormatMessageW(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, error, 0, reinterpret_cast<wchar_t*>( &buffer ), 0, nullptr );

            if ( length == 0 || buffer == nullptr )
            {
                return std::format( "Windows error {}", error );
            }

            std::wstring text( buffer, length );
            LocalFree( buffer );

            while ( !text.empty() && ( text.back() == L'\n' || text.back() == L'\r' || text.back() == L'.' ) )
            {
                text.pop_back();
            }

            const std::u8string utf8 = std::filesystem::path( text ).u8string();

            return std::string( utf8.begin(), utf8.end() );
        }

        [[noreturn]] void fail( std::string_view what, DWORD error )
        {
            throw std::runtime_error( std::format( "{}: {}", what, describe( error ) ) );
        }

        std::string utf8( std::wstring_view text )
        {
            const std::u8string bytes = std::filesystem::path( text ).u8string();

            return std::string( bytes.begin(), bytes.end() );
        }

        FILE_STANDARD_INFO standardInformation( NativeHandle handle )
        {
            FILE_STANDARD_INFO information{};

            if ( !GetFileInformationByHandleEx( native( handle ), FileStandardInfo, &information, sizeof( information ) ) )
            {
                fail( "the file's information could not be read", GetLastError() );
            }

            return information;
        }

        /// Opens a name in a folder opened earlier: the name is found in that folder, not by walking a path again.
        std::optional<NativeHandle> openRelative( NativeHandle folder, const std::filesystem::path& name, bool as_folder,
            bool write, ULONG disposition, ULONG options )
        {
            const std::wstring& text = name.native();

            UNICODE_STRING object_name{};
            object_name.Buffer = const_cast<wchar_t*>( text.c_str() );
            object_name.Length = static_cast<USHORT>( text.size() * sizeof( wchar_t ) );
            object_name.MaximumLength = object_name.Length;

            OBJECT_ATTRIBUTES attributes{};
            InitializeObjectAttributes( &attributes, &object_name, OBJ_CASE_INSENSITIVE, native( folder ), nullptr );

            const ACCESS_MASK access = as_folder
                ? FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE
                : ( write ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ ) | SYNCHRONIZE;

            options |= FILE_SYNCHRONOUS_IO_NONALERT | ( as_folder ? FILE_DIRECTORY_FILE : FILE_NON_DIRECTORY_FILE );

            HANDLE opened = nullptr;
            IO_STATUS_BLOCK status_block{};

            const NTSTATUS status = NtCreateFile( &opened, access, &attributes, &status_block, nullptr, FILE_ATTRIBUTE_NORMAL,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, disposition, options, nullptr, 0 );

            if ( status >= 0 )
            {
                return reinterpret_cast<NativeHandle>( opened );
            }

            if ( status == kStatusObjectNameNotFound || status == kStatusObjectPathNotFound )
            {
                return std::nullopt;
            }

            if ( status == kStatusObjectNameCollision )
            {
                throw std::runtime_error( std::format(
                    "'{}' already exists as something that is not a file or folder of its own, such as a link", utf8( text ) ) );
            }

            if ( status == kStatusNotADirectory )
            {
                throw std::runtime_error( std::format( "'{}' is a file, not a folder", utf8( text ) ) );
            }

            if ( status == kStatusFileIsADirectory )
            {
                throw std::runtime_error( std::format( "'{}' is a folder, not a file", utf8( text ) ) );
            }

            fail( std::format( "'{}' could not be opened", utf8( text ) ), RtlNtStatusToDosError( status ) );
        }
    }

    /**
     * @brief Opens what a full path names, every link on the way followed, a file or a folder alike.
     *
     * Never truncates or creates anything. Sharing allows others to keep reading, writing and renaming, since the
     * handle, not the name, is what is checked and used.
     *
     * @return Nothing when no file or folder is there.
     * @throws std::runtime_error when something is there and cannot be opened.
     */
    export std::optional<NativeHandle> openExisting( const std::filesystem::path& path, bool write )
    {
        const HANDLE handle = CreateFileW( path.c_str(), write ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr );

        if ( handle != INVALID_HANDLE_VALUE )
        {
            return wrap( handle );
        }

        const DWORD error = GetLastError();

        if ( error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND )
        {
            return std::nullopt;
        }

        fail( "it could not be opened", error );
    }

    /**
     * @brief Opens an existing name in an open folder, following a link the name is.
     *
     * @return Nothing when the name is not there.
     * @throws std::runtime_error when it is there as the other kind, or cannot be opened.
     */
    export std::optional<NativeHandle> openChild( NativeHandle folder, const std::filesystem::path& name, bool as_folder, bool write )
    {
        return openRelative( folder, name, as_folder, write, FILE_OPEN, 0 );
    }

    /**
     * @brief Creates a new file, opened to read and write, or a new folder, in an open folder.
     *
     * Fails when anything already has the name, a link included: the name is never followed, so what is created
     * is always in the folder the handle holds.
     *
     * @throws std::runtime_error when it cannot be created.
     */
    export NativeHandle createChild( NativeHandle folder, const std::filesystem::path& name, bool as_folder )
    {
        const auto created = openRelative( folder, name, as_folder, true, FILE_CREATE, FILE_OPEN_REPARSE_POINT );

        if ( !created )
        {
            throw std::runtime_error( std::format( "'{}' could not be created", utf8( name.native() ) ) );
        }

        return *created;
    }

    /// The path the operating system reports for an open handle, every link resolved, in its \\?\ form.
    export std::filesystem::path finalPath( NativeHandle handle )
    {
        std::wstring path( MAX_PATH, L'\0' );

        for ( ;; )
        {
            const DWORD length = GetFinalPathNameByHandleW( native( handle ), path.data(), static_cast<DWORD>( path.size() ),
                FILE_NAME_NORMALIZED | VOLUME_NAME_DOS );

            if ( length == 0 )
            {
                fail( "its final path could not be read", GetLastError() );
            }

            if ( length < path.size() )
            {
                path.resize( length );

                return std::filesystem::path( path );
            }

            // Too small: the length is then the size needed, the terminating null included.
            path.resize( length );
        }
    }

    export bool isFolder( NativeHandle handle )
    {
        return standardInformation( handle ).Directory != FALSE;
    }

    /// How many names the file has on its volume; more than one means hard links.
    export std::uint32_t linkCount( NativeHandle handle )
    {
        return standardInformation( handle ).NumberOfLinks;
    }

    export std::uint64_t size( NativeHandle handle )
    {
        return static_cast<std::uint64_t>( standardInformation( handle ).EndOfFile.QuadPart );
    }

    /// The file's bytes from its start, up to the most asked for.
    export std::string read( NativeHandle handle, std::uint64_t most )
    {
        LARGE_INTEGER start{};

        if ( !SetFilePointerEx( native( handle ), start, nullptr, FILE_BEGIN ) )
        {
            fail( "the file could not be read", GetLastError() );
        }

        std::string bytes;
        std::array<char, 64 * 1024> buffer{};

        while ( bytes.size() < most )
        {
            const auto wanted = static_cast<DWORD>( std::min<std::uint64_t>( buffer.size(), most - bytes.size() ) );
            DWORD got = 0;

            if ( !ReadFile( native( handle ), buffer.data(), wanted, &got, nullptr ) )
            {
                fail( "the file could not be read", GetLastError() );
            }

            if ( got == 0 )
            {
                break;
            }

            bytes.append( buffer.data(), got );
        }

        return bytes;
    }

    /// Replaces the file's whole content through the handle.
    export void replace( NativeHandle handle, std::string_view bytes )
    {
        LARGE_INTEGER start{};

        if ( !SetFilePointerEx( native( handle ), start, nullptr, FILE_BEGIN ) || !SetEndOfFile( native( handle ) ) )
        {
            fail( "the file could not be written", GetLastError() );
        }

        while ( !bytes.empty() )
        {
            const auto wanted = static_cast<DWORD>( std::min<std::size_t>( bytes.size(), 1u << 30 ) );
            DWORD written = 0;

            if ( !WriteFile( native( handle ), bytes.data(), wanted, &written, nullptr ) )
            {
                fail( "the file could not be written", GetLastError() );
            }

            bytes.remove_prefix( written );
        }
    }

    /// The names in an open folder, as its own handle lists them, without "." and "..".
    export std::vector<FolderEntry> list( NativeHandle folder )
    {
        std::vector<FolderEntry> entries;
        std::vector<std::byte> buffer( 64 * 1024 );
        FILE_INFO_BY_HANDLE_CLASS kind = FileIdBothDirectoryRestartInfo;

        for ( ;; )
        {
            if ( !GetFileInformationByHandleEx( native( folder ), kind, buffer.data(), static_cast<DWORD>( buffer.size() ) ) )
            {
                const DWORD error = GetLastError();

                if ( error == ERROR_NO_MORE_FILES )
                {
                    break;
                }

                fail( "the folder could not be listed", error );
            }

            kind = FileIdBothDirectoryInfo;

            for ( std::size_t offset = 0;; )
            {
                const auto* entry = reinterpret_cast<const FILE_ID_BOTH_DIR_INFO*>( buffer.data() + offset );
                const std::wstring_view name( entry->FileName, entry->FileNameLength / sizeof( wchar_t ) );

                if ( name != L"." && name != L".." )
                {
                    const bool is_folder = ( entry->FileAttributes & FILE_ATTRIBUTE_DIRECTORY ) != 0;

                    entries.push_back( FolderEntry{ utf8( name ), is_folder,
                        ( entry->FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT ) != 0,
                        is_folder ? 0 : static_cast<std::uint64_t>( entry->EndOfFile.QuadPart ) } );
                }

                if ( entry->NextEntryOffset == 0 )
                {
                    break;
                }

                offset += entry->NextEntryOffset;
            }
        }

        return entries;
    }

    export void close( NativeHandle handle ) noexcept
    {
        if ( handle != kNoHandle )
        {
            CloseHandle( native( handle ) );
        }
    }
}
