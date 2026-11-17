/**
 * @file GrantFileSystem.Apple.ixx
 * @brief The operating system's file calls a grant makes on macOS: open, create, list, and what an open file
 *        descriptor reports of itself.
 *
 * One of three candidate files for Agents.GrantFileSystem; CMake compiles this one on Apple platforms. Every function
 * works on an open descriptor or opens one, so a check is made on what was opened rather than on a name
 * (ToolCalling.md 3.4, Files on macOS).
 */

module;

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

export module Agents.GrantFileSystem;

import std;
import Agents.FolderEntry;
import Agents.GrantPathForm;

namespace Mila::Agents::GrantFileSystem
{
    /// An open file or folder: a file descriptor, or kNoHandle.
    export using NativeHandle = std::uintptr_t;

    // Descriptor 0 is a valid one, so no descriptor is the one value no descriptor can be.
    export inline constexpr NativeHandle kNoHandle = std::numeric_limits<NativeHandle>::max();

    /// Whether this build can hold a tool to a folder.
    export inline constexpr bool kEnforcesFolders = true;

    /// The form of path a grant accepts here.
    export inline constexpr GrantPathForm kPathForm = GrantPathForm::Posix;

    namespace
    {
        int native( NativeHandle handle ) noexcept
        {
            return static_cast<int>( handle );
        }

        [[noreturn]] void fail( std::string_view what, int error )
        {
            throw std::runtime_error( std::format( "{}: {}", what, std::generic_category().message( error ) ) );
        }

        std::string shown( const std::filesystem::path& path )
        {
            return path.native();
        }

        /// A refusal an open or a create met that is not a missing name, said as what it is.
        [[noreturn]] void refuse( std::string_view name, int error, std::string_view done )
        {
            switch ( error )
            {
                case EPERM:
                    // macOS's privacy consent refuses with EPERM, never ENOENT (measured, ToolCalling.md 3.4).
                    throw std::runtime_error( std::format(
                        "macOS did not allow '{}' to be {}; if it is in a folder macOS protects, such as Desktop, Documents or "
                        "Downloads, that is allowed in System Settings, under Privacy & Security", name, done ) );

                case ENOTCAPABLE:
                    // O_UNIQUE's refusal of a file with more than one name.
                    throw std::runtime_error( std::format(
                        "'{}' has more than one name on disk (a hard link), so where it lies cannot be told", name ) );

                case EEXIST:
                    throw std::runtime_error( std::format(
                        "'{}' already exists as something that is not a file or folder of its own, such as a link", name ) );

                case EISDIR:
                    throw std::runtime_error( std::format( "'{}' is a folder, not a file", name ) );

                case ENAMETOOLONG:
                    throw std::runtime_error( std::format( "the path to '{}' is longer than macOS can open", name ) );

                case EOPNOTSUPP:
                case ENXIO:
                    throw std::runtime_error( std::format( "'{}' is not a file or folder, but a pipe, socket or device", name ) );

                default:
                    fail( std::format( "'{}' could not be {}", name, done ), error );
            }
        }

        int openAt( int folder, const char* name, int flags, mode_t mode = 0 )
        {
            for ( ;; )
            {
                const int descriptor = ::openat( folder, name, flags, mode );

                if ( descriptor != -1 || errno != EINTR )
                {
                    return descriptor;
                }
            }
        }

        struct stat statusOf( NativeHandle handle )
        {
            struct stat status{};

            if ( ::fstat( native( handle ), &status ) != 0 )
            {
                fail( "the file's information could not be read", errno );
            }

            return status;
        }

        /// Keeps a descriptor only for a regular file or a folder; a pipe, a socket or a device is closed and refused.
        NativeHandle keepFileOrFolder( int descriptor, std::string_view name )
        {
            struct stat status{};

            if ( ::fstat( descriptor, &status ) != 0 )
            {
                const int error = errno;
                ::close( descriptor );
                fail( std::format( "'{}' could not be read", name ), error );
            }

            if ( !S_ISREG( status.st_mode ) && !S_ISDIR( status.st_mode ) )
            {
                ::close( descriptor );

                throw std::runtime_error( std::format( "'{}' is not a file or folder, but a pipe, socket or device", name ) );
            }

            return static_cast<NativeHandle>( descriptor );
        }

        // O_CLOEXEC: no program a tool starts later inherits the descriptor. O_NONBLOCK: a named pipe cannot hold
        // the open waiting for a writer; it opens at once and fstat refuses it.
        constexpr int kOpenFlags = O_CLOEXEC | O_NONBLOCK;
    }

    /**
     * @brief The path a full path from the root starts at: its first name, as grantPathNames gives it, is "/".
     */
    export std::filesystem::path rootPath( std::string_view )
    {
        return std::filesystem::path( "/" );
    }

    /**
     * @brief Opens what a full path names, every link on the way followed, a file or a folder alike.
     *
     * Never truncates or creates anything. A file with more than one name is opened, and its link count, read from
     * the open descriptor, decides as it does on Windows.
     *
     * @return Nothing when no file or folder is there.
     * @throws std::runtime_error when something is there and cannot be opened, or is not a file or folder.
     */
    export std::optional<NativeHandle> openExisting( const std::filesystem::path& path, bool write )
    {
        const int descriptor = openAt( AT_FDCWD, path.c_str(), ( write ? O_RDWR : O_RDONLY ) | kOpenFlags );

        if ( descriptor == -1 )
        {
            const int error = errno;

            if ( error == ENOENT || error == ENOTDIR )
            {
                return std::nullopt;
            }

            refuse( shown( path ), error, "opened" );
        }

        return keepFileOrFolder( descriptor, shown( path ) );
    }

    /**
     * @brief Opens an existing name in an open folder, following a link the name is.
     *
     * A file is opened with O_UNIQUE, so one with more than one name is refused by the open itself.
     *
     * @return Nothing when the name is not there, a link to nothing included.
     * @throws std::runtime_error when it is there as the other kind, is not a file or folder, or cannot be opened.
     */
    export std::optional<NativeHandle> openChild( NativeHandle folder, const std::filesystem::path& name, bool as_folder, bool write )
    {
        const int access = as_folder ? O_RDONLY | O_DIRECTORY : ( write ? O_RDWR : O_RDONLY ) | O_UNIQUE;
        const int descriptor = openAt( native( folder ), name.c_str(), access | kOpenFlags );

        if ( descriptor == -1 )
        {
            const int error = errno;

            if ( error == ENOENT )
            {
                return std::nullopt;
            }

            if ( error == ENOTDIR && as_folder )
            {
                throw std::runtime_error( std::format( "'{}' is a file, not a folder", shown( name ) ) );
            }

            refuse( shown( name ), error, "opened" );
        }

        const NativeHandle opened = keepFileOrFolder( descriptor, shown( name ) );

        if ( !as_folder && S_ISDIR( statusOf( opened ).st_mode ) )
        {
            ::close( descriptor );

            throw std::runtime_error( std::format( "'{}' is a folder, not a file", shown( name ) ) );
        }

        return opened;
    }

    /**
     * @brief Creates a new file, opened to read and write, or a new folder, in an open folder.
     *
     * Fails when anything already has the name, a link included: the name is never followed, so what is created is
     * always in the folder the descriptor holds. A new folder is opened through the same folder without following a
     * link, so one swapped in for it between the two calls is refused.
     *
     * @throws std::runtime_error when it cannot be created.
     */
    export NativeHandle createChild( NativeHandle folder, const std::filesystem::path& name, bool as_folder )
    {
        if ( !as_folder )
        {
            const int descriptor = openAt( native( folder ), name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666 );

            if ( descriptor == -1 )
            {
                refuse( shown( name ), errno, "created" );
            }

            return static_cast<NativeHandle>( descriptor );
        }

        if ( ::mkdirat( native( folder ), name.c_str(), 0777 ) != 0 )
        {
            refuse( shown( name ), errno, "created" );
        }

        const int descriptor = openAt( native( folder ), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | kOpenFlags );

        if ( descriptor == -1 )
        {
            const int error = errno;

            // A link in its place fails O_DIRECTORY | O_NOFOLLOW with ENOTDIR, not ELOOP (measured, ToolCalling.md 3.4).
            if ( error == ENOTDIR || error == ELOOP || error == ENOENT )
            {
                throw std::runtime_error( std::format(
                    "'{}' was replaced by something that is not a folder before it could be opened", shown( name ) ) );
            }

            refuse( shown( name ), error, "opened" );
        }

        return static_cast<NativeHandle>( descriptor );
    }

    /**
     * @brief The path the operating system reports for an open descriptor, every link and firmlink resolved, in the
     *        case and Unicode form stored on disk.
     *
     * @throws std::runtime_error when the path is longer than F_GETPATH can report.
     */
    export std::filesystem::path finalPath( NativeHandle handle )
    {
        std::array<char, MAXPATHLEN> path{};

        if ( ::fcntl( native( handle ), F_GETPATH, path.data() ) == -1 )
        {
            const int error = errno;

            if ( error == ENOSPC || error == ENAMETOOLONG )
            {
                throw std::runtime_error( std::format(
                    "the path is longer than the {} bytes macOS can report, so where it lies cannot be told", MAXPATHLEN ) );
            }

            fail( "its final path could not be read", error );
        }

        return std::filesystem::path( path.data() );
    }

    export bool isFolder( NativeHandle handle )
    {
        return S_ISDIR( statusOf( handle ).st_mode );
    }

    /**
     * @brief How many names the file has on its volume; more than one means hard links.
     *
     * A folder's st_nlink counts its subfolders, not its names, and no folder can be given a second name (link()
     * refuses one on every volume, measured), so a folder reports one.
     */
    export std::uint32_t linkCount( NativeHandle handle )
    {
        const struct stat status = statusOf( handle );

        return S_ISDIR( status.st_mode ) ? 1u : static_cast<std::uint32_t>( status.st_nlink );
    }

    export std::uint64_t size( NativeHandle handle )
    {
        return static_cast<std::uint64_t>( statusOf( handle ).st_size );
    }

    /// The file's bytes from its start, up to the most asked for.
    export std::string read( NativeHandle handle, std::uint64_t most )
    {
        std::string bytes;
        std::array<char, 64 * 1024> buffer{};
        off_t offset = 0;

        while ( bytes.size() < most )
        {
            const auto wanted = static_cast<std::size_t>( std::min<std::uint64_t>( buffer.size(), most - bytes.size() ) );
            const ssize_t got = ::pread( native( handle ), buffer.data(), wanted, offset );

            if ( got < 0 )
            {
                if ( errno == EINTR )
                {
                    continue;
                }

                fail( "the file could not be read", errno );
            }

            if ( got == 0 )
            {
                break;
            }

            bytes.append( buffer.data(), static_cast<std::size_t>( got ) );
            offset += got;
        }

        return bytes;
    }

    /// Replaces the file's whole content through the descriptor.
    export void replace( NativeHandle handle, std::string_view bytes )
    {
        if ( ::ftruncate( native( handle ), 0 ) != 0 )
        {
            fail( "the file could not be written", errno );
        }

        off_t offset = 0;

        while ( !bytes.empty() )
        {
            const ssize_t written = ::pwrite( native( handle ), bytes.data(), std::min<std::size_t>( bytes.size(), 1u << 30 ), offset );

            if ( written < 0 )
            {
                if ( errno == EINTR )
                {
                    continue;
                }

                fail( "the file could not be written", errno );
            }

            bytes.remove_prefix( static_cast<std::size_t>( written ) );
            offset += written;
        }
    }

    /**
     * @brief The names in an open folder, as its own descriptor lists them, without "." and "..".
     *
     * Each is described by fstatat on the folder: a link is listed as one, and as a folder when what it leads to is
     * one; a link to nothing is no folder.
     */
    export std::vector<FolderEntry> list( NativeHandle folder )
    {
        const int copy = ::dup( native( folder ) );

        if ( copy == -1 )
        {
            fail( "the folder could not be listed", errno );
        }

        // fdopendir takes the duplicate, and closedir closes it; the folder's own descriptor stays open.
        std::unique_ptr<DIR, int ( * )( DIR* )> stream( ::fdopendir( copy ), &::closedir );

        if ( !stream )
        {
            const int error = errno;
            ::close( copy );
            fail( "the folder could not be listed", error );
        }

        // A duplicate shares the descriptor's position, so a folder listed before is read again from its start.
        ::rewinddir( stream.get() );

        std::vector<FolderEntry> entries;

        for ( ;; )
        {
            errno = 0;
            const dirent* entry = ::readdir( stream.get() );

            if ( entry == nullptr )
            {
                if ( errno != 0 )
                {
                    fail( "the folder could not be listed", errno );
                }

                break;
            }

            const std::string_view name( entry->d_name, entry->d_namlen );

            if ( name == "." || name == ".." )
            {
                continue;
            }

            struct stat own{};

            // A name removed since it was listed is left out.
            if ( ::fstatat( native( folder ), entry->d_name, &own, AT_SYMLINK_NOFOLLOW ) != 0 )
            {
                continue;
            }

            const bool is_link = S_ISLNK( own.st_mode );
            bool is_folder = S_ISDIR( own.st_mode );

            if ( is_link )
            {
                struct stat target{};
                is_folder = ::fstatat( native( folder ), entry->d_name, &target, 0 ) == 0 && S_ISDIR( target.st_mode );
            }

            entries.push_back( FolderEntry{ std::string( name ), is_folder, is_link,
                S_ISREG( own.st_mode ) ? static_cast<std::uint64_t>( own.st_size ) : 0 } );
        }

        return entries;
    }

    export void close( NativeHandle handle ) noexcept
    {
        if ( handle != kNoHandle )
        {
            ::close( native( handle ) );
        }
    }
}
