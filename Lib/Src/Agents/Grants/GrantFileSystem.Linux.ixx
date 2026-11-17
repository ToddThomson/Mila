/**
 * @file GrantFileSystem.Linux.ixx
 * @brief The operating system's file calls a grant makes on Linux: open, create, list, and what an open file
 *        descriptor reports of itself.
 *
 * One of four candidate files for Agents.GrantFileSystem; CMake compiles this one on Linux. Every function works on an
 * open descriptor or opens one, so a check is made on what was opened rather than on a name (ToolCalling.md 3.4,
 * Files on Linux).
 */

module;

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/openat2.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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
        // The longest path /proc reports for a descriptor; past it readlink fails with ENAMETOOLONG (measured).
        constexpr std::size_t kReportedPathLimit = 4096;

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
                case ELOOP:
                    // RESOLVE_NO_MAGICLINKS refuses a /proc link with ELOOP, as too many links are refused.
                    throw std::runtime_error( std::format(
                        "'{}' leads through a link in /proc to something a program holds open, or through too many links, "
                        "and a grant follows neither", name ) );

                case EEXIST:
                    throw std::runtime_error( std::format(
                        "'{}' already exists as something that is not a file or folder of its own, such as a link", name ) );

                case EISDIR:
                    throw std::runtime_error( std::format( "'{}' is a folder, not a file", name ) );

                case ENAMETOOLONG:
                    throw std::runtime_error( std::format( "the path to '{}' is longer than Linux can open", name ) );

                case ENXIO:
                case EOPNOTSUPP:
                    throw std::runtime_error( std::format( "'{}' is not a file or folder, but a pipe, socket or device", name ) );

                case ENOSYS:
                    throw std::runtime_error( std::format(
                        "'{}' could not be {}: a grant needs Linux 5.6 or later, and a sandbox that allows openat2", name, done ) );

                default:
                    fail( std::format( "'{}' could not be {}", name, done ), error );
            }
        }

        /**
         * openat2 with RESOLVE_NO_MAGICLINKS: a link in /proc to something a process holds open is never followed,
         * at the end of the path or within it (ToolCalling.md 3.4). glibc has no wrapper for openat2.
         */
        int openAt( int folder, const char* name, int flags, mode_t mode = 0 )
        {
            open_how how{};
            how.flags = static_cast<std::uint64_t>( static_cast<unsigned int>( flags ) );
            how.mode = ( flags & O_CREAT ) != 0 ? mode : 0;
            how.resolve = RESOLVE_NO_MAGICLINKS;

            for ( ;; )
            {
                const long descriptor = ::syscall( SYS_openat2, folder, name, &how, sizeof( how ) );

                if ( descriptor != -1 || errno != EINTR )
                {
                    return static_cast<int>( descriptor );
                }
            }
        }

        /// A descriptor this file opened for its own use, closed when it goes.
        class Descriptor
        {
        public:

            explicit Descriptor( int descriptor ) noexcept
                : descriptor_( descriptor )
            {
            }

            Descriptor( const Descriptor& ) = delete;
            Descriptor& operator=( const Descriptor& ) = delete;

            Descriptor( Descriptor&& other ) noexcept
                : descriptor_( std::exchange( other.descriptor_, -1 ) )
            {
            }

            Descriptor& operator=( Descriptor&& other ) noexcept
            {
                if ( this != &other )
                {
                    reset();
                    descriptor_ = std::exchange( other.descriptor_, -1 );
                }

                return *this;
            }

            ~Descriptor()
            {
                reset();
            }

            [[nodiscard]] int get() const noexcept
            {
                return descriptor_;
            }

            [[nodiscard]] bool valid() const noexcept
            {
                return descriptor_ != -1;
            }

        private:

            void reset() noexcept
            {
                if ( descriptor_ != -1 )
                {
                    ::close( descriptor_ );
                    descriptor_ = -1;
                }
            }

            int descriptor_{ -1 };
        };

        struct stat statusOf( int descriptor )
        {
            struct stat status{};

            if ( ::fstat( descriptor, &status ) != 0 )
            {
                fail( "the file's information could not be read", errno );
            }

            return status;
        }

        bool sameFile( const struct stat& left, const struct stat& right ) noexcept
        {
            return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
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

        /**
         * Calls each name in an open folder's listing, without "." and "..", until one returns true; a duplicate of the
         * descriptor is listed, so the folder's own stays open and its position is left alone.
         */
        template <typename TVisit>
        void eachName( int folder, TVisit&& visit )
        {
            const int copy = ::dup( folder );

            if ( copy == -1 )
            {
                fail( "the folder could not be listed", errno );
            }

            // fdopendir takes the duplicate, and closedir closes it.
            std::unique_ptr<DIR, int ( * )( DIR* )> stream( ::fdopendir( copy ), &::closedir );

            if ( !stream )
            {
                const int error = errno;
                ::close( copy );
                fail( "the folder could not be listed", error );
            }

            // A duplicate shares the descriptor's position, so a folder listed before is read again from its start.
            ::rewinddir( stream.get() );

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

                    return;
                }

                const std::string_view name( entry->d_name );

                if ( name == "." || name == ".." )
                {
                    continue;
                }

                if ( visit( entry->d_name ) )
                {
                    return;
                }
            }
        }

        /**
         * The name a folder's listing stores for the file or folder it holds, found by device and inode, so in the
         * case the volume keeps; a mount's root is found by the name it is mounted on, which fstatat crosses.
         */
        std::optional<std::string> storedName( int parent, const struct stat& wanted )
        {
            std::optional<std::string> found;

            eachName( parent, [&]( const char* name )
                {
                    struct stat status{};

                    if ( ::fstatat( parent, name, &status, AT_SYMLINK_NOFOLLOW ) == 0 && sameFile( status, wanted ) )
                    {
                        found = std::string( name );

                        return true;
                    }

                    return false;
                } );

            return found;
        }

        /// A folder's path from "/" in the names each listing on the way stores, found by going up through "..".
        std::filesystem::path storedPathOfFolder( int folder )
        {
            std::vector<std::string> names;
            Descriptor current( ::dup( folder ) );

            if ( !current.valid() )
            {
                fail( "where the folder lies could not be read", errno );
            }

            for ( ;; )
            {
                const struct stat here = statusOf( current.get() );
                Descriptor parent( openAt( current.get(), "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NONBLOCK ) );

                if ( !parent.valid() )
                {
                    fail( "a folder on the way to '/' could not be opened", errno );
                }

                // "/" is its own parent.
                if ( sameFile( here, statusOf( parent.get() ) ) )
                {
                    break;
                }

                const std::optional<std::string> name = storedName( parent.get(), here );

                if ( !name )
                {
                    throw std::runtime_error( "a folder on the way to '/' does not list the one below it, so where it lies cannot be told" );
                }

                names.push_back( *name );
                current = std::move( parent );
            }

            std::filesystem::path path( "/" );

            for ( auto name = names.rbegin(); name != names.rend(); ++name )
            {
                path /= *name;
            }

            return path;
        }

        // O_CLOEXEC: no program a tool starts later inherits the descriptor. O_NONBLOCK: a named pipe cannot hold
        // the open waiting for a writer; it opens at once and fstat refuses it. O_NOCTTY: a terminal opened through a
        // link never becomes the session's controlling terminal.
        constexpr int kOpenFlags = O_CLOEXEC | O_NONBLOCK | O_NOCTTY;
    }

    /**
     * @brief The path a full path from the root starts at: its first name, as grantPathNames gives it, is "/".
     */
    export std::filesystem::path rootPath( std::string_view )
    {
        return std::filesystem::path( "/" );
    }

    /**
     * @brief Opens what a full path names, every symbolic link on the way followed and no link in /proc, a file or a
     *        folder alike.
     *
     * Never truncates or creates anything. A file with more than one name is opened, and its link count, read from
     * the open descriptor, decides.
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
     * @brief Opens an existing name in an open folder, following a symbolic link the name is, never a link in /proc.
     *
     * @return Nothing when the name is not there, a link to nothing included.
     * @throws std::runtime_error when it is there as the other kind, is not a file or folder, or cannot be opened.
     */
    export std::optional<NativeHandle> openChild( NativeHandle folder, const std::filesystem::path& name, bool as_folder, bool write )
    {
        const int access = as_folder ? O_RDONLY | O_DIRECTORY : ( write ? O_RDWR : O_RDONLY );
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

        if ( !as_folder && S_ISDIR( statusOf( descriptor ).st_mode ) )
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
            const int descriptor = openAt( native( folder ), name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY, 0666 );

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

            // A link in its place fails O_DIRECTORY | O_NOFOLLOW with ENOTDIR (measured, ToolCalling.md 3.4).
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
     * @brief The path the kernel keeps for an open descriptor, read from /proc/self/fd, every symbolic link resolved,
     *        each name in the case it was looked up by.
     *
     * @throws std::runtime_error when /proc is not mounted, or the path is longer than /proc can report.
     */
    export std::filesystem::path finalPath( NativeHandle handle )
    {
        const std::string link = std::format( "/proc/self/fd/{}", native( handle ) );
        std::string path( kReportedPathLimit + 1, '\0' );
        const ssize_t length = ::readlink( link.c_str(), path.data(), path.size() );

        if ( length == -1 )
        {
            const int error = errno;

            if ( error == ENAMETOOLONG )
            {
                throw std::runtime_error( std::format(
                    "the path is longer than the {} bytes Linux can report, so where it lies cannot be told", kReportedPathLimit ) );
            }

            if ( error == ENOENT && ::access( "/proc/self", F_OK ) != 0 )
            {
                throw std::runtime_error( "Linux's /proc is not mounted here, so where a file lies cannot be told" );
            }

            fail( "its final path could not be read", error );
        }

        if ( static_cast<std::size_t>( length ) == path.size() )
        {
            throw std::runtime_error( std::format(
                "the path is longer than the {} bytes Linux can report, so where it lies cannot be told", kReportedPathLimit ) );
        }

        path.resize( static_cast<std::size_t>( length ) );

        if ( !path.starts_with( '/' ) )
        {
            throw std::runtime_error( std::format( "Linux reports '{}' for it, which is no path, so where it lies cannot be told", path ) );
        }

        return std::filesystem::path( path );
    }

    /**
     * @brief The path in the names the volume stores, case included, which on a volume that ignores case may differ
     *        from finalPath's.
     *
     * A folder's own path is found going up through "..", each name by device and inode in its parent's listing; a
     * file's is its folder's -- the one finalPath names, checked to hold this file -- and the name that folder lists
     * for it. Costs a listing per folder up to "/" (ToolCalling.md 3.4, Case).
     *
     * @throws std::runtime_error when a folder on the way cannot be opened or listed, or the file was moved meanwhile.
     */
    export std::filesystem::path onDiskPath( NativeHandle handle )
    {
        const struct stat status = statusOf( native( handle ) );

        if ( S_ISDIR( status.st_mode ) )
        {
            return storedPathOfFolder( native( handle ) );
        }

        const std::filesystem::path reported = finalPath( handle );
        const Descriptor folder( openAt( AT_FDCWD, reported.parent_path().c_str(), O_RDONLY | O_DIRECTORY | kOpenFlags ) );

        if ( !folder.valid() )
        {
            fail( "the folder holding it could not be opened", errno );
        }

        const std::optional<std::string> name = storedName( folder.get(), status );

        if ( !name )
        {
            throw std::runtime_error( "it was moved or removed while it was checked, so where it lies cannot be told" );
        }

        return storedPathOfFolder( folder.get() ) / *name;
    }

    export bool isFolder( NativeHandle handle )
    {
        return S_ISDIR( statusOf( native( handle ) ).st_mode );
    }

    /**
     * @brief How many names the file has on its volume; more than one means hard links.
     *
     * A folder's st_nlink counts its subfolders, not its names, and no folder can be given a second name (link()
     * refuses one, measured), so a folder reports one.
     */
    export std::uint32_t linkCount( NativeHandle handle )
    {
        const struct stat status = statusOf( native( handle ) );

        return S_ISDIR( status.st_mode ) ? 1u : static_cast<std::uint32_t>( status.st_nlink );
    }

    export std::uint64_t size( NativeHandle handle )
    {
        return static_cast<std::uint64_t>( statusOf( native( handle ) ).st_size );
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
        std::vector<FolderEntry> entries;

        eachName( native( folder ), [&]( const char* name )
            {
                struct stat own{};

                // A name removed since it was listed is left out.
                if ( ::fstatat( native( folder ), name, &own, AT_SYMLINK_NOFOLLOW ) != 0 )
                {
                    return false;
                }

                const bool is_link = S_ISLNK( own.st_mode );
                bool is_folder = S_ISDIR( own.st_mode );

                if ( is_link )
                {
                    struct stat target{};
                    is_folder = ::fstatat( native( folder ), name, &target, 0 ) == 0 && S_ISDIR( target.st_mode );
                }

                entries.push_back( FolderEntry{ std::string( name ), is_folder, is_link,
                    S_ISREG( own.st_mode ) ? static_cast<std::uint64_t>( own.st_size ) : 0 } );

                return false;
            } );

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
