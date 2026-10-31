/**
 * @file SecondNameWatch.ixx
 * @brief The files under one folder that have more than one name on disk, kept current between runs by watching the
 *        folder rather than walking it again. Windows only.
 */

module;

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>

export module AI.SecondNameWatch;

import std;

namespace Mila::AI
{
    // Of module linkage rather than internal, since the exported class's members use them.
    namespace SecondNameWatchDetail
    {
        /// The form a Windows call takes a path of any length in.
        inline std::wstring extended( const std::wstring& path )
        {
            if ( path.starts_with( LR"(\\?\)" ) )
            {
                return path;
            }

            if ( path.starts_with( LR"(\\)" ) )
            {
                return LR"(\\?\UNC\)" + path.substr( 2 );
            }

            return LR"(\\?\)" + path;
        }

        inline bool isWithin( const std::wstring& path, const std::wstring& folder )
        {
            return path.size() > folder.size() && path.starts_with( folder ) && path[ folder.size() ] == L'\\';
        }

        /// A file's attributes and how many names it has; nothing when it cannot be read or is not there.
        inline std::optional<FILE_STAT_BASIC_INFORMATION> stat( const std::wstring& path )
        {
            FILE_STAT_BASIC_INFORMATION information{};

            if ( !GetFileInformationByName( extended( path ).c_str(), FileStatBasicByNameInfo, &information, sizeof( information ) ) )
            {
                return std::nullopt;
            }

            return information;
        }
    }

    /**
     * @brief The walk and watcher of ContainedPrograms.md 4.3 for one folder.
     *
     * Watching starts when this is made, before the first walk, so nothing that changes during it is missed. The
     * watcher names a file that gains a name inside (as an added name) and, on ReFS, one that gains a name outside
     * (as a change to its attributes); it does not name a file whose other name is removed, so every file already
     * recorded is read again on each refresh. A watcher that overflows, or stops, names nothing, and the folder is
     * walked again. On any file system but ReFS a name made outside goes unreported, so the folder is walked whole
     * on every refresh (ContainedPrograms.md decision 7).
     *
     * Links and junctions are not entered: the container refuses what they lead to outside its folders, and what
     * they lead to inside is walked where it lies. A folder or file the person cannot read is passed over, since a
     * contained program, which has less, cannot read it either.
     */
    export class SecondNameWatch
    {
    public:

        /// Each file with more than one name, by its name under the folder, with every name it has.
        using Files = std::map<std::filesystem::path, std::vector<std::filesystem::path>>;

        /**
         * @brief Starts watching a folder.
         *
         * @param folder An absolute path in drive-letter or UNC form, as the grant's final path gives it.
         * @throws std::runtime_error when the folder cannot be opened or watched.
         */
        explicit SecondNameWatch( const std::filesystem::path& folder )
            : folder_( folder.native() )
        {
            while ( folder_.size() > 3 && folder_.back() == L'\\' )
            {
                folder_.pop_back();
            }

            std::wstring volume( folder_.size() + 2, L'\0' );

            if ( !GetVolumePathNameW( folder_.c_str(), volume.data(), static_cast<DWORD>( volume.size() ) ) )
            {
                throw std::runtime_error( std::format( "the volume of '{}' could not be read (error {})", folder.string(), GetLastError() ) );
            }

            volume.resize( std::wcslen( volume.c_str() ) );

            if ( !volume.empty() && volume.back() == L'\\' )
            {
                volume.pop_back();
            }

            volume_ = volume;

            std::array<wchar_t, MAX_PATH + 1> file_system{};

            if ( GetVolumeInformationW( ( volume_ + L"\\" ).c_str(), nullptr, 0, nullptr, nullptr, nullptr,
                file_system.data(), static_cast<DWORD>( file_system.size() ) ) )
            {
                watcher_reports_every_name_ = std::wstring_view( file_system.data() ) == L"ReFS";
            }

            folder_handle_ = CreateFileW( SecondNameWatchDetail::extended( folder_ ).c_str(), FILE_LIST_DIRECTORY,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr );

            if ( folder_handle_ == INVALID_HANDLE_VALUE )
            {
                throw std::runtime_error( std::format( "'{}' could not be opened to watch (error {})", folder.string(), GetLastError() ) );
            }

            read_event_ = CreateEventW( nullptr, TRUE, FALSE, nullptr );
            stop_event_ = CreateEventW( nullptr, TRUE, FALSE, nullptr );

            if ( !startRead() )
            {
                const DWORD error = GetLastError();
                close();

                throw std::runtime_error( std::format( "'{}' cannot be watched (error {})", folder.string(), error ) );
            }

            watcher_ = std::thread( [this] { watch(); } );
        }

        SecondNameWatch( const SecondNameWatch& ) = delete;
        SecondNameWatch& operator=( const SecondNameWatch& ) = delete;

        ~SecondNameWatch()
        {
            SetEvent( stop_event_ );

            if ( watcher_.joinable() )
            {
                watcher_.join();
            }

            close();
        }

        /// The folder watched, as given.
        [[nodiscard]] std::filesystem::path folder() const
        {
            return folder_;
        }

        /**
         * @brief Brings files() up to date: the whole folder walked the first time and after the watcher overflowed,
         *        otherwise the names the watcher reported since and every file already recorded read again.
         *
         * Not to be called from two threads at once.
         *
         * @return False when the deadline passed or stop was requested first; the folder is then walked whole next
         *         time.
         */
        bool refresh( std::chrono::steady_clock::time_point deadline, std::stop_token stop )
        {
            std::set<std::wstring> names;
            bool walk_whole = !walked_;

            {
                const std::lock_guard lock( mutex_ );
                names.swap( pending_ );
                walk_whole = walk_whole || overflowed_ || watching_ended_ || !watcher_reports_every_name_;
                overflowed_ = false;
            }

            if ( walk_whole )
            {
                Files fresh;

                if ( !walk( folder_, fresh, deadline, stop ) )
                {
                    markOverflowed();

                    return false;
                }

                files_.swap( fresh );
                walked_ = true;

                // What changed during the walk is read now, so the run starts from the folder as it is.
                const std::lock_guard lock( mutex_ );
                names.clear();
                names.swap( pending_ );
            }

            std::vector<std::wstring> recorded;

            for ( const auto& [name, all] : files_ )
            {
                recorded.push_back( name.native() );
            }

            for ( const std::wstring& name : recorded )
            {
                readAgain( name );
            }

            for ( const std::wstring& relative : names )
            {
                if ( stop.stop_requested() || std::chrono::steady_clock::now() > deadline )
                {
                    markOverflowed();

                    return false;
                }

                const std::wstring full = folder_ + L"\\" + relative;

                if ( !readName( full, deadline, stop ) )
                {
                    markOverflowed();

                    return false;
                }
            }

            return true;
        }

        /// Whether a refresh after the first reads only what the watcher named, rather than walking the folder whole.
        [[nodiscard]] bool watches() const noexcept
        {
            return watcher_reports_every_name_;
        }

        /// What the last refresh found.
        [[nodiscard]] const Files& files() const noexcept
        {
            return files_;
        }

    private:

        static constexpr DWORD kFilter = FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_ATTRIBUTES
            | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_CREATION | FILE_NOTIFY_CHANGE_SECURITY;

        // 64 KB lost nothing through a clean rebuild of every Mila target with the repository watched; 4 KB
        // overflowed 19 times (Notebooks/MxcConformance.md section 7).
        static constexpr DWORD kBufferBytes = 64 * 1024;

        void close() noexcept
        {
            if ( folder_handle_ != INVALID_HANDLE_VALUE )
            {
                CancelIoEx( folder_handle_, nullptr );
                CloseHandle( folder_handle_ );
                folder_handle_ = INVALID_HANDLE_VALUE;
            }

            if ( read_event_ != nullptr )
            {
                CloseHandle( read_event_ );
                read_event_ = nullptr;
            }

            if ( stop_event_ != nullptr )
            {
                CloseHandle( stop_event_ );
                stop_event_ = nullptr;
            }
        }

        bool startRead()
        {
            ResetEvent( read_event_ );
            overlapped_ = OVERLAPPED{};
            overlapped_.hEvent = read_event_;

            return ReadDirectoryChangesW( folder_handle_, buffer_.data(), kBufferBytes, TRUE, kFilter, nullptr, &overlapped_, nullptr ) != FALSE;
        }

        void markOverflowed()
        {
            const std::lock_guard lock( mutex_ );
            overflowed_ = true;
        }

        void watch()
        {
            const HANDLE waits[] = { stop_event_, read_event_ };

            for ( ;; )
            {
                if ( WaitForMultipleObjects( 2, waits, FALSE, INFINITE ) != WAIT_OBJECT_0 + 1 )
                {
                    CancelIoEx( folder_handle_, &overlapped_ );

                    DWORD ignored = 0;
                    GetOverlappedResult( folder_handle_, &overlapped_, &ignored, TRUE );

                    return;
                }

                DWORD bytes = 0;
                const bool read = GetOverlappedResult( folder_handle_, &overlapped_, &bytes, FALSE ) != FALSE;

                {
                    const std::lock_guard lock( mutex_ );

                    if ( !read || bytes == 0 )
                    {
                        overflowed_ = true;
                    }
                    else
                    {
                        for ( std::size_t offset = 0;; )
                        {
                            const auto* entry = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>( buffer_.data() + offset );
                            pending_.emplace( entry->FileName, entry->FileNameLength / sizeof( wchar_t ) );

                            if ( entry->NextEntryOffset == 0 )
                            {
                                break;
                            }

                            offset += entry->NextEntryOffset;
                        }
                    }
                }

                if ( !startRead() )
                {
                    // Watching has ended, the folder removed or its volume gone: every refresh walks it whole.
                    const std::lock_guard lock( mutex_ );
                    watching_ended_ = true;

                    return;
                }
            }
        }

        /// Every name a file has, each a full path on its volume.
        std::vector<std::filesystem::path> namesOf( const std::wstring& path ) const
        {
            std::vector<std::filesystem::path> names;
            std::wstring name( 512, L'\0' );
            DWORD length = static_cast<DWORD>( name.size() );
            HANDLE find = FindFirstFileNameW( SecondNameWatchDetail::extended( path ).c_str(), 0, &length, name.data() );

            if ( find == INVALID_HANDLE_VALUE && GetLastError() == ERROR_MORE_DATA )
            {
                name.resize( length );
                find = FindFirstFileNameW( SecondNameWatchDetail::extended( path ).c_str(), 0, &length, name.data() );
            }

            if ( find == INVALID_HANDLE_VALUE )
            {
                // Its names cannot be read, so none can be shown to lie inside: the file is recorded as its own only
                // name's, which no folder holds whole.
                names.emplace_back();

                return names;
            }

            for ( bool more = true; more; )
            {
                names.emplace_back( volume_ + std::wstring( name.c_str() ) );

                for ( ;; )
                {
                    length = static_cast<DWORD>( name.size() );

                    if ( FindNextFileNameW( find, &length, name.data() ) )
                    {
                        break;
                    }

                    if ( GetLastError() != ERROR_MORE_DATA )
                    {
                        more = false;

                        break;
                    }

                    name.resize( length );
                }
            }

            FindClose( find );

            return names;
        }

        void record( Files& files, const std::wstring& path ) const
        {
            files[ path ] = namesOf( path );
        }

        /// A file already recorded: kept with its names now while it has more than one, else dropped.
        void readAgain( const std::wstring& path )
        {
            const auto information = SecondNameWatchDetail::stat( path );

            if ( !information || ( information->FileAttributes & FILE_ATTRIBUTE_DIRECTORY ) != 0 || information->NumberOfLinks <= 1 )
            {
                files_.erase( path );

                return;
            }

            record( files_, path );
        }

        /// A name the watcher reported: a file read, a folder walked, a name gone dropped with all under it.
        bool readName( const std::wstring& path, std::chrono::steady_clock::time_point deadline, std::stop_token stop )
        {
            const auto information = SecondNameWatchDetail::stat( path );

            if ( !information || ( information->FileAttributes & FILE_ATTRIBUTE_DIRECTORY ) != 0 )
            {
                std::erase_if( files_, [&]( const auto& entry )
                    {
                        return entry.first.native() == path || SecondNameWatchDetail::isWithin( entry.first.native(), path );
                    } );
            }

            if ( !information )
            {
                return true;
            }

            if ( ( information->FileAttributes & FILE_ATTRIBUTE_DIRECTORY ) != 0 )
            {
                // A folder moved in brings its files with no name of their own reported.
                if ( ( information->FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT ) != 0 )
                {
                    return true;
                }

                return walk( path, files_, deadline, stop );
            }

            if ( information->NumberOfLinks > 1 )
            {
                record( files_, path );
            }
            else
            {
                files_.erase( path );
            }

            return true;
        }

        /// @return False when the deadline passed or stop was requested first.
        bool walk( const std::wstring& root, Files& files, std::chrono::steady_clock::time_point deadline, std::stop_token stop ) const
        {
            std::vector<std::wstring> folders{ root };
            std::size_t seen = 0;

            while ( !folders.empty() )
            {
                if ( stop.stop_requested() || std::chrono::steady_clock::now() > deadline )
                {
                    return false;
                }

                const std::wstring folder = std::move( folders.back() );
                folders.pop_back();

                WIN32_FIND_DATAW data{};
                const HANDLE find = FindFirstFileExW( ( SecondNameWatchDetail::extended( folder ) + L"\\*" ).c_str(),
                    FindExInfoBasic, &data, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH );

                if ( find == INVALID_HANDLE_VALUE )
                {
                    continue;
                }

                do
                {
                    if ( ++seen % 256 == 0 && ( stop.stop_requested() || std::chrono::steady_clock::now() > deadline ) )
                    {
                        FindClose( find );

                        return false;
                    }

                    const std::wstring_view name( data.cFileName );

                    if ( name == L"." || name == L".." )
                    {
                        continue;
                    }

                    const std::wstring path = folder + L"\\" + std::wstring( name );

                    if ( ( data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ) != 0 )
                    {
                        if ( ( data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT ) == 0 )
                        {
                            folders.push_back( path );
                        }

                        continue;
                    }

                    const auto information = SecondNameWatchDetail::stat( path );

                    if ( information && information->NumberOfLinks > 1 )
                    {
                        record( files, path );
                    }
                } while ( FindNextFileW( find, &data ) );

                FindClose( find );
            }

            return true;
        }

        std::wstring folder_;

        // The volume's own path, a drive letter or a mount point, with no trailing separator: a file's names are
        // reported from it.
        std::wstring volume_;

        HANDLE folder_handle_{ INVALID_HANDLE_VALUE };
        HANDLE read_event_{ nullptr };
        HANDLE stop_event_{ nullptr };
        OVERLAPPED overlapped_{};
        alignas( DWORD ) std::array<std::byte, kBufferBytes> buffer_{};
        std::thread watcher_;

        std::mutex mutex_;
        std::set<std::wstring> pending_;
        bool overflowed_{ false };
        bool watching_ended_{ false };

        // Only on ReFS was a name made outside for a file inside measured to be reported; NTFS reports none
        // (Notebooks/MxcConformance.md section 7), so elsewhere the folder is walked whole on every refresh.
        bool watcher_reports_every_name_{ false };

        bool walked_{ false };
        Files files_;
    };
}
