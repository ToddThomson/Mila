/**
 * @file ToolGrant.ixx
 * @brief What one allowed tool call may reach -- folders, read-only or read-write, and whether the network -- and
 *        the object the tool reaches them through.
 */

export module AI.ToolGrant;

import std;
import AI.FolderAccess;
import AI.GrantFolder;
import AI.FolderEntry;
import AI.GrantFile;
import AI.GrantFileSystem;
import AI.GrantPath;
import AI.PublicAddress;
import AI.ToolNeeds;
import Distribution.HttpTransport;
import Distribution.HttpTransportBackend;

namespace Mila::AI
{
    // Not exported, and of module linkage rather than internal, since the exported class's members use them.
    namespace ToolGrantDetail
    {
        /// An open file or folder, closed when this goes.
        class HeldHandle
        {
        public:

            HeldHandle() noexcept = default;

            explicit HeldHandle( GrantFileSystem::NativeHandle handle ) noexcept
                : handle_( handle )
            {
            }

            HeldHandle( HeldHandle&& other ) noexcept
                : handle_( std::exchange( other.handle_, GrantFileSystem::kNoHandle ) )
            {
            }

            HeldHandle& operator=( HeldHandle&& other ) noexcept
            {
                if ( this != &other )
                {
                    GrantFileSystem::close( handle_ );
                    handle_ = std::exchange( other.handle_, GrantFileSystem::kNoHandle );
                }

                return *this;
            }

            ~HeldHandle()
            {
                GrantFileSystem::close( handle_ );
            }

            [[nodiscard]] GrantFileSystem::NativeHandle get() const noexcept
            {
                return handle_;
            }

            [[nodiscard]] GrantFileSystem::NativeHandle release() noexcept
            {
                return std::exchange( handle_, GrantFileSystem::kNoHandle );
            }

        private:

            GrantFileSystem::NativeHandle handle_{ GrantFileSystem::kNoHandle };
        };

        inline std::filesystem::path pathFromUtf8( std::string_view text )
        {
            return std::filesystem::path( std::u8string( reinterpret_cast<const char8_t*>( text.data() ), text.size() ) );
        }

        inline bool isDrive( std::string_view name ) noexcept
        {
            return name.size() == 2 && name[ 1 ] == ':';
        }
    }

    /**
     * @brief The grant a tool acts through (ToolCalling.md 3.4): it opens files and makes connections, and checks
     *        each at that moment.
     *
     * A path is accepted in one form only (grantPathNames): a drive-letter path, or a path relative to the first
     * folder granted. A file or folder is opened, the final path the operating system reports for the open handle
     * is taken, and it is kept only when that path lies within a granted folder's own final path, taken when the
     * grant was made. Short names, differences of case, symbolic links, junctions and mount points all lead to that
     * final path. A file with more than one name on disk -- a hard link -- is refused, since the name reported may
     * be one inside while another is outside. A file or folder is created only in a folder already opened and
     * checked, and never through a name that already exists, a link included.
     *
     * The network reaches public addresses only (isPublicAddress), decided on each connection as it is made, a
     * redirect's included, and never through a proxy.
     *
     * A refusal throws, naming the path as the model gave it; the gate gives the model the message as the call's
     * error. Copies share nothing that can change.
     */
    export class ToolGrant
    {
    public:

        /// Called for each name a walk finds: a file's false ends the walk; a folder's false leaves it unentered.
        using Visit = std::function<bool( std::string_view relative, const FolderEntry& entry )>;

        /// A grant of nothing: no folder and no network.
        ToolGrant() = default;

        /**
         * @brief A grant of these folders and, when asked for, the network.
         *
         * Each folder's own final path is taken now; everything a tool opens is held to those.
         *
         * @throws std::invalid_argument for a folder that does not exist, is not a folder, or whose final path the
         *         operating system cannot report; and for any folder where this build cannot hold a tool to one.
         */
        ToolGrant( std::vector<GrantFolder> folders, bool network )
        {
            for ( auto& granted : folders )
            {
                if ( !GrantFileSystem::kEnforcesFolders )
                {
                    throw std::invalid_argument( "a grant of folders is enforced only on Windows so far" );
                }

                granted.folder = std::filesystem::absolute( granted.folder ).lexically_normal();

                const auto opened = GrantFileSystem::openExisting( granted.folder, false );

                if ( !opened )
                {
                    throw std::invalid_argument( std::format( "'{}' does not exist", shown( granted.folder ) ) );
                }

                const ToolGrantDetail::HeldHandle held( *opened );

                if ( !GrantFileSystem::isFolder( held.get() ) )
                {
                    throw std::invalid_argument( std::format( "'{}' is not a folder", shown( granted.folder ) ) );
                }

                final_paths_.push_back( GrantFileSystem::finalPath( held.get() ) );
                folders_.push_back( std::move( granted ) );
            }

            if ( network )
            {
                network_ = Distribution::makeCheckedHttpTransport( []( std::span<const std::uint8_t> address )
                    {
                        return isPublicAddress( address );
                    } );
            }
        }

        /// The folders granted, each as the application named it, made absolute, and its access.
        [[nodiscard]] std::span<const GrantFolder> folders() const noexcept
        {
            return folders_;
        }

        [[nodiscard]] bool network() const noexcept
        {
            return network_ != nullptr;
        }

        /**
         * @brief The part of this grant a tool with these needs is given: folders only to a tool that reads or
         *        writes files, read-only unless it writes them, and the network only to a tool that uses it.
         */
        [[nodiscard]] ToolGrant narrowedTo( const ToolNeeds& needs ) const
        {
            ToolGrant narrowed = *this;

            if ( !needs.reads_files && !needs.writes_files )
            {
                narrowed.folders_.clear();
                narrowed.final_paths_.clear();
            }
            else if ( !needs.writes_files )
            {
                for ( auto& folder : narrowed.folders_ )
                {
                    folder.access = FolderAccess::Read;
                }
            }

            if ( !needs.uses_network )
            {
                narrowed.network_.reset();
            }

            return narrowed;
        }

        /**
         * @brief Opens an existing file to read.
         *
         * @throws std::invalid_argument for a path in a form the grant refuses.
         * @throws std::runtime_error when nothing is there, it is a folder, it lies outside every granted folder,
         *         or it has more than one name on disk.
         */
        [[nodiscard]] GrantFile openFile( std::string_view path ) const
        {
            ToolGrantDetail::HeldHandle held = openChecked( path, false );

            return GrantFile( held.release(), false );
        }

        /**
         * @brief Opens a file to read and replace, creating it, and the folders on its way, when asked to and it is
         *        not there.
         *
         * Nothing is truncated or created before the folder it lies in is checked: a folder's own final path must
         * lie within a folder granted read-write.
         *
         * @throws std::invalid_argument for a path in a form the grant refuses.
         * @throws std::runtime_error when it is not there and create is false, it is a folder, it or a folder on
         *         its way lies outside every folder granted read-write, or it has more than one name on disk.
         */
        [[nodiscard]] GrantFile openFileToWrite( std::string_view path, bool create ) const
        {
            requireFolders( path );

            if ( std::ranges::none_of( folders_, []( const GrantFolder& folder ) { return folder.access == FolderAccess::ReadWrite; } ) )
            {
                throw std::runtime_error( std::format( "this tool was granted its folders only to read, so it cannot write '{}'", path ) );
            }

            const std::vector<std::string> names = grantPathNames( path );
            const std::size_t base = !names.empty() && ToolGrantDetail::isDrive( names.front() ) ? 1 : 0;

            if ( names.size() <= base )
            {
                throw std::runtime_error( std::format( "'{}' is a folder, not a file", path ) );
            }

            // The deepest folder on the way that exists: nothing below it is opened by its whole path.
            ToolGrantDetail::HeldHandle folder;
            std::size_t next = names.size() - 1;

            for ( ;; --next )
            {
                const std::vector<std::string> on_the_way( names.begin(), names.begin() + static_cast<std::ptrdiff_t>( next ) );

                if ( const auto opened = GrantFileSystem::openExisting( fullPath( on_the_way ), false ) )
                {
                    folder = ToolGrantDetail::HeldHandle( *opened );

                    break;
                }

                if ( next == base )
                {
                    throw std::runtime_error( std::format( "no folder on the way to '{}' exists", path ) );
                }
            }

            if ( !GrantFileSystem::isFolder( folder.get() ) )
            {
                throw std::runtime_error( std::format( "'{}' lies under a file, not a folder", path ) );
            }

            requireWritable( folder.get(), path );

            for ( ; next + 1 < names.size(); ++next )
            {
                const std::filesystem::path name = ToolGrantDetail::pathFromUtf8( names[ next ] );
                std::optional<GrantFileSystem::NativeHandle> opened = GrantFileSystem::openChild( folder.get(), name, true, false );

                if ( !opened )
                {
                    if ( !create )
                    {
                        throw std::runtime_error( std::format( "there is no file '{}'", path ) );
                    }

                    opened = GrantFileSystem::createChild( folder.get(), name, true );
                }

                folder = ToolGrantDetail::HeldHandle( *opened );
                requireWritable( folder.get(), path );
            }

            const std::filesystem::path name = ToolGrantDetail::pathFromUtf8( names.back() );
            std::optional<GrantFileSystem::NativeHandle> opened = GrantFileSystem::openChild( folder.get(), name, false, true );

            if ( !opened )
            {
                if ( !create )
                {
                    throw std::runtime_error( std::format( "there is no file '{}'", path ) );
                }

                opened = GrantFileSystem::createChild( folder.get(), name, false );
            }

            ToolGrantDetail::HeldHandle file( *opened );
            requireWritable( file.get(), path );
            requireOneName( file.get(), path );

            return GrantFile( file.release(), true );
        }

        /**
         * @brief What is at a path: a file, a folder, or nothing.
         *
         * @throws std::invalid_argument for a path in a form the grant refuses.
         * @throws std::runtime_error when something is there and lies outside every granted folder.
         */
        [[nodiscard]] std::optional<FolderEntry> find( std::string_view path ) const
        {
            requireFolders( path );

            const std::vector<std::string> names = grantPathNames( path );
            const auto opened = GrantFileSystem::openExisting( fullPath( names ), false );

            if ( !opened )
            {
                return std::nullopt;
            }

            const ToolGrantDetail::HeldHandle held( *opened );
            requireReadable( held.get(), path );

            const bool is_folder = GrantFileSystem::isFolder( held.get() );

            return FolderEntry{ names.empty() ? std::string( "." ) : names.back(), is_folder, false,
                is_folder ? 0 : GrantFileSystem::size( held.get() ) };
        }

        /**
         * @brief The names in a folder.
         *
         * @throws std::invalid_argument for a path in a form the grant refuses.
         * @throws std::runtime_error when there is no folder there or it lies outside every granted folder.
         */
        [[nodiscard]] std::vector<FolderEntry> list( std::string_view path ) const
        {
            const ToolGrantDetail::HeldHandle folder = openChecked( path, true );

            return GrantFileSystem::list( folder.get() );
        }

        /**
         * @brief Every name under a folder, each with its path relative to it, folders entered as visit allows.
         *
         * A folder is entered through its own handle and only when its final path lies within a granted folder; a
         * link or junction leading outside is not entered, and nor is a folder that cannot be opened or listed.
         * Files are only listed: opening one is openFile's, with its own check.
         *
         * @return How many folders were not entered because they lead outside or could not be opened or listed.
         * @throws std::invalid_argument for a path in a form the grant refuses.
         * @throws std::runtime_error when there is no folder at the path or it lies outside every granted folder.
         */
        std::size_t walk( std::string_view path, const Visit& visit ) const
        {
            const ToolGrantDetail::HeldHandle folder = openChecked( path, true );
            std::size_t not_entered = 0;

            walkFolder( folder.get(), GrantFileSystem::list( folder.get() ), {}, visit, not_entered );

            return not_entered;
        }

        /**
         * @brief The transport a tool fetches through: each connection it makes is to a public address, or refused.
         *
         * @throws std::runtime_error when the grant has no network.
         */
        [[nodiscard]] std::shared_ptr<const Distribution::IHttpTransport> transport() const
        {
            if ( !network_ )
            {
                throw std::runtime_error( "this tool was granted no network" );
            }

            return network_;
        }

    private:

        static std::string shown( const std::filesystem::path& path )
        {
            const std::u8string text = path.u8string();

            return std::string( text.begin(), text.end() );
        }

        void requireFolders( std::string_view path ) const
        {
            if ( folders_.empty() )
            {
                throw std::runtime_error( std::format( "this tool was granted no folder, so it cannot open '{}'", path ) );
            }
        }

        /// A drive-letter path as named, or a relative one read against the first folder's final path.
        std::filesystem::path fullPath( const std::vector<std::string>& names ) const
        {
            std::filesystem::path full;
            std::size_t first = 0;

            if ( !names.empty() && ToolGrantDetail::isDrive( names.front() ) )
            {
                // \\?\ sends the path to the file system as written, so nothing checked here is read again.
                full = ToolGrantDetail::pathFromUtf8( "\\\\?\\" + names.front() + "\\" );
                first = 1;
            }
            else
            {
                full = final_paths_.front();
            }

            for ( std::size_t index = first; index < names.size(); ++index )
            {
                full /= ToolGrantDetail::pathFromUtf8( names[ index ] );
            }

            return full;
        }

        /// Whether a final path is a folder's own, or lies under it -- never merely shares its first characters.
        static bool isWithin( const std::filesystem::path& path, const std::filesystem::path& folder )
        {
            const auto& text = path.native();
            const auto& root = folder.native();
            constexpr auto separator = std::filesystem::path::preferred_separator;

            if ( !text.starts_with( root ) )
            {
                return false;
            }

            return text.size() == root.size() || ( !root.empty() && root.back() == separator ) || text[ root.size() ] == separator;
        }

        /// Whether a granted folder holds the path, and whether one granted read-write does.
        std::pair<bool, bool> holders( const std::filesystem::path& final_path ) const
        {
            bool readable = false;
            bool writable = false;

            for ( std::size_t index = 0; index < folders_.size(); ++index )
            {
                if ( isWithin( final_path, final_paths_[ index ] ) )
                {
                    readable = true;
                    writable = writable || folders_[ index ].access == FolderAccess::ReadWrite;
                }
            }

            return { readable, writable };
        }

        void requireReadable( GrantFileSystem::NativeHandle handle, std::string_view path ) const
        {
            if ( !holders( GrantFileSystem::finalPath( handle ) ).first )
            {
                throw std::runtime_error( std::format( "'{}' lies outside the folders this tool was granted", path ) );
            }
        }

        void requireWritable( GrantFileSystem::NativeHandle handle, std::string_view path ) const
        {
            const auto [readable, writable] = holders( GrantFileSystem::finalPath( handle ) );

            if ( !readable )
            {
                throw std::runtime_error( std::format( "'{}' lies outside the folders this tool was granted", path ) );
            }

            if ( !writable )
            {
                throw std::runtime_error( std::format( "'{}' lies in a folder this tool was granted only to read", path ) );
            }
        }

        static void requireOneName( GrantFileSystem::NativeHandle handle, std::string_view path )
        {
            if ( GrantFileSystem::linkCount( handle ) > 1 )
            {
                throw std::runtime_error( std::format(
                    "'{}' has more than one name on disk (a hard link), so where it lies cannot be told", path ) );
            }
        }

        /// Opens an existing file or folder by its path and checks it is inside, and is what was asked for.
        ToolGrantDetail::HeldHandle openChecked( std::string_view path, bool as_folder ) const
        {
            requireFolders( path );

            const std::vector<std::string> names = grantPathNames( path );
            const auto opened = GrantFileSystem::openExisting( fullPath( names ), false );

            if ( !opened )
            {
                throw std::runtime_error( std::format( "there is no {} '{}'", as_folder ? "folder" : "file", path ) );
            }

            ToolGrantDetail::HeldHandle held( *opened );
            requireReadable( held.get(), path );

            const bool is_folder = GrantFileSystem::isFolder( held.get() );

            if ( as_folder && !is_folder )
            {
                throw std::runtime_error( std::format( "'{}' is a file, not a folder", path ) );
            }

            if ( !as_folder && is_folder )
            {
                throw std::runtime_error( std::format( "'{}' is a folder, not a file", path ) );
            }

            if ( !as_folder )
            {
                requireOneName( held.get(), path );
            }

            return held;
        }

        /// @return False when a visit ended the walk.
        bool walkFolder( GrantFileSystem::NativeHandle folder, const std::vector<FolderEntry>& entries,
            const std::string& relative, const Visit& visit, std::size_t& not_entered ) const
        {
            for ( const FolderEntry& entry : entries )
            {
                const std::string path = relative.empty() ? entry.name : relative + "/" + entry.name;

                if ( !entry.is_folder )
                {
                    if ( !visit( path, entry ) )
                    {
                        return false;
                    }

                    continue;
                }

                ToolGrantDetail::HeldHandle child;
                std::vector<FolderEntry> child_entries;

                // A folder that cannot be opened or listed is passed over, as one leading outside is.
                try
                {
                    const auto opened = GrantFileSystem::openChild( folder, ToolGrantDetail::pathFromUtf8( entry.name ), true, false );

                    if ( !opened )
                    {
                        ++not_entered;

                        continue;
                    }

                    child = ToolGrantDetail::HeldHandle( *opened );

                    if ( !holders( GrantFileSystem::finalPath( child.get() ) ).first )
                    {
                        ++not_entered;

                        continue;
                    }

                    child_entries = GrantFileSystem::list( child.get() );
                }
                catch ( const std::runtime_error& )
                {
                    ++not_entered;

                    continue;
                }

                if ( visit( path, entry ) && !walkFolder( child.get(), child_entries, path, visit, not_entered ) )
                {
                    return false;
                }
            }

            return true;
        }

        std::vector<GrantFolder> folders_;

        // Each folder's own final path, taken when the grant was made, in the same order.
        std::vector<std::filesystem::path> final_paths_;

        std::shared_ptr<const Distribution::IHttpTransport> network_;
    };
}
