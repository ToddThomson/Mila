/**
 * @file GrantFile.ixx
 * @brief A file opened through a grant: read and written through the handle that was checked.
 */

export module AI.GrantFile;

import std;
import AI.GrantFileSystem;

namespace Mila::AI
{
    /**
     * @brief One file a grant opened and checked (ToolCalling.md 3.4).
     *
     * Every read and write goes through the handle whose final path the grant checked, so a name changed or a
     * link swapped since then changes nothing. Move-only; closes the file when destroyed.
     */
    export class GrantFile
    {
    public:

        /// Made by a grant, which owns the check; takes ownership of the handle.
        GrantFile( GrantFileSystem::NativeHandle handle, bool writable ) noexcept
            : handle_( handle ), writable_( writable )
        {
        }

        GrantFile( GrantFile&& other ) noexcept
            : handle_( std::exchange( other.handle_, GrantFileSystem::kNoHandle ) ), writable_( other.writable_ )
        {
        }

        GrantFile& operator=( GrantFile&& other ) noexcept
        {
            if ( this != &other )
            {
                GrantFileSystem::close( handle_ );
                handle_ = std::exchange( other.handle_, GrantFileSystem::kNoHandle );
                writable_ = other.writable_;
            }

            return *this;
        }

        GrantFile( const GrantFile& ) = delete;
        GrantFile& operator=( const GrantFile& ) = delete;

        ~GrantFile()
        {
            GrantFileSystem::close( handle_ );
        }

        [[nodiscard]] std::uint64_t size() const
        {
            return GrantFileSystem::size( handle_ );
        }

        /// The file's bytes from its start, up to the most asked for.
        [[nodiscard]] std::string read( std::uint64_t most = std::numeric_limits<std::uint64_t>::max() ) const
        {
            return GrantFileSystem::read( handle_, most );
        }

        /**
         * @brief Replaces the file's whole content.
         *
         * @throws std::logic_error when the grant opened the file only to read.
         */
        void replace( std::string_view bytes )
        {
            if ( !writable_ )
            {
                throw std::logic_error( "the file was opened only to read" );
            }

            GrantFileSystem::replace( handle_, bytes );
        }

        [[nodiscard]] bool writable() const noexcept
        {
            return writable_;
        }

    private:

        GrantFileSystem::NativeHandle handle_;
        bool writable_;
    };
}
