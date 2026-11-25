/**
 * @file TransferLock.cpp
 * @brief The transfer lock's bodies, each platform by its own calls. The interface is Store/TransferLock.ixx.
 */

module;

#if defined( _WIN32 )
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

module Distribution.TransferLock;

import std;

namespace Mila::Distribution
{
#if defined( _WIN32 )

    TransferLock::TransferLock( std::filesystem::path path )
        : path_( std::move( path ) )
    {
        // No sharing: a second open fails while this handle lives, and the system closes it when the process ends,
        // however it ends. Delete-on-close then removes the file, so nothing is left to mistake for a holder.
        const HANDLE file = ::CreateFileW( path_.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr );

        if ( file == INVALID_HANDLE_VALUE )
        {
            return;
        }

        handle_ = reinterpret_cast<std::intptr_t>( file );
        held_ = true;
    }

    TransferLock::~TransferLock()
    {
        if ( held_ )
        {
            ::CloseHandle( reinterpret_cast<HANDLE>( handle_ ) );
        }
    }

#else

    namespace
    {
        bool sameFile( int descriptor, const std::filesystem::path& path )
        {
            struct stat opened {};
            struct stat named {};

            return ::fstat( descriptor, &opened ) == 0 && ::stat( path.c_str(), &named ) == 0
                && opened.st_dev == named.st_dev && opened.st_ino == named.st_ino;
        }
    }

    TransferLock::TransferLock( std::filesystem::path path )
        : path_( std::move( path ) )
    {
        // A holder unlinks the file before it lets go, so a lock taken on a file no longer at the path was taken on one
        // just released: open the path again. Two tries cover a release between the open and the lock; a third
        // would mean the path keeps changing under us, which is a holder's to say.
        for ( int attempt = 0; attempt < 3; ++attempt )
        {
            const int descriptor = ::open( path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644 );

            if ( descriptor < 0 )
            {
                return;
            }

            if ( ::flock( descriptor, LOCK_EX | LOCK_NB ) != 0 )
            {
                ::close( descriptor );

                return;
            }

            if ( sameFile( descriptor, path_ ) )
            {
                handle_ = descriptor;
                held_ = true;

                return;
            }

            ::close( descriptor );
        }
    }

    TransferLock::~TransferLock()
    {
        if ( held_ )
        {
            // Unlinked while still held, so no process can lock this file after it is released.
            ::unlink( path_.c_str() );
            ::close( static_cast<int>( handle_ ) );
        }
    }

#endif
}
