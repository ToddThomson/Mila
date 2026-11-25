/**
 * @file TransferLock.ixx
 * @brief The lock that lets one process at a time write a blob's transfer into the store, held by the operating system
 *        so that it ends with the process however the process ends. The bodies are Implementation/TransferLock.cpp.
 */

export module Distribution.TransferLock;

import std;

namespace Mila::Distribution
{
    /**
     * @brief A lock on a file in the store's tmp/, taken without waiting and released on destruction.
     *
     * The operating system holds it, not the file's existence: on Windows the file is opened with no sharing and
     * deleted when its handle closes, and on macOS and Linux it is `flock`ed and unlinked by its holder before the
     * descriptor closes. A process that ends without unwinding therefore frees it, and a lock file left by one that
     * held an earlier kind of lock is taken over rather than refused.
     */
    export class TransferLock
    {
    public:

        /// Tries once; `held()` says whether this process now holds it.
        explicit TransferLock( std::filesystem::path path );

        ~TransferLock();

        TransferLock( const TransferLock& ) = delete;
        TransferLock& operator=( const TransferLock& ) = delete;

        /// False when a live process holds it, or the file could not be opened.
        [[nodiscard]] bool held() const noexcept { return held_; }

        [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    private:

        std::filesystem::path path_;

        /// The open file: a Windows HANDLE or a POSIX descriptor, by platform.
        std::intptr_t handle_{ -1 };
        bool held_{ false };
    };
}
