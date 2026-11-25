/**
 * @file TransferInProgressError.ixx
 * @brief A transfer refused because another process is writing the same blob into the store.
 */

export module Distribution.TransferInProgressError;

import std;

namespace Mila::Distribution
{
    /**
     * @brief Thrown when another live process holds the blob's transfer lock, so that a program can tell this case
     *        from a failed transfer and say it in its own words; the bytes that process writes are the same, and the
     *        next attempt after it finishes finds the blob in place.
     */
    export class TransferInProgressError : public std::runtime_error
    {
    public:

        explicit TransferInProgressError( std::string description )
            : std::runtime_error( std::format( "ModelStore: another process is already writing {} into the store", description ) ),
            description_( std::move( description ) )
        {
        }

        /// What the blob is, as the caller described it: typically its file and the model it came from.
        [[nodiscard]] const std::string& description() const noexcept
        {
            return description_;
        }

    private:

        std::string description_;
    };
}
