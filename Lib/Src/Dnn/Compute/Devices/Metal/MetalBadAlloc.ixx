/**
 * @file MetalBadAlloc.ixx
 * @brief The allocation failure of a Metal buffer, with the sizes that caused it.
 */

export module Compute.MetalBadAlloc;

import std;

namespace Mila::Dnn::Compute
{
    /// A `std::bad_alloc`, as every allocation failure in Mila is, that says what was asked for.
    export class MetalBadAlloc : public std::bad_alloc
    {
    public:

        explicit MetalBadAlloc( std::string message )
            : message_( std::move( message ) )
        {
        }

        [[nodiscard]] const char* what() const noexcept override
        {
            return message_.c_str();
        }

    private:

        std::string message_;
    };
}
