/**
 * @file DeviceOutOfMemoryError.ixx
 * @brief Work the GPU could not run because its memory was taken, by Mila or by another application.
 */

export module Compute.DeviceOutOfMemoryError;

import std;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The GPU refused work for want of memory after the work was handed to it.
     *
     * Not an allocation failure, which is `std::bad_alloc`: the memory was Mila's, and the GPU could not hold it while the
     * work ran, which on a Mac happens when another application holds the GPU's memory. The work did not finish, and
     * nothing it wrote can be trusted; the model stays loaded, and the same work can be tried again once the memory is free.
     */
    export class DeviceOutOfMemoryError : public std::runtime_error
    {
    public:

        explicit DeviceOutOfMemoryError( const std::string& message )
            : std::runtime_error( message )
        {
        }
    };
}
