/**
 * @file MemoryHold.ixx
 * @brief How firmly a device keeps the memory Mila has allocated on it.
 */

export module Compute.MemoryHold;

import std;

namespace Mila::Dnn::Compute
{
    /**
     * @brief How firmly a device keeps Mila's memory while it is not in use.
     *
     * Only a device whose memory the operating system can page or hand to another application has anything to keep; on
     * CUDA every hold is the same. On a Mac (Specifications/Metal/MetalBackend.md section 5, W3d):
     */
    export enum class MemoryHold
    {
        /// Nothing held. Between uses the memory can be compressed or paged out, and the next use brings it back.
        None,

        /// Mila's process locks its pages: they stay in memory, and other applications' GPU work still runs. A use after
        /// a few seconds idle pays about 0.1 s for the GPU's own setup.
        Locked,

        /// Locked, and the GPU is kept ready by a residency request renewed every second: every use starts at once, and
        /// another application's GPU work that does not fit beside it fails.
        LockedAndRequested,
    };

    export [[nodiscard]] constexpr std::string_view toString( MemoryHold hold ) noexcept
    {
        switch ( hold )
        {
            case MemoryHold::None:
                return "none";
            case MemoryHold::Locked:
                return "locked";
            case MemoryHold::LockedAndRequested:
                return "locked and requested";
        }

        return "unknown";
    }
}
