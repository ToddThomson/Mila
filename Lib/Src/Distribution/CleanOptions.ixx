/**
 * @file CleanOptions.ixx
 * @brief What cleaning the store may reclaim beyond rejected transfers and abandoned locks.
 */

module;

export module Distribution.CleanOptions;

namespace Mila::Distribution
{
    export struct CleanOptions
    {
        /**
         * @brief Also discard in-flight partials.
         *
         * Off by default: a partial left by a failed transfer is good bytes that the next attempt
         * resumes onto, so deleting it silently converts a cheap retry into a full re-download.
         */
        bool discard_partials{ false };
    };
}
