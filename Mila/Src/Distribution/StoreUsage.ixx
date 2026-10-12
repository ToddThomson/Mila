/**
 * @file StoreUsage.ixx
 * @brief What the store holds and what could be reclaimed.
 */

module;
#include <cstdint>

export module Distribution.StoreUsage;

namespace Mila::Distribution
{
    export struct StoreUsage
    {
        uint64_t blob_bytes{ 0 };

        /// What clean() would free: rejected transfers and abandoned locks.
        uint64_t reclaimable_bytes{ 0 };
        uint64_t partial_bytes{ 0 };
        int model_count{ 0 };
        int blob_count{ 0 };
    };
}
