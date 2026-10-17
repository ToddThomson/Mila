/**
 * @file Mila_py.Wrappers.StoreUsageInfo.ixx
 * @brief What the model store holds on disk.
 */

export module Mila.Bindings:StoreUsageInfo;

import std;

namespace Mila::Bindings
{
    export struct StoreUsageInfo
    {
        std::uint64_t blob_bytes{ 0 };
        std::uint64_t reclaimable_bytes{ 0 };
        std::uint64_t partial_bytes{ 0 };
        int model_count{ 0 };
        int blob_count{ 0 };
    };
}
