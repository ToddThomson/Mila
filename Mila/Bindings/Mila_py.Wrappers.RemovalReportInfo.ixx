/**
 * @file Mila_py.Wrappers.RemovalReportInfo.ixx
 * @brief What removing a model from the store reclaimed.
 */

export module Mila.Bindings:RemovalReportInfo;

import std;

namespace Mila::Bindings
{
    export struct RemovalReportInfo
    {
        int records_removed{ 0 };
        int blobs_removed{ 0 };
        int files_removed{ 0 };
        std::uint64_t bytes_reclaimed{ 0 };

        /// Paths the platform refused to delete, most often a blob a live process still maps.
        std::vector<std::string> retained;

        /// Record files that could not be read; while any exist, no blob is deleted.
        std::vector<std::string> unreadable_records;
    };
}
