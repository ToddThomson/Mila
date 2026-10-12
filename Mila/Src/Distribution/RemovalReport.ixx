/**
 * @file RemovalReport.ixx
 * @brief What removing a model, or cleaning the store, deleted and kept.
 */

module;
#include <cstdint>
#include <string>
#include <vector>

export module Distribution.RemovalReport;

namespace Mila::Distribution
{
    export struct RemovalReport
    {
        /// Records unlinked.
        int records_removed{ 0 };

        /// Blobs the removed model named and no other record does.
        int blobs_removed{ 0 };

        /// Rejected transfers and abandoned locks, which belong to no record at all.
        int files_removed{ 0 };

        uint64_t bytes_reclaimed{ 0 };

        /// Paths the platform refused to delete, most often a blob still mapped by a live process.
        std::vector<std::string> retained;

        /// Record files that could not be read. While any exist no blob is deleted, since one of
        /// them may name it.
        std::vector<std::string> unreadable_records;
    };
}
