/**
 * @file FetchReport.ixx
 * @brief What one fetch attempt reported, and the shape of the callable that makes one.
 */

export module Distribution.FetchReport;

import std;
export import Distribution.FetchOutcome;

namespace Mila::Distribution
{
    export struct FetchReport
    {
        FetchOutcome outcome{ FetchOutcome::Failed };
        std::string message;
    };

    /**
     * @brief Fetches one blob from resume_from, handing each chunk to sink.
     *
     * The store supplies the sink so it can hash and write in one pass. What is being fetched
     * is the fetcher's business: the store deals in digests, and a URL scheme belongs to
     * whichever hub produced the fetcher.
     */
    export using BlobFetcher = std::function<FetchReport(
        std::uint64_t resume_from,
        const std::function<bool( const char*, std::size_t )>& sink )>;
}
