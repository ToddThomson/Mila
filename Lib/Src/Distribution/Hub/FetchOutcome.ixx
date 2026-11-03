/**
 * @file FetchOutcome.ixx
 * @brief The four outcomes of a fetch attempt that the store acts on.
 */

module;

export module Distribution.FetchOutcome;

namespace Mila::Distribution
{
    /**
     * @brief What a fetch attempt reported back to the store.
     *
     * Narrower than HttpResult: the store cares about four outcomes, and keeping it decoupled from the HTTP
     * client is what makes it testable offline.
     */
    export enum class FetchOutcome
    {
        Complete,      ///< The requested byte range arrived in full.
        RangeIgnored,  ///< A resume was requested and the server sent the whole file.

        /// A resume was requested from at or past the end, so nothing was sent and the partial
        /// already holds every byte there is. Whether they are the right bytes is the digest's
        /// question, not the transport's.
        RangeNotSatisfiable,

        Failed         ///< Anything else; message carries the detail.
    };
}
