/**
 * @file Mila_py.Wrappers.ModelStoreHandle.ixx
 * @brief The local model store, and the HTTP transport a caller supplies for pull.
 */

export module Mila.Bindings:ModelStoreHandle;

import std;
import :StoredModelInfo;
import :StoreUsageInfo;
import :RemovalReportInfo;
import :HubModelInfo;
import :HttpHeaderInfo;
import :HttpResponseInfo;

namespace Mila::Bindings
{
    /// Receives body bytes. Return false to abort the transfer.
    export using ChunkSink = std::function<bool( const char* data, std::size_t length )>;

    /// Called once when the status and headers are known, before any body.
    export using HeadersInfoCallback =
        std::function<void( long http_code, std::uint64_t content_length )>;

    /**
     * @brief One HTTP GET, supplied by the caller.
     *
     * The seam is the transport, and it is deliberately below everything that requires
     * judgement. Which URL to ask for, where the token lives, whether to follow a redirect,
     * what may be sent to the next host, what a 403 means, when a Range was ignored -- all of
     * that is compiled into the library whether or not it has an HTTP client. So a host
     * language supplies bytes and reimplements nothing.
     *
     * One callable rather than an abstract class, so Python implements it without a
     * trampoline. Bytes handed to the sink are hashed and written in one pass, never staged
     * and re-read.
     *
     * **The delegate is not trusted with anything.** It is handed the exact headers to send --
     * already decided to be safe for this exact host -- and must not follow redirects, add
     * headers, or deliver a non-2xx body to the sink. It never sees a token it should not
     * send, because the library never puts one in a header bound for the wrong host.
     */
    export using HttpFetchDelegate = std::function<HttpResponseInfo(
        const std::string& url,
        const std::vector<HttpHeaderInfo>& headers,
        const ChunkSink& sink )>;

    /// The hub owner Mila publishes under. Passed by the consumer, never baked into a hub.
    export std::string defaultHubOwner();

    /**
     * @brief The local model store: the only thing a load reads from.
     *
     * Pull and load are separate verbs. Nothing here reaches a network except pull(), and pull
     * reaches only as far as the delegate it is handed.
     */
    export class ModelStoreHandle
    {
    public:
        /// Empty root takes MILA_CACHE_DIR, then the platform user cache.
        explicit ModelStoreHandle( const std::string& root = {} );

        std::string root() const;

        std::vector<StoredModelInfo> list() const;

        /// Nullopt when no such model is installed, or when a declared blob is missing.
        std::optional<StoredModelInfo> locate( const std::string& name ) const;

        RemovalReportInfo remove( const std::string& name );

        StoreUsageInfo usage() const;

        /// Install a package directory. Moving is free on one volume and leaves no second copy.
        StoredModelInfo install(
            const std::string& package_directory,
            const std::string& name = {},
            bool replace = false,
            bool move_files = true );

        /**
         * @brief Pull a published model, fetching every file its manifest declares.
         *
         * The library builds the HuggingFace URLs, discovers the token, parses the manifest
         * and verifies each digest; the delegate only moves bytes. An empty delegate uses
         * whichever transport this build was compiled with.
         *
         * @throws std::runtime_error if the name is path-shaped, the manifest is malformed, a
         *         digest does not match, or the model requires a newer Mila.
         */
        StoredModelInfo pull(
            const std::string& name,
            const std::string& owner,
            const HttpFetchDelegate& transport = {} );

        /// Every repository an owner publishes, marked with what this store already holds.
        std::vector<HubModelInfo> listHubModels(
            const std::string& owner,
            const HttpFetchDelegate& transport = {} ) const;

        ~ModelStoreHandle();

    private:
        struct Impl;

        std::unique_ptr<Impl> impl_;
    };
}
