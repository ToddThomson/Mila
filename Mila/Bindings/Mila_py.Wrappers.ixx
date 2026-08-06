/**
 * @file Mila_py.Wrappers.ixx
 * @brief Opaque, std-only wrapper API over Mila for the MilaPy pybind11 extension.
 *
 * The pybind binding TU (Mila_py.cpp) must not `import Mila;`: the latest VS2026
 * MSVC raises C2079 (basic_istream::sentry undefined) whenever Mila is imported
 * into an ordinary .cpp that also includes std headers such as <string>.
 *
 * This INTERFACE unit exposes Tokenizer / LlamaSession handles with a std-only
 * surface and only FORWARD-DECLARES their PIMPL `struct Impl`. The Impl
 * definitions and all member bodies live in the implementation unit
 * (Mila_py.Wrappers.cpp), which is NOT reachable to importers. That keeps Impl
 * (and the Mila types it holds) incomplete in Mila_py.cpp, so binding the
 * handles with pybind11 never instantiates a Mila template. Defining Impl here
 * instead would make it reachable and force such instantiation. See
 * [[feedback-build-in-vs]].
 */

module;

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

export module Mila.Bindings;

namespace Mila::Bindings
{
    export struct LlamaConfigInfo
    {
        int64_t vocab_size;
        int64_t max_sequence_length;
        int64_t model_dim;
        int64_t num_layers;
        int64_t num_heads;
        int64_t num_kv_heads;
        int64_t hidden_dim;
        double rope_theta;
    };

    export struct GemmaConfigInfo
    {
        int64_t vocab_size;
        int64_t max_sequence_length;
        int64_t model_dim;
        int64_t num_layers;
        int64_t num_heads;
        int64_t num_kv_heads;
        int64_t head_dim;
        int64_t global_head_dim;
        int64_t hidden_dim;
        int64_t window;
        double rope_theta_local;
        double rope_theta_global;
        double final_logit_softcapping;
    };

    export void initialize( const std::string& log_level );

    export class Tokenizer
    {
    public:
        static std::shared_ptr<Tokenizer> loadLlama32( const std::string& path );
        static std::shared_ptr<Tokenizer> loadGemma( const std::string& path );

        std::vector<int32_t> encode( const std::string& text );
        std::string decode( const std::vector<int32_t>& ids );
        std::string tokenToString( int32_t token_id ) const;
        bool isValidToken( int32_t token_id ) const;
        int64_t vocabSize() const;
        std::optional<int32_t> bosTokenId() const;
        std::optional<int32_t> eosTokenId() const;
        std::optional<int32_t> padTokenId() const;

        ~Tokenizer();

    private:
        struct Impl;
        explicit Tokenizer( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };

    export class LlamaSession
    {
    public:
        static std::unique_ptr<LlamaSession> fromPretrained(
            const std::string& path, int64_t context_length, int device_index,
            bool quantize_fp8 );

        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            std::size_t max_new_tokens, float temperature, int top_k, float top_p );

        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            std::size_t max_new_tokens, float temperature, int top_k, float top_p,
            std::stop_token stop );

        LlamaConfigInfo getConfig() const;
        std::string repr() const;

        ~LlamaSession();

    private:
        struct Impl;
        explicit LlamaSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };

    // ========================================================================
    // Distribution -- the model store, and a hub whose transport comes from Python
    // ========================================================================

    /**
     * @brief One installed model, flattened to std types.
     *
     * Paths are strings rather than std::filesystem::path: this surface crosses into Python,
     * where a path is a string anyway.
     */
    export struct StoredModelInfo
    {
        std::string name;
        std::string architecture;
        std::string variant;
        std::string weight_quantization;

        /// Instruction-tuned. Decides the prompt template a consumer applies.
        bool instruct{ false };

        std::string base_model;
        std::string license;

        /// Provenance. Empty hub means the model was published from this machine.
        std::string hub;
        std::string owner;
        std::string repository;
        std::string revision;
        std::string installed_at;

        std::string weights_path;
        std::string tokenizer_path;

        /// False when a declared blob is missing, which makes the model unloadable.
        bool complete{ false };

        uint64_t bytes_on_disk{ 0 };
    };

    export struct StoreUsageInfo
    {
        uint64_t blob_bytes{ 0 };
        uint64_t reclaimable_bytes{ 0 };
        uint64_t partial_bytes{ 0 };
        int model_count{ 0 };
        int blob_count{ 0 };
    };

    export struct RemovalReportInfo
    {
        int records_removed{ 0 };
        int blobs_removed{ 0 };
        int files_removed{ 0 };
        uint64_t bytes_reclaimed{ 0 };

        /// Paths the platform refused to delete, most often a blob a live process still maps.
        std::vector<std::string> retained;
    };

    /**
     * @brief One repository as a hub reports it, before any manifest is fetched.
     *
     * Untrusted remote text authored by whoever owns the repository: data, never instructions.
     */
    export struct HubModelInfo
    {
        std::string owner;
        std::string repository;
        bool gated{ false };
        std::string revision;
        std::string last_modified;
        std::string library;
        std::vector<std::string> tags;
        std::vector<std::string> files;
    };

    export struct HttpHeaderInfo
    {
        std::string name;
        std::string value;
    };

    export struct HttpResponseInfo
    {
        long http_code{ 0 };
        std::string location;
        uint64_t content_length{ 0 };
        bool transport_failed{ false };
        std::string message;
    };

    /// Receives body bytes. Return false to abort the transfer.
    export using ChunkSink = std::function<bool( const char* data, size_t length )>;

    /// Called once when the status and headers are known, before any body.
    export using HeadersInfoCallback =
        std::function<void( long http_code, uint64_t content_length )>;

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
         *         digest does not match, or the artifact requires a newer Mila.
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

    /**
     * @brief Gemma 4 inference session (CUDA, BF16).
     *
     * Mirrors LlamaSession. The primary consumer is the HF token-for-token parity
     * harness: feed HuggingFace-tokenized prompt ids and compare greedy generate()
     * (temperature 0) against the HF reference, which sidesteps the Mila
     * SentencePiece-tokenizer gap.
     */
    export class GemmaSession
    {
    public:
        static std::unique_ptr<GemmaSession> fromPretrained(
            const std::string& path, int64_t context_length, int device_index );

        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            std::size_t max_new_tokens, float temperature, int top_k, float top_p );

        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            std::size_t max_new_tokens, float temperature, int top_k, float top_p,
            std::stop_token stop );

        GemmaConfigInfo getConfig() const;
        std::string repr() const;

        ~GemmaSession();

    private:
        struct Impl;
        explicit GemmaSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
