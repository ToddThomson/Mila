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

    /**
     * @brief Qwen 3.8's shape, including the axes no other family in this binding has.
     *
     * The DeltaNet fields describe the recurrent mixer that occupies three of every four
     * layers, and full_attention_interval is what says how often the attention layer falls.
     * They are reported rather than derived, because nothing outside the checkpoint knows them.
     */
    export struct QwenConfigInfo
    {
        int64_t vocab_size;
        int64_t max_sequence_length;
        int64_t model_dim;
        int64_t num_layers;
        int64_t num_heads;
        int64_t num_kv_heads;
        int64_t head_dim;
        int64_t hidden_dim;
        double rope_theta;
        double partial_rotary_factor;
        bool attention_output_gate;
        int64_t full_attention_interval;
        int64_t linear_num_key_heads;
        int64_t linear_num_value_heads;
        int64_t linear_head_dim;
        int64_t linear_conv_kernel_dim;
    };

    export void initialize( const std::string& log_level );

    // ========================================================================
    // Chat protocol -- the model's own grammar, projected rather than reimplemented
    // ========================================================================

    /**
     * @brief One call the model asked for.
     *
     * Arguments are a JSON object as text, which is what the model wrote and what a host hands
     * on. Parsing it into a typed value here could only lose something.
     *
     * No id: no template in the library renders one, so correlating a call with its result is
     * the host's business and a host that needs an id mints it where it is needed.
     */
    export struct ToolCallInfo
    {
        std::string name;
        std::string arguments;
    };

    /**
     * @brief One turn of a conversation, in the shape a host already holds it.
     *
     * Role is a string -- "system", "user", "assistant", "tool" -- rather than an enum, because
     * every wire protocol a host speaks already carries it that way. An enum would make Python
     * translate twice for no gain.
     */
    export struct TurnInfo
    {
        std::string role;
        std::string content;
        std::vector<ToolCallInfo> tool_calls;
    };

    /**
     * @brief The control tokens a family's grammar is built from.
     *
     * Exposed because a host that streams has to recognise them as they arrive -- generation
     * stops at a closing tool-call marker, or the model fabricates the result itself. Reported
     * from the runtime rather than written down again, which is the whole point of this section.
     */
    export struct ProtocolTokens
    {
        std::string turn_open;
        std::string turn_close;
        std::string reasoning_open;
        std::string reasoning_close;
        std::string tool_call_open;
        std::string tool_call_close;
        std::string tool_response_open;
        std::string tool_response_close;
    };

    /**
     * @brief Render a Qwen 3.8 conversation into the prompt its checkpoint was trained on.
     *
     * The whole template is here rather than in the caller: turn structure, the reasoning gate,
     * the ordering of the system turn's parts, and the tools section's exact wording. A host
     * supplies a conversation and gets a prompt, and reimplements nothing.
     *
     * @param reasoning_effort_scale 1..5. Mapped onto the three levels the checkpoint knows,
     *        whose middle level deliberately emits no instruction at all.
     * @param tools_json A JSON array of tool signature objects, or empty for none. Empty omits
     *        the tools section, and that absence is what tells the model there are none.
     *
     * @throws std::runtime_error if a role is not one of the four, or if the history is empty
     *         or ends on an assistant turn.
     */
    export std::string qwenFormatPrompt(
        const std::vector<TurnInfo>& history,
        bool enable_thinking,
        int reasoning_effort_scale,
        const std::string& tools_json );

    /**
     * @brief The first tool call in a Qwen response, or nothing when it holds none.
     *
     * @throws std::runtime_error if the span holds something that is not a call -- the model
     *         failing at its own protocol, which is worth surfacing rather than reading as prose.
     */
    export std::optional<ToolCallInfo> qwenParseToolCall( const std::string& response );

    /// Qwen's control tokens, as the checkpoint vocabulary registers them.
    export ProtocolTokens qwenProtocolTokens();

    /**
     * @brief Render a Gemma 4 conversation into the prompt its checkpoint was trained on.
     *
     * The whole template is here rather than in the caller: turn structure, the role spellings,
     * the empty-thought prime that opens a fresh model turn, and where declarations attach to
     * the system turn. A host supplies a conversation and gets a prompt.
     *
     * @param tool_declarations What gemmaToolDeclarations returns, or empty for none.
     * @param continue_open Emit the final turn OPEN so the next token continues it, which is
     *        the shape after a tool response. No thought prime is emitted in that case -- the
     *        turn already carries its channel, and a second one mid-turn is off-distribution.
     *
     * @throws std::runtime_error if a role is not one of the four, or if continue_open is set
     *         on an empty history.
     */
    export std::string gemmaFormatPrompt(
        const std::vector<TurnInfo>& history,
        const std::string& tool_declarations,
        bool continue_open );

    /**
     * @brief Tool schemas rendered in Gemma's trained <|tool>declaration:...<tool|> grammar.
     *
     * @param tools_json A JSON array of tool schemas, OpenAI function envelopes or bare
     *        declarations. Which tools to advertise is the host's choice: a harness with
     *        UI-only tools filters them out before calling.
     */
    export std::string gemmaToolDeclarations( const std::string& tools_json );

    /// The most recent tool call in a Gemma response, or nothing when it holds none.
    export std::optional<ToolCallInfo> gemmaParseToolCall( const std::string& response );

    /// One assistant tool call rendered back into the native call grammar, for replay.
    export std::string gemmaFormatToolCall( const std::string& name, const std::string& arguments );

    /**
     * @brief A client-executed tool result in Gemma's <|tool_response> grammar.
     *
     * A JSON envelope surfaces only its primary output field; metadata siblings are dropped, and
     * a failed tool's `error` is surfaced explicitly so the model does not blind-retry.
     */
    export std::string gemmaFormatToolResponse( const std::string& name, const std::string& result );

    /// A channel-structured response reduced to the user-facing answer.
    export std::string gemmaExtractAnswer( const std::string& text );

    /// Every registered control token removed from decoded text.
    export std::string gemmaStripControlTokens( const std::string& text );

    /// Gemma's control tokens, as the checkpoint vocabulary registers them.
    export ProtocolTokens gemmaProtocolTokens();

    export class Tokenizer
    {
    public:
        static std::shared_ptr<Tokenizer> loadLlama32( const std::string& path );
        static std::shared_ptr<Tokenizer> loadGemma( const std::string& path );
        static std::shared_ptr<Tokenizer> loadQwen( const std::string& path );

        /**
         * @brief The tokenizer of an installed model, by store name.
         *
         * Which loader to use is a property of the weights, not of the caller, so the
         * record decides it. That is what removes the pairing a consumer previously had to
         * keep correct by hand: a tokenizer path and a weights path that had to describe
         * the same model, with nothing checking that they did.
         *
         * @throws std::runtime_error if no such model is installed, its files are missing,
         *         or its architecture has no tokenizer in this binding.
         */
        static std::shared_ptr<Tokenizer> fromStore( const std::string& name );

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

    // Quantization is named the way the store names it -- "bf16", "fp8", "fp4", and for Qwen
    // "cb2-3" -- by both session entry points, so a variant means the same thing
    // whether it came from a record or from a caller. "fp32" is rejected rather than ignored:
    // these sessions are BF16 instantiations, and loading FP32 weights at BF16 is a
    // different model than the one asked for.

    export class LlamaSession
    {
    public:
        /**
         * @param quantization Applied on the way in, against unquantized weights.
         *        Pre-quantized weights must be loaded through fromStore, which reads what
         *        their bytes already are.
         */
        static std::unique_ptr<LlamaSession> fromPretrained(
            const std::string& path, int64_t context_length, int device_index,
            const std::string& quantization );

        /**
         * @brief Load an installed model by store name, as its weights already are.
         *
         * The record decides the quantization, which is the whole point: a published
         * model is already FP4 or FP8 bytes, and a caller-supplied flag could only agree
         * with them by luck. Nothing here reaches a network -- an uninstalled name is an
         * error, never a download.
         *
         * @throws std::runtime_error if no such model is installed, its files are missing,
         *         or its architecture or variant is not one this session can load.
         */
        static std::unique_ptr<LlamaSession> fromStore(
            const std::string& name, int64_t context_length, int device_index );

        /**
         * @brief Generate from a prompt, streaming each token through on_token.
         *
         * The same shape as the library's own LanguageModel::generate: tokens leave
         * through the callback and the return value is why generation stopped, which
         * is the one outcome a caller cannot reconstruct from the token stream. The
         * status crosses as its wire spelling -- "stop", "length", "context_limit" or
         * "cancelled" -- so no enum has to cross with it.
         */
        std::string generate(
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
        /**
         * @param quantization Applied on the way in, against unquantized weights.
         *        Defaults to FP4 at the binding layer rather than to none: a BF16 Gemma 4
         *        12B needs ~24 GB and would OOM at load on the cards this targets.
         */
        static std::unique_ptr<GemmaSession> fromPretrained(
            const std::string& path, int64_t context_length, int device_index,
            const std::string& quantization );

        /// As LlamaSession::fromStore -- the record decides the quantization.
        static std::unique_ptr<GemmaSession> fromStore(
            const std::string& name, int64_t context_length, int device_index );

        /// As LlamaSession::generate -- tokens through the callback, why it stopped returned.
        std::string generate(
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

    /**
     * @brief Qwen 3.8 inference session (CUDA, BF16).
     *
     * Mirrors GemmaSession. Two deployments are published -- a per-group FP4 build and a
     * codebook build that spends 2 and 3 bits per weight across the body -- and both are
     * pre-quantized, so fromStore is the entry point that matters.
     */
    export class QwenSession
    {
    public:
        /**
         * @param quantization Defaults to FP4 at the binding layer for the same reason
         *        GemmaSession does: a BF16 27B does not fit any card this targets.
         *        "cb2-3" names a plan fitted offline, so it selects packed
         *        weights' format rather than applying anything on the way in.
         */
        static std::unique_ptr<QwenSession> fromPretrained(
            const std::string& path, int64_t context_length, int device_index,
            const std::string& quantization );

        /// As GemmaSession::fromStore -- the record decides the quantization.
        static std::unique_ptr<QwenSession> fromStore(
            const std::string& name, int64_t context_length, int device_index );

        /// As LlamaSession::generate -- tokens through the callback, why it stopped returned.
        std::string generate(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            std::size_t max_new_tokens, float temperature, int top_k, float top_p,
            std::stop_token stop );

        QwenConfigInfo getConfig() const;
        std::string repr() const;

        ~QwenSession();

    private:
        struct Impl;
        explicit QwenSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
