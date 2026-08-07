/**
 * @file Mila_py.cpp
 * @brief pybind11 entry point for the MilaPy extension.
 *
 * This TU deliberately does NOT `import Mila;`. The latest VS2026 MSVC raises
 * C2079 (basic_istream::sentry undefined) whenever Mila is imported into an
 * ordinary .cpp alongside std includes such as <string>. All Mila access goes
 * through the std-only opaque handles exported by Mila.Bindings. See
 * [[feedback-build-in-vs]].
 */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

import Mila.Bindings;

namespace py = pybind11;

using Mila::Bindings::GemmaConfigInfo;
using Mila::Bindings::GemmaSession;
using Mila::Bindings::LlamaConfigInfo;
using Mila::Bindings::LlamaSession;
using Mila::Bindings::Tokenizer;

// ============================================================================
// StopController — exposes std::stop_source to Python
// ============================================================================

/**
 * @brief Python-visible handle for cooperative generation cancellation.
 *
 * Construct one instance per request; pass to generate_streaming() and call
 * request_stop() on client disconnect. A single instance must not be shared
 * across concurrent requests.
 */
class StopController
{
public:
    void request_stop() { source_.request_stop(); }
    bool stop_requested() const { return source_.get_token().stop_requested(); }
    std::stop_token get_token() { return source_.get_token(); }

private:
    std::stop_source source_;
};

// ============================================================================
// BpeTokenizer bindings
// ============================================================================

static void bind_tokenizer( py::module_& m )
{
    py::class_<Tokenizer, std::shared_ptr<Tokenizer>>( m, "BpeTokenizer" )
        .def_static( "load_llama32",
            []( const std::string& path ) {
                return Tokenizer::loadLlama32( path );
            },
            py::arg( "path" ),
            "Load a Llama 3.2 tokenizer from a Mila binary vocabulary file." )
        .def_static( "load_gemma",
            []( const std::string& path ) {
                return Tokenizer::loadGemma( path );
            },
            py::arg( "path" ),
            "Load a Gemma 4 SentencePiece tokenizer from a Mila binary vocabulary file." )
        .def( "encode",
            []( Tokenizer& self, const std::string& text ) -> std::vector<int32_t> {
                py::gil_scoped_release _;
                return self.encode( text );
            },
            py::arg( "text" ),
            "Encode UTF-8 text to a list of token IDs." )
        .def( "decode",
            []( Tokenizer& self, const std::vector<int32_t>& ids ) -> std::string {
                py::gil_scoped_release _;
                return self.decode( ids );
            },
            py::arg( "ids" ),
            "Decode a list of token IDs to a UTF-8 string." )
        .def( "token_to_string",
            []( const Tokenizer& self, int32_t token_id ) {
                return self.tokenToString( token_id );
            },
            py::arg( "token_id" ) )
        .def( "is_valid_token",
            []( const Tokenizer& self, int32_t token_id ) {
                return self.isValidToken( token_id );
            },
            py::arg( "token_id" ) )
        .def_property_readonly( "vocab_size",
            []( const Tokenizer& self ) { return self.vocabSize(); } )
        .def_property_readonly( "bos_token_id",
            []( const Tokenizer& self ) -> py::object {
                auto id = self.bosTokenId();
                return id ? py::cast( *id ) : py::none();
            } )
        .def_property_readonly( "eos_token_id",
            []( const Tokenizer& self ) -> py::object {
                auto id = self.eosTokenId();
                return id ? py::cast( *id ) : py::none();
            } )
        .def_property_readonly( "pad_token_id",
            []( const Tokenizer& self ) -> py::object {
                auto id = self.padTokenId();
                return id ? py::cast( *id ) : py::none();
            } );
}

// ============================================================================
// Distribution bindings — the model store, and a hub driven from Python
// ============================================================================

/**
 * @brief The store's byte sink, presented to Python as a callable.
 *
 * A transport implemented in Python hands chunks here as they arrive. The bytes go straight
 * into the store's hash-and-write pass, so they are touched once: there is no staging file to
 * re-read and no second digest. The GIL is released around that pass, or a 6 GB transfer would
 * serialize against the interpreter.
 *
 * The buffer must not be mutated while the call is in flight. `bytes` -- what every Python HTTP
 * library yields -- is immutable, so this is only a caution for a caller passing a bytearray.
 */
class ChunkWriter
{
public:
    explicit ChunkWriter( const Mila::Bindings::ChunkSink& sink ) : sink_( sink ) {}

    bool write( py::buffer data ) const
    {
        const py::buffer_info info = data.request();

        const char* bytes = static_cast<const char*>( info.ptr );
        const std::size_t length =
            static_cast<std::size_t>( info.size ) * static_cast<std::size_t>( info.itemsize );

        py::gil_scoped_release release;

        return sink_( bytes, length );
    }

private:
    const Mila::Bindings::ChunkSink& sink_;
};

/**
 * @brief Wrap a Python callable as the library's HTTP fetch delegate.
 *
 * None means "use whatever transport this build has", which is an empty delegate rather than
 * one that fails -- the library decides, not this shim.
 */
static Mila::Bindings::HttpFetchDelegate makeHttpFetchDelegate( py::object transport )
{
    if ( transport.is_none() )
    {
        return {};
    }

    return [transport](
        const std::string& url,
        const std::vector<Mila::Bindings::HttpHeaderInfo>& headers,
        const Mila::Bindings::ChunkSink& sink ) -> Mila::Bindings::HttpResponseInfo
        {
            py::gil_scoped_acquire acquire;

            // Headers as a plain dict: order carries no meaning here and a mapping is what a
            // Python HTTP library wants.
            py::dict header_map;

            for ( const auto& header : headers )
            {
                header_map[ py::str( header.name ) ] = py::str( header.value );
            }

            // Scoped to this call: the writer borrows a sink valid only while the store is
            // inside ensureBlob. Reference policy so Python never takes ownership of
            // something with a shorter life than itself.
            ChunkWriter writer( sink );

            return transport(
                url, header_map,
                py::cast( &writer, py::return_value_policy::reference ) )
                .cast<Mila::Bindings::HttpResponseInfo>();
        };
}

static void bind_distribution( py::module_& m )
{
    py::class_<Mila::Bindings::HttpResponseInfo>( m, "HttpResponse" )
        .def( py::init<>() )
        .def( py::init( []( long http_code, std::string location, std::uint64_t content_length,
                bool transport_failed, std::string message ) {
                return Mila::Bindings::HttpResponseInfo{
                    http_code, std::move( location ), content_length,
                    transport_failed, std::move( message ) };
            } ),
            py::arg( "http_code" ) = 0,
            py::arg( "location" ) = std::string{},
            py::arg( "content_length" ) = 0,
            py::arg( "transport_failed" ) = false,
            py::arg( "message" ) = std::string{} )
        .def_readwrite( "http_code", &Mila::Bindings::HttpResponseInfo::http_code )
        .def_readwrite( "location", &Mila::Bindings::HttpResponseInfo::location,
            "The Location header verbatim. Do not follow it -- report it and stop." )
        .def_readwrite( "content_length", &Mila::Bindings::HttpResponseInfo::content_length )
        .def_readwrite( "transport_failed", &Mila::Bindings::HttpResponseInfo::transport_failed,
            "A connection, TLS or I/O failure, as distinct from any HTTP status." )
        .def_readwrite( "message", &Mila::Bindings::HttpResponseInfo::message );

    py::class_<ChunkWriter>( m, "ChunkWriter" )
        .def( "__call__", &ChunkWriter::write, py::arg( "data" ),
            "Hand one chunk to the store. Returns False to abort the transfer." );

    py::class_<Mila::Bindings::HubModelInfo>( m, "HubModel" )
        .def( py::init<>() )
        .def_readwrite( "owner", &Mila::Bindings::HubModelInfo::owner )
        .def_readwrite( "repository", &Mila::Bindings::HubModelInfo::repository )
        .def_readwrite( "gated", &Mila::Bindings::HubModelInfo::gated )
        .def_readwrite( "revision", &Mila::Bindings::HubModelInfo::revision )
        .def_readwrite( "last_modified", &Mila::Bindings::HubModelInfo::last_modified )
        .def_readwrite( "library", &Mila::Bindings::HubModelInfo::library )
        .def_readwrite( "tags", &Mila::Bindings::HubModelInfo::tags )
        .def_readwrite( "files", &Mila::Bindings::HubModelInfo::files );

    py::class_<Mila::Bindings::StoredModelInfo>( m, "StoredModel" )
        .def_readonly( "name", &Mila::Bindings::StoredModelInfo::name )
        .def_readonly( "architecture", &Mila::Bindings::StoredModelInfo::architecture )
        .def_readonly( "variant", &Mila::Bindings::StoredModelInfo::variant )
        .def_readonly( "weight_quantization",
            &Mila::Bindings::StoredModelInfo::weight_quantization )
        .def_readonly( "instruct", &Mila::Bindings::StoredModelInfo::instruct,
            "Instruction-tuned. Decides the prompt template a consumer applies." )
        .def_readonly( "base_model", &Mila::Bindings::StoredModelInfo::base_model )
        .def_readonly( "license", &Mila::Bindings::StoredModelInfo::license )
        .def_readonly( "hub", &Mila::Bindings::StoredModelInfo::hub,
            "Empty when the model was published from this machine." )
        .def_readonly( "owner", &Mila::Bindings::StoredModelInfo::owner )
        .def_readonly( "repository", &Mila::Bindings::StoredModelInfo::repository )
        .def_readonly( "revision", &Mila::Bindings::StoredModelInfo::revision )
        .def_readonly( "installed_at", &Mila::Bindings::StoredModelInfo::installed_at )
        .def_readonly( "weights_path", &Mila::Bindings::StoredModelInfo::weights_path )
        .def_readonly( "tokenizer_path", &Mila::Bindings::StoredModelInfo::tokenizer_path )
        .def_readonly( "complete", &Mila::Bindings::StoredModelInfo::complete )
        .def_readonly( "bytes_on_disk", &Mila::Bindings::StoredModelInfo::bytes_on_disk )
        .def( "__repr__",
            []( const Mila::Bindings::StoredModelInfo& self ) {
                return "<StoredModel " + self.name + " (" + self.architecture + ")>";
            } );

    py::class_<Mila::Bindings::StoreUsageInfo>( m, "StoreUsage" )
        .def_readonly( "blob_bytes", &Mila::Bindings::StoreUsageInfo::blob_bytes )
        .def_readonly( "reclaimable_bytes", &Mila::Bindings::StoreUsageInfo::reclaimable_bytes )
        .def_readonly( "partial_bytes", &Mila::Bindings::StoreUsageInfo::partial_bytes )
        .def_readonly( "model_count", &Mila::Bindings::StoreUsageInfo::model_count )
        .def_readonly( "blob_count", &Mila::Bindings::StoreUsageInfo::blob_count );

    py::class_<Mila::Bindings::RemovalReportInfo>( m, "RemovalReport" )
        .def_readonly( "records_removed", &Mila::Bindings::RemovalReportInfo::records_removed )
        .def_readonly( "blobs_removed", &Mila::Bindings::RemovalReportInfo::blobs_removed )
        .def_readonly( "files_removed", &Mila::Bindings::RemovalReportInfo::files_removed )
        .def_readonly( "bytes_reclaimed", &Mila::Bindings::RemovalReportInfo::bytes_reclaimed )
        .def_readonly( "retained", &Mila::Bindings::RemovalReportInfo::retained,
            "Paths the platform refused to delete, most often a blob a live process maps." );

    py::class_<Mila::Bindings::ModelStoreHandle>( m, "ModelStore" )
        .def( py::init<const std::string&>(), py::arg( "root" ) = std::string{},
            "Open the local store. An empty root takes MILA_CACHE_DIR, then the platform "
            "user cache -- the same store Chat and the Mila Inference Server use." )
        .def_property_readonly( "root",
            []( const Mila::Bindings::ModelStoreHandle& self ) { return self.root(); } )
        .def( "list", &Mila::Bindings::ModelStoreHandle::list,
            "Every installed model, complete or not." )
        .def( "locate", &Mila::Bindings::ModelStoreHandle::locate, py::arg( "name" ),
            "The model under this name, or None when it is absent or a blob is missing." )
        .def( "remove", &Mila::Bindings::ModelStoreHandle::remove, py::arg( "name" ),
            "Remove a model, reclaiming only blobs no surviving record references." )
        .def( "usage", &Mila::Bindings::ModelStoreHandle::usage,
            "What the store holds and what could be reclaimed." )
        .def( "install",
            []( Mila::Bindings::ModelStoreHandle& self, const std::string& package_directory,
                const std::string& name, bool replace, bool move_files ) {
                py::gil_scoped_release _;

                return self.install( package_directory, name, replace, move_files );
            },
            py::arg( "package_directory" ),
            py::arg( "name" ) = std::string{},
            py::arg( "replace" ) = false,
            py::arg( "move_files" ) = true,
            "Install a package directory into the store. Every file is hashed as it is\n"
            "adopted, so a blob is trusted only after its digest matches the manifest." )
        .def( "pull",
            []( Mila::Bindings::ModelStoreHandle& self, const std::string& name,
                const std::string& owner, py::object transport ) {
                auto delegate = makeHttpFetchDelegate( transport );

                // Released for the whole pull; the delegate reacquires per request, and
                // ChunkWriter drops it again for each hash-and-write.
                py::gil_scoped_release release;

                return self.pull( name, owner, delegate );
            },
            py::arg( "name" ),
            py::arg( "owner" ),
            py::arg( "transport" ) = py::none(),
            "Pull a published model, fetching every file its manifest declares and writing\n"
            "the record last, so a failed pull leaves nothing that looks installed.\n\n"
            "The library builds the URLs, finds the token, sets any Range, parses the\n"
            "manifest and verifies every digest. transport only moves bytes:\n\n"
            "    transport(url, headers, writer) -> HttpResponse\n\n"
            "headers is a dict to send verbatim -- the token and any Range are already in\n"
            "it. Call writer(chunk) for each chunk of a 2xx body; those bytes are hashed and\n"
            "written in one pass, so nothing is staged and re-read, and a False return means\n"
            "stop. Pass None for mila.http_transport, the standard-library implementation\n"
            "the package ships.\n\n"
            "Two obligations. Do NOT follow redirects: report location verbatim and stop, or\n"
            "the token travels to whatever host the 302 names. And report the status without\n"
            "interpreting it -- a Range answered 200 rather than 206 is the library's call." )
        .def( "list_hub_models",
            []( const Mila::Bindings::ModelStoreHandle& self, const std::string& owner,
                py::object transport ) {
                auto delegate = makeHttpFetchDelegate( transport );

                py::gil_scoped_release release;

                return self.listHubModels( owner, delegate );
            },
            py::arg( "owner" ),
            py::arg( "transport" ) = py::none(),
            "Every repository the owner publishes that Mila can load. Same transport\n"
            "contract as pull(); the listing is parsed by the library." );

    m.def( "default_hub_owner", &Mila::Bindings::defaultHubOwner,
        "The hub owner Mila publishes under." );
}

// ============================================================================
// LlamaModel bindings — Llama 3.2 3B Instruct, CUDA BF16
// ============================================================================

static void bind_llama_model( py::module_& m )
{
    py::class_<LlamaSession>( m, "LlamaModel" )
        .def_static( "from_pretrained",
            []( const std::string& path,
                int64_t context_length,
                int device_index,
                bool quantize_fp8 ) -> std::unique_ptr<LlamaSession>
            {
                py::gil_scoped_release _;

                return LlamaSession::fromPretrained(
                    path, context_length, device_index, quantize_fp8 );
            },
            py::arg( "path" ),
            py::arg( "context_length" ),
            py::arg( "device_index" ) = 0,
            py::arg( "quantize_fp8" ) = false,
            "Load Llama 3.x Instruct pretrained weights from a Mila artifact.\n\n"
            "Args:\n"
            "    path:           Path to the Mila pretrained artifact.\n"
            "    context_length: Maximum sequence length to build for.\n"
            "    device_index:   CUDA device index (default: 0).\n"
            "    quantize_fp8:   Quantize Linear weights to FP8_E4M3 at load time\n"
            "                    (default: False). The KV cache stays uncompressed.\n"
            "                    Requires SM >= 8.9 (RTX 40xx / Ada Lovelace)." )
        .def( "generate",
            []( LlamaSession& self,
                const std::vector<int32_t>& prompt_tokens,
                std::size_t max_new_tokens,
                float temperature,
                int top_k,
                float top_p ) -> std::vector<int32_t>
            {
                py::gil_scoped_release _;
                return self.generate( prompt_tokens, max_new_tokens, temperature, top_k, top_p );
            },
            py::arg( "prompt_tokens" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            py::arg( "top_p" ) = 1.0f,
            "Blocking generation. Returns prompt tokens followed by all generated tokens." )
        .def( "generate_streaming",
            []( LlamaSession& self,
                const std::vector<int32_t>& prompt_tokens,
                py::function on_token,
                std::size_t max_new_tokens,
                float temperature,
                int top_k,
                float top_p,
                StopController* stop_ctrl )
            {
                std::stop_token stop = stop_ctrl
                    ? stop_ctrl->get_token()
                    : std::stop_token{};

                py::gil_scoped_release release;

                self.generateStreaming(
                    prompt_tokens,
                    [&on_token]( int32_t tok ) {
                        py::gil_scoped_acquire acquire;
                        on_token( tok );
                    },
                    max_new_tokens, temperature, top_k, top_p,
                    std::move( stop ) );
            },
            py::arg( "prompt_tokens" ),
            py::arg( "on_token" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            py::arg( "top_p" ) = 1.0f,
            py::arg( "stop_controller" ) = py::none(),
            "Stream generation token by token. on_token(id: int) is called for each "
            "generated token (EOS excluded). Blocks until generation completes or "
            "stop_controller.request_stop() is called." )
        .def( "get_config",
            []( const LlamaSession& self ) {
                const LlamaConfigInfo cfg = self.getConfig();
                py::dict d;
                d["vocab_size"] = cfg.vocab_size;
                d["max_sequence_length"] = cfg.max_sequence_length;
                d["model_dim"] = cfg.model_dim;
                d["num_layers"] = cfg.num_layers;
                d["num_heads"] = cfg.num_heads;
                d["num_kv_heads"] = cfg.num_kv_heads;
                d["hidden_dim"] = cfg.hidden_dim;
                d["rope_theta"] = cfg.rope_theta;
                return d;
            } )
        .def( "__repr__",
            []( const LlamaSession& self ) { return self.repr(); } );
}

// ============================================================================
// GemmaModel bindings — Gemma 4 12B, CUDA BF16
// ============================================================================

static void bind_gemma_model( py::module_& m )
{
    py::class_<GemmaSession>( m, "GemmaModel" )
        .def_static( "from_pretrained",
            []( const std::string& path,
                int64_t context_length,
                int device_index ) -> std::unique_ptr<GemmaSession>
            {
                py::gil_scoped_release _;

                return GemmaSession::fromPretrained( path, context_length, device_index );
            },
            py::arg( "path" ),
            py::arg( "context_length" ),
            py::arg( "device_index" ) = 0,
            "Load Gemma 4 pretrained weights from a Mila artifact.\n\n"
            "Args:\n"
            "    path:           Path to the Mila pretrained artifact.\n"
            "    context_length: Maximum sequence length to build for.\n"
            "    device_index:   CUDA device index (default: 0)." )
        .def( "generate",
            []( GemmaSession& self,
                const std::vector<int32_t>& prompt_tokens,
                std::size_t max_new_tokens,
                float temperature,
                int top_k,
                float top_p ) -> std::vector<int32_t>
            {
                py::gil_scoped_release _;
                return self.generate( prompt_tokens, max_new_tokens, temperature, top_k, top_p );
            },
            py::arg( "prompt_tokens" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            py::arg( "top_p" ) = 1.0f,
            "Blocking generation. Returns prompt tokens followed by all generated tokens.\n"
            "For HF token-for-token parity use temperature=0.0 (greedy argmax)." )
        .def( "generate_streaming",
            []( GemmaSession& self,
                const std::vector<int32_t>& prompt_tokens,
                py::function on_token,
                std::size_t max_new_tokens,
                float temperature,
                int top_k,
                float top_p,
                StopController* stop_ctrl )
            {
                std::stop_token stop = stop_ctrl
                    ? stop_ctrl->get_token()
                    : std::stop_token{};

                py::gil_scoped_release release;

                self.generateStreaming(
                    prompt_tokens,
                    [&on_token]( int32_t tok ) {
                        py::gil_scoped_acquire acquire;
                        on_token( tok );
                    },
                    max_new_tokens, temperature, top_k, top_p,
                    std::move( stop ) );
            },
            py::arg( "prompt_tokens" ),
            py::arg( "on_token" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            py::arg( "top_p" ) = 1.0f,
            py::arg( "stop_controller" ) = py::none(),
            "Stream generation token by token. on_token(id: int) is called for each "
            "generated token (EOS excluded)." )
        .def( "get_config",
            []( const GemmaSession& self ) {
                const GemmaConfigInfo cfg = self.getConfig();
                py::dict d;
                d["vocab_size"] = cfg.vocab_size;
                d["max_sequence_length"] = cfg.max_sequence_length;
                d["model_dim"] = cfg.model_dim;
                d["num_layers"] = cfg.num_layers;
                d["num_heads"] = cfg.num_heads;
                d["num_kv_heads"] = cfg.num_kv_heads;
                d["head_dim"] = cfg.head_dim;
                d["global_head_dim"] = cfg.global_head_dim;
                d["hidden_dim"] = cfg.hidden_dim;
                d["window"] = cfg.window;
                d["rope_theta_local"] = cfg.rope_theta_local;
                d["rope_theta_global"] = cfg.rope_theta_global;
                d["final_logit_softcapping"] = cfg.final_logit_softcapping;
                return d;
            } )
        .def( "__repr__",
            []( const GemmaSession& self ) { return self.repr(); } );
}

// ============================================================================
// StopController binding
// ============================================================================

static void bind_stop_controller( py::module_& m )
{
    py::class_<StopController>( m, "StopController" )
        .def( py::init<>() )
        .def( "request_stop", &StopController::request_stop,
            "Signal the running generate_streaming() call to halt." )
        .def_property_readonly( "stop_requested", &StopController::stop_requested );
}

// ============================================================================
// Module entry point
// ============================================================================

// The extension is the PRIVATE half of the `mila` package: users import `mila`,
// whose __init__ registers the CUDA DLL directories and then re-exports from here.
// A bare top-level `mila` extension cannot do that -- there is no __init__ to run
// before the DLL load -- which is why the module is named _mila.
PYBIND11_MODULE( _mila, m )
{
    m.doc() =
        "Mila inference bindings for CUDA.\n\n"
        "Models:\n"
        "    GemmaModel  Gemma 4 Instruct, BF16 compute with FP4 weights (the flagship).\n"
        "    LlamaModel  Llama 3.x Instruct, BF16 compute, optional FP8 weights.\n\n"
        "Both are loaded from a Mila binary artifact with from_pretrained() and paired\n"
        "with a BpeTokenizer. The GIL is released around generation, so streaming\n"
        "callbacks, a StopController and Ctrl-C all work from Python.\n\n"
        "Distribution:\n"
        "    ModelStore  The local store Chat and the Mila Inference Server share:\n"
        "                list, locate, remove, usage, install, pull.\n\n"
        "Pull and load are separate verbs and only the store is loadable. pull() takes the\n"
        "transport as an argument, so this build needs no HTTP client of its own.";

    m.def( "initialize",
        []( const std::string& level ) {
            Mila::Bindings::initialize( level );
        },
        py::arg( "log_level" ) = "warning",
        "Initialize the Mila framework. log_level: trace | info | warning | error." );

    bind_stop_controller( m );
    bind_distribution( m );
    bind_tokenizer( m );
    bind_llama_model( m );
    bind_gemma_model( m );
}
