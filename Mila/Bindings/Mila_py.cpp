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
                (void)quantize_fp8;

                py::gil_scoped_release _;

                return LlamaSession::fromPretrained( path, context_length, device_index );
            },
            py::arg( "path" ),
            py::arg( "context_length" ),
            py::arg( "device_index" ) = 0,
            py::arg( "quantize_fp8" ) = false,
            "Load Llama 3.2 3B Instruct pretrained weights from a Mila artifact.\n\n"
            "Args:\n"
            "    path:          Path to the Mila pretrained artifact.\n"
            "    context_length: Maximum sequence length to build for.\n"
            "    device_index:  CUDA device index (default: 0).\n"
            "    quantize_fp8:  Quantize weights to FP8_E4M3 at load time (default: False).\n"
            "                   Requires SM >= 8.9 (RTX 40xx / Ada Lovelace)." )
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

PYBIND11_MODULE( mila, m )
{
    m.doc() = "Mila inference bindings — Llama 3.2 3B Instruct on CUDA BF16.";

    m.def( "initialize",
        []( const std::string& level ) {
            Mila::Bindings::initialize( level );
        },
        py::arg( "log_level" ) = "warning",
        "Initialize the Mila framework. log_level: trace | info | warning | error." );

    bind_stop_controller( m );
    bind_tokenizer( m );
    bind_llama_model( m );
    bind_gemma_model( m );
}
