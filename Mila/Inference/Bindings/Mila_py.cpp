#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <filesystem>
#include <future>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <vector>

import Mila;

namespace py = pybind11;

using namespace Mila::Data;
using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;

using LlamaCudaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

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
    py::class_<BpeTokenizer, std::shared_ptr<BpeTokenizer>>( m, "BpeTokenizer" )
        .def_static( "load_llama32",
            []( const std::string& path ) {
                return BpeTokenizer::loadLlama32( std::filesystem::path( path ) );
            },
            py::arg( "path" ),
            "Load a Llama 3.2 tokenizer from a Mila binary vocabulary file." )
        .def( "encode",
            []( BpeTokenizer& self, const std::string& text ) -> std::vector<int32_t> {
                py::gil_scoped_release _;
                return self.encode( text );
            },
            py::arg( "text" ),
            "Encode UTF-8 text to a list of token IDs." )
        .def( "decode",
            []( BpeTokenizer& self, const std::vector<int32_t>& ids ) -> std::string {
                py::gil_scoped_release _;
                return self.decode( std::span<const int32_t>( ids ) );
            },
            py::arg( "ids" ),
            "Decode a list of token IDs to a UTF-8 string." )
        .def( "token_to_string", &BpeTokenizer::tokenToString, py::arg( "token_id" ) )
        .def( "is_valid_token", &BpeTokenizer::isValidToken, py::arg( "token_id" ) )
        .def_property_readonly( "vocab_size", &BpeTokenizer::getVocabSize )
        .def_property_readonly( "bos_token_id",
            []( const BpeTokenizer& self ) -> py::object {
                auto id = self.getBosTokenId();
                return id ? py::cast( *id ) : py::none();
            } )
        .def_property_readonly( "eos_token_id",
            []( const BpeTokenizer& self ) -> py::object {
                auto id = self.getEosTokenId();
                return id ? py::cast( *id ) : py::none();
            } )
        .def_property_readonly( "pad_token_id",
            []( const BpeTokenizer& self ) -> py::object {
                auto id = self.getPadTokenId();
                return id ? py::cast( *id ) : py::none();
            } );
}

// ============================================================================
// LlamaModel bindings — Llama 3.2 3B Instruct, CUDA BF16
// ============================================================================

static void bind_llama_model( py::module_& m )
{
    py::class_<LlamaCudaBf16>( m, "LlamaModel" )
        .def_static( "from_pretrained",
            []( const std::string& path,
                int64_t context_length,
                int device_index,
                bool quantize_fp8 ) -> std::unique_ptr<LlamaCudaBf16>
            {
                py::gil_scoped_release _;

                DeviceId device_id{ DeviceType::Cuda, device_index };

                LlamaModelConfig model_config = LlamaModelConfig( static_cast<dim_t>(context_length) );
                model_config.withQuantization( quantize_fp8 ? QuantizationConfig::fp8() : QuantizationConfig::none() );

                return LlamaCudaBf16::fromPretrained(
                    std::filesystem::path( path ),
                    model_config,
                    device_id );
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
            []( LlamaCudaBf16& self,
                const std::vector<int32_t>& prompt_tokens,
                size_t max_new_tokens,
                float temperature,
                int top_k ) -> std::vector<int32_t>
            {
                py::gil_scoped_release _;
                return self.generate( prompt_tokens, max_new_tokens, temperature, top_k );
            },
            py::arg( "prompt_tokens" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            "Blocking generation. Returns prompt tokens followed by all generated tokens." )
        .def( "generate_streaming",
            []( LlamaCudaBf16& self,
                const std::vector<int32_t>& prompt_tokens,
                py::function on_token,
                size_t max_new_tokens,
                float temperature,
                int top_k,
                StopController* stop_ctrl )
            {
                // Callback runs on the generateAsync worker thread; re-acquire the GIL
                // before invoking the Python callable.
                auto callback = [on_token]( int32_t tok ) {
                    py::gil_scoped_acquire _;
                    on_token( tok );
                };

                std::stop_token stop = stop_ctrl
                    ? stop_ctrl->get_token()
                    : std::stop_token{};

                std::future<void> fut;

                {
                    py::gil_scoped_release _;
                    fut = self.generateAsync(
                        prompt_tokens, std::move( callback ),
                        max_new_tokens, temperature, top_k,
                        std::move( stop ) );
                }

                fut.get();
            },
            py::arg( "prompt_tokens" ),
            py::arg( "on_token" ),
            py::arg( "max_new_tokens" ) = 64,
            py::arg( "temperature" ) = 1.0f,
            py::arg( "top_k" ) = 0,
            py::arg( "stop_controller" ) = py::none(),
            "Stream generation token by token. on_token(id: int) is called on a "
            "worker thread for each generated token (EOS excluded). Blocks until "
            "generation completes or stop_controller.request_stop() is called." )
        .def( "get_config",
            []( const LlamaCudaBf16& self ) {
                const auto& cfg = self.getConfig();
                py::dict d;
                d["vocab_size"] = cfg.getVocabSize();
                d["max_sequence_length"] = cfg.getMaxSequenceLength();
                d["model_dim"] = cfg.getModelDim();
                d["num_layers"] = cfg.getNumLayers();
                d["num_heads"] = cfg.getNumHeads();
                d["num_kv_heads"] = cfg.getNumKVHeads();
                d["hidden_dim"] = cfg.getHiddenDimension();
                d["rope_theta"] = cfg.getRoPETheta();
                return d;
            } )
        .def( "__repr__", &LlamaCudaBf16::toString );
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
    Mila::initialize();

    m.doc() = "Mila inference bindings — Llama 3.2 3B Instruct on CUDA BF16.";

    bind_stop_controller( m );
    bind_tokenizer( m );
    bind_llama_model( m );
}