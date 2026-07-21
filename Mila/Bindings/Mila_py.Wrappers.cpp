/**
 * @file Mila_py.Wrappers.cpp
 * @brief Implementation unit for Mila.Bindings: defines the PIMPL Impl structs
 *        and all wrapper member bodies.
 *
 * This is a module IMPLEMENTATION unit (`module Mila.Bindings;`, no export), so
 * `import Mila;` is safe here (the C2079 regression only affects ordinary,
 * non-module .cpp TUs). Because implementation-unit definitions are not
 * reachable to importers, the Impl structs and the Mila types they hold stay
 * invisible to Mila_py.cpp. See [[feedback-build-in-vs]].
 */

module;

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

module Mila.Bindings;

import Mila;

namespace Mila::Bindings
{
    using namespace Mila::Data;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using LlamaCudaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

    void initialize( const std::string& log_level )
    {
        static const std::unordered_map<std::string, Mila::Logging::LogLevel> level_map = {
            { "trace",   Mila::Logging::LogLevel::Trace   },
            { "info",    Mila::Logging::LogLevel::Info    },
            { "warning", Mila::Logging::LogLevel::Warning },
            { "error",   Mila::Logging::LogLevel::Error   },
        };

        const auto it = level_map.find( log_level );
        const auto resolved = (it != level_map.end()) ? it->second : Mila::Logging::LogLevel::Warning;

        auto sink = std::make_shared<Mila::Logging::ConsoleSink>( resolved );
        Mila::initialize( 0, std::move( sink ) );
    }

    // ---- Tokenizer ----------------------------------------------------------

    struct Tokenizer::Impl
    {
        std::shared_ptr<BpeTokenizer> tokenizer;
    };

    Tokenizer::Tokenizer( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    Tokenizer::~Tokenizer() = default;

    std::shared_ptr<Tokenizer> Tokenizer::loadLlama32( const std::string& path )
    {
        auto impl = std::make_unique<Impl>();
        impl->tokenizer = BpeTokenizer::loadLlama32( std::filesystem::path( path ) );

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::shared_ptr<Tokenizer> Tokenizer::loadGemma( const std::string& path )
    {
        auto impl = std::make_unique<Impl>();
        impl->tokenizer = BpeTokenizer::loadGemma( std::filesystem::path( path ) );

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::vector<int32_t> Tokenizer::encode( const std::string& text )
    {
        return impl_->tokenizer->encode( text );
    }

    std::string Tokenizer::decode( const std::vector<int32_t>& ids )
    {
        return impl_->tokenizer->decode( std::span<const int32_t>( ids ) );
    }

    std::string Tokenizer::tokenToString( int32_t token_id ) const
    {
        return impl_->tokenizer->tokenToString( token_id );
    }

    bool Tokenizer::isValidToken( int32_t token_id ) const
    {
        return impl_->tokenizer->isValidToken( token_id );
    }

    int64_t Tokenizer::vocabSize() const
    {
        return static_cast<int64_t>( impl_->tokenizer->getVocabSize() );
    }

    std::optional<int32_t> Tokenizer::bosTokenId() const
    {
        auto id = impl_->tokenizer->getBosTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<int32_t> Tokenizer::eosTokenId() const
    {
        auto id = impl_->tokenizer->getEosTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<int32_t> Tokenizer::padTokenId() const
    {
        auto id = impl_->tokenizer->getPadTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    // ---- LlamaSession -------------------------------------------------------

    struct LlamaSession::Impl
    {
        std::unique_ptr<LlamaCudaBf16> model;
    };

    LlamaSession::LlamaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    LlamaSession::~LlamaSession() = default;

    std::unique_ptr<LlamaSession> LlamaSession::fromPretrained(
        const std::string& path, int64_t context_length, int device_index )
    {
        DeviceId device_id{ DeviceType::Cuda, device_index };
        LlamaModelConfig model_config( static_cast<dim_t>( context_length ) );

        auto impl = std::make_unique<Impl>();
        impl->model = LlamaCudaBf16::fromPretrained(
            std::filesystem::path( path ), model_config, device_id );

        return std::unique_ptr<LlamaSession>( new LlamaSession( std::move( impl ) ) );
    }

    std::vector<int32_t> LlamaSession::generate(
        const std::vector<int32_t>& prompt_tokens,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;

        // Blocking convenience over the streaming-only core primitive: collect the
        // generated tokens onto the prompt so the caller receives prompt + completion.
        std::vector<int32_t> output( prompt_tokens.begin(), prompt_tokens.end() );
        // Finish reason is not part of this blocking convenience shape; the caller
        // infers completion from the returned token list.
        (void)impl_->model->generate(
            prompt_tokens,
            [&output]( int32_t token ) { output.push_back( token ); },
            params );

        return output;
    }

    void LlamaSession::generateStreaming(
        const std::vector<int32_t>& prompt_tokens,
        const std::function<void( int32_t )>& on_token,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p,
        std::stop_token stop )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;
        (void)impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );
    }

    LlamaConfigInfo LlamaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getConfig();

        return LlamaConfigInfo{
            .vocab_size          = static_cast<int64_t>( cfg.getVocabSize() ),
            .max_sequence_length = static_cast<int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim           = static_cast<int64_t>( cfg.getModelDim() ),
            .num_layers          = static_cast<int64_t>( cfg.getNumLayers() ),
            .num_heads           = static_cast<int64_t>( cfg.getNumHeads() ),
            .num_kv_heads        = static_cast<int64_t>( cfg.getNumKVHeads() ),
            .hidden_dim          = static_cast<int64_t>( cfg.getHiddenDimension() ),
            .rope_theta          = static_cast<double>( cfg.getRoPETheta() ),
        };
    }

    std::string LlamaSession::repr() const
    {
        return impl_->model->toString();
    }

    // ---- GemmaSession -------------------------------------------------------

    struct GemmaSession::Impl
    {
        std::unique_ptr<GemmaCudaBf16> model;
    };

    GemmaSession::GemmaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    GemmaSession::~GemmaSession() = default;

    std::unique_ptr<GemmaSession> GemmaSession::fromPretrained(
        const std::string& path, int64_t context_length, int device_index )
    {
        DeviceId device_id{ DeviceType::Cuda, device_index };

        // Hardcode FP4 (PerGroupFp4<128> weights + sliding-window KV ring): Gemma 4 12B
        // only fits the 12 GB target under FP4 -- the default None (BF16) weights would
        // need ~24 GB and OOM at load. Mirrors the chat catalog's gemma-12b default.
        GemmaModelConfig model_config = GemmaModelConfig( static_cast<dim_t>( context_length ) )
            .withFP4Quantization();

        auto impl = std::make_unique<Impl>();
        impl->model = GemmaCudaBf16::fromPretrained(
            std::filesystem::path( path ), model_config, device_id );

        return std::unique_ptr<GemmaSession>( new GemmaSession( std::move( impl ) ) );
    }

    std::vector<int32_t> GemmaSession::generate(
        const std::vector<int32_t>& prompt_tokens,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;

        // Blocking convenience over the streaming-only core primitive: collect the
        // generated tokens onto the prompt so the caller receives prompt + completion.
        std::vector<int32_t> output( prompt_tokens.begin(), prompt_tokens.end() );
        // Finish reason is not part of this blocking convenience shape; the caller
        // infers completion from the returned token list.
        (void)impl_->model->generate(
            prompt_tokens,
            [&output]( int32_t token ) { output.push_back( token ); },
            params );

        return output;
    }

    void GemmaSession::generateStreaming(
        const std::vector<int32_t>& prompt_tokens,
        const std::function<void( int32_t )>& on_token,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p,
        std::stop_token stop )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;
        (void)impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );
    }

    GemmaConfigInfo GemmaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getNetworkConfig();

        return GemmaConfigInfo{
            .vocab_size              = static_cast<int64_t>( cfg.getVocabSize() ),
            .max_sequence_length     = static_cast<int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim               = static_cast<int64_t>( cfg.getModelDim() ),
            .num_layers              = static_cast<int64_t>( cfg.getNumLayers() ),
            .num_heads               = static_cast<int64_t>( cfg.getNumHeads() ),
            .num_kv_heads            = static_cast<int64_t>( cfg.getNumKVHeads() ),
            .head_dim                = static_cast<int64_t>( cfg.getHeadDim() ),
            .global_head_dim         = static_cast<int64_t>( cfg.getGlobalHeadDim() ),
            .hidden_dim              = static_cast<int64_t>( cfg.getHiddenDimension() ),
            .window                  = static_cast<int64_t>( cfg.getWindow() ),
            .rope_theta_local        = static_cast<double>( cfg.getRoPEThetaLocal() ),
            .rope_theta_global       = static_cast<double>( cfg.getRoPEThetaGlobal() ),
            .final_logit_softcapping = static_cast<double>( cfg.getFinalLogitSoftcapping() ),
        };
    }

    std::string GemmaSession::repr() const
    {
        return impl_->model->toString();
    }
}
