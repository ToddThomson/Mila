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
            const std::string& path, int64_t context_length, int device_index );

        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            std::size_t max_new_tokens, float temperature, int top_k );

        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            std::size_t max_new_tokens, float temperature, int top_k,
            std::stop_token stop );

        LlamaConfigInfo getConfig() const;
        std::string repr() const;

        ~LlamaSession();

    private:
        struct Impl;
        explicit LlamaSession( std::unique_ptr<Impl> impl );

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
            std::size_t max_new_tokens, float temperature, int top_k );

        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            std::size_t max_new_tokens, float temperature, int top_k,
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
