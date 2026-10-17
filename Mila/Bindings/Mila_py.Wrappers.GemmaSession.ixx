/**
 * @file Mila_py.Wrappers.GemmaSession.ixx
 * @brief Gemma 4 inference session (CUDA, BF16).
 */

export module Mila.Bindings:GemmaSession;

import std;
import :GemmaConfigInfo;

namespace Mila::Bindings
{
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
        static std::unique_ptr<GemmaSession> load(
            const std::string& path, std::optional<std::int64_t> context_length, int device_index,
            const std::string& quantization, const std::string& kv_cache );

        /// As LlamaSession::fromStore -- the record decides the quantization.
        static std::unique_ptr<GemmaSession> fromStore(
            const std::string& name, std::optional<std::int64_t> context_length, int device_index,
            const std::string& kv_cache );

        /// As LlamaSession::generate -- tokens through the callback, why it stopped returned.
        std::string generate(
            const std::vector<std::int32_t>& prompt_tokens,
            const std::function<void( std::int32_t )>& on_token,
            std::size_t max_new_tokens,
            std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
            std::optional<float> presence_penalty, std::optional<bool> thinking,
            std::stop_token stop );

        GemmaConfigInfo getConfig() const;

        /// As LlamaSession::contextLength.
        std::int64_t contextLength() const;

        std::string repr() const;

        ~GemmaSession();

    private:
        struct Impl;
        explicit GemmaSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
