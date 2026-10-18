/**
 * @file Mila_py.Wrappers.QwenSession.ixx
 * @brief Qwen 3.8 inference session (CUDA, BF16).
 */

export module Mila.Bindings:QwenSession;

import std;
import :QwenConfigInfo;

namespace Mila::Bindings
{
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
        static std::unique_ptr<QwenSession> load(
            const std::string& path, std::optional<std::int64_t> context_length, int device_index,
            const std::string& quantization, const std::string& kv_cache );

        /// As GemmaSession::fromStore -- the record decides the quantization.
        static std::unique_ptr<QwenSession> fromStore(
            const std::string& name, std::optional<std::int64_t> context_length, int device_index,
            const std::string& kv_cache );

        /// As LlamaSession::generate -- tokens through the callback, why it stopped returned.
        std::string generate(
            const std::vector<std::int32_t>& prompt_tokens,
            const std::function<void( std::int32_t )>& on_token,
            std::size_t max_new_tokens,
            std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
            std::optional<float> presence_penalty, std::optional<bool> thinking,
            std::optional<std::uint64_t> seed,
            std::stop_token stop );

        QwenConfigInfo getConfig() const;

        /// As LlamaSession::contextLength.
        std::int64_t contextLength() const;

        std::string repr() const;

        ~QwenSession();

    private:
        struct Impl;
        explicit QwenSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
