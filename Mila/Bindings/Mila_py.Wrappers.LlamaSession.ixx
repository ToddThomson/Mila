/**
 * @file Mila_py.Wrappers.LlamaSession.ixx
 * @brief Llama 3.x inference session (CUDA, BF16).
 */

export module Mila.Bindings:LlamaSession;

import std;
import :LlamaConfigInfo;

namespace Mila::Bindings
{
    export class LlamaSession
    {
    public:
        /**
         * @param quantization Applied on the way in, against unquantized weights.
         *        Pre-quantized weights must be loaded through fromStore, which reads what
         *        their bytes already are.
         */
        static std::unique_ptr<LlamaSession> load(
            const std::string& path, std::optional<std::int64_t> context_length, int device_index,
            const std::string& quantization, const std::string& kv_cache );

        /**
         * @brief Load an installed model by store name, as its weights already are.
         *
         * The record decides the quantization, which is the whole point: a published
         * model is already FP4 or FP8 bytes, and a caller-supplied flag could only agree
         * with them by luck. Nothing here reaches a network -- an uninstalled name is an
         * error, never a download.
         *
         * @throws std::runtime_error if no such model is installed, its files are missing,
         *         its architecture or variant is not one this session can load, or it does
         *         not fit the device at the context length asked for.
         */
        static std::unique_ptr<LlamaSession> fromStore(
            const std::string& name, std::optional<std::int64_t> context_length, int device_index,
            const std::string& kv_cache );

        /**
         * @brief Generate from a prompt, streaming each token through on_token.
         *
         * The same shape as the library's own LanguageModel::generate: tokens leave
         * through the callback and the return value is why generation stopped, which
         * is the one outcome a caller cannot reconstruct from the token stream. The
         * status crosses as its wire spelling -- "stop", "length", "context_limit" or
         * "cancelled" -- so no enum has to cross with it.
         *
         * @param thinking How the prompt was rendered, which picks the model's recommended setting for every
         *        knob not given. Empty is the template's default.
         */
        std::string generate(
            const std::vector<std::int32_t>& prompt_tokens,
            const std::function<void( std::int32_t )>& on_token,
            std::size_t max_new_tokens,
            std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
            std::optional<float> presence_penalty, std::optional<bool> thinking,
            std::stop_token stop );

        LlamaConfigInfo getConfig() const;

        /// The context length this session was built for: the caller's, or the one the planner chose.
        std::int64_t contextLength() const;

        std::string repr() const;

        ~LlamaSession();

    private:
        struct Impl;
        explicit LlamaSession( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
