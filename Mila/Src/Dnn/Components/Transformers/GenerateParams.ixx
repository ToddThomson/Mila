/**
 * @file GenerateParams.ixx
 * @brief Per-call generation request: loop control plus the sampling shape.
 */

export module Dnn.GenerateParams;

import std;
import Data.Tokenizer;
import Dnn.SamplingParams;

namespace Mila::Dnn
{
    using TokenId = Data::TokenId;

    /**
     * @brief Per-call inputs to LanguageModel::generate.
     *
     * max_new_tokens is an optional ceiling -- nullopt runs to EOS or the deployment
     * context bound (no magic default, no silent truncation). sampling is the only
     * slice forwarded to the sampler; loop control never reaches it. stop_tokens
     * overrides the model's default stop set for this call (empty => model defaults),
     * kept for advanced structured generation -- EOS is otherwise a model/tokenizer
     * property established at construction, not a per-call parameter.
     *
     * reuse_kv_prefix lets the model skip the prompt prefix its KV cache already holds
     * (PromptCaching.md). Reuse is exact in tokens but not in floats: a prompt prefilled
     * from a reused prefix rounds differently from the same prompt prefilled whole, which
     * can change a sampled token. A reply that must not depend on what the model ran
     * before -- a seeded one -- turns it off and prefills the whole prompt.
     */
    export struct GenerateParams
    {
        std::optional<int> max_new_tokens;    // nullopt => run to EOS / context bound
        SamplingParams sampling{};
        std::vector<TokenId> stop_tokens{};   // empty => model's default stop set
        bool reuse_kv_prefix{ true };
    };
}
