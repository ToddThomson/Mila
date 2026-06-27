module;
#include <optional>
#include <cstdint>

export module Dnn.GenerateParams;

import Data.Tokenizer;

namespace Mila::Dnn
{
    using TokenId = Data::TokenId;

    export struct GenerateParams {
        int max_new_tokens = 128;
        float temperature = 1.0f;
        int top_k = 0;                  // 0 disables top-k truncation; 1 == greedy
        float top_p = 1.0f;             // 1.0 disables nucleus (top-p) truncation
        std::optional<uint64_t> seed;   // nullopt => nondeterministic (time-seeded)
        std::optional<TokenId> eos_token_id;

        // Future expansion:
        // float repetition_penalty = 1.0f;
    };

    // The per-call sampling slice consumed by TokenSampler / SamplingOp. Aliased
    // (not duplicated) onto GenerateParams so temperature / top_k / top_p / seed have
    // a single source of truth (see TokenSampling.md section 4.1).
    export using SamplingParams = GenerateParams;
}
