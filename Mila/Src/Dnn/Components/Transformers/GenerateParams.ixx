module;
#include <optional>

export module Dnn.GenerateParams;

import Data.Tokenizer;

namespace Mila::Dnn
{
    using TokenId = Data::TokenId;

    export struct GenerateParams {
        int max_new_tokens = 128;
        float temperature = 1.0f;
        std::optional<TokenId> eos_token_id;

        // Future expansion:
        // int top_k = 0;
        // float top_p = 1.0f;
        // float repetition_penalty = 1.0f;
    };
}