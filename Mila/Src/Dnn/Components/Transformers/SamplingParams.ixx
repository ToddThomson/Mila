/**
 * @file SamplingParams.ixx
 * @brief Per-call sampling parameters -- the "sampling shape".
 *
 * The knobs that turn a logits row into a token: temperature, top-k, top-p.
 * Carried on GenerateParams::sampling and forwarded to the TokenSampler; the
 * sampler never sees generation-loop control. Model-fixed sampler setup
 * (vocabulary size, final-logit softcap) lives in SamplingConfig, not here.
 */

export module Dnn.SamplingParams;

namespace Mila::Dnn
{
    /**
     * @brief Per-call sampling knobs consumed by the TokenSampler.
     *
     * Deliberately free of loop control (max_new_tokens) and lifetime state (seed):
     * loop bounds belong to GenerateParams, and reproducibility is a stream property
     * seeded once via LanguageModel::seedSampler, not a per-call value.
     */
    export struct SamplingParams
    {
        float temperature = 1.0f;
        int top_k = 0;      // 0 disables top-k truncation; 1 == greedy
        float top_p = 1.0f; // 1.0 disables nucleus (top-p) truncation
    };
}
