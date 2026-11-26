/**
 * @file Assistant.KvCacheChoice.ixx
 * @brief The KV cache Mila runs a model with, chosen by the harness from the planner's prices (AssistantOnboarding.md
 *        4.4, round 12).
 */

export module Assistant.KvCacheChoice;

import std;
import Mila;

namespace Mila::Assistant
{
    /**
     * @brief The uncompressed cache, unless it leaves the model short of the context agentic work is measured at and
     *        the FP8 cache reaches it.
     *
     * The planner prices both and the caller chooses (Specifications/Deployment.md section 2). Choosing this way runs a
     * model measured agentic with the FP8 cache on the 16 GB card -- Gemma 4 26B-A4B, Qwen 3.8 27B fp4-cb3 -- as it was
     * measured (Todd, 2026-11-25), and leaves every model that reaches that context uncompressed as it was.
     *
     * @param uncompressed The best context the uncompressed cache plans; empty when nothing fits.
     * @param fp8 The same with the FP8 cache; empty when nothing fits or the model takes no FP8 cache.
     */
    export inline Mila::Dnn::KvCacheCompression chooseKvCache( std::optional<Mila::Dnn::dim_t> uncompressed,
        std::optional<Mila::Dnn::dim_t> fp8 ) noexcept
    {
        const bool short_of_agentic = !uncompressed || *uncompressed < Mila::Deployment::kAgenticContextLength;
        const bool fp8_reaches = fp8 && *fp8 >= Mila::Deployment::kAgenticContextLength;

        return short_of_agentic && fp8_reaches ? Mila::Dnn::KvCacheCompression::FP8 : Mila::Dnn::KvCacheCompression::None;
    }
}
