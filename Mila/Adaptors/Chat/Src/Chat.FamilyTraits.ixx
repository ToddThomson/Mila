/**
 * @file Chat.FamilyTraits.ixx
 * @brief What is true of every checkpoint of an architecture, in one row per family.
 *
 * Layer 1 of the resolution order in Specifications/ChatConfiguration.md section 3: the compiled
 * facts a publisher cannot change, and the context a run with no configuration file opens at.
 */

module;
#include <cstddef>

export module Chat.FamilyTraits;

import Chat.Config;

namespace Mila::ChatApp
{
    /**
     * @brief The compiled facts about one architecture.
     *
     * These were four switch statements in the model catalog, two of them 245 lines apart, which
     * is how "Gemma has a reasoning channel" and "Gemma opens at 512 context" came to live in one
     * file with nothing reconciling them. One row per family puts a contradiction where it cannot
     * be missed.
     */
    export struct FamilyTraits
    {
        /// Whether the architecture has a reasoning channel at all. A property of the weights: no
        /// configuration key can give a family a channel it was not trained with.
        bool thinking_capable;

        /// Whether the display can route this family's output token by token. A harness
        /// capability, unlike the line above -- only Gemma's channels and tool calls are protocol
        /// tokens a per-token router can see -- so it moves when a family's streaming path is
        /// written, not when a checkpoint is published.
        bool streaming_capable;

        /// The largest context the architecture can ADDRESS, not the one it opens at. Only GPT-2's
        /// is a hard architectural limit: its positions are a 1024-row learned table, so a larger
        /// value indexes past it and the load fails. The RoPE families extrapolate, so their
        /// ceiling is a memory question that the footprint pre-flight answers.
        std::size_t max_context;

        /// What to open at when no layer above says otherwise, and the floor auto falls back to
        /// when no device can be measured (section 6 of the specification). It therefore has to be
        /// a context that WORKS, not merely one that loads: Gemma's was 512, which is the balanced
        /// reasoning budget exactly, so every round ended on `length` with no stop token for the
        /// users who have no config file. A fallback must clear the same floor /context enforces.
        std::size_t default_context;
    };

    /**
     * @brief The row for one family. The table itself, so the rows sit adjacent.
     */
    export constexpr FamilyTraits familyTraits( ModelType family )
    {
        constexpr FamilyTraits gemma{
            .thinking_capable = true,
            .streaming_capable = true,
            .max_context = 131072,
            .default_context = 4096 };

        constexpr FamilyTraits llama{
            .thinking_capable = false,
            .streaming_capable = false,
            .max_context = 131072,
            .default_context = 4096 };

        constexpr FamilyTraits gpt{
            .thinking_capable = false,
            .streaming_capable = false,
            .max_context = 1024,
            .default_context = 1024 };

        switch ( family )
        {
            case ModelType::Gemma: return gemma;
            case ModelType::Llama: return llama;
            case ModelType::Gpt:   return gpt;
        }

        return llama;
    }
}
