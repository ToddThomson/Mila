/**
 * @file Chat.FamilyTraits.ixx
 * @brief What is true of every checkpoint of an architecture, in one row per family.
 *
 * Layer 1 of the resolution order in Specifications/ChatConfiguration.md section 3: the compiled
 * facts a publisher cannot change, and the context a run with no configuration file opens at.
 */

export module Chat.FamilyTraits;

import std;
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
        /// The largest context the architecture can ADDRESS, not the one it opens at. Every family
        /// Chat runs uses RoPE and extrapolates, so these ceilings are memory questions the
        /// footprint pre-flight answers rather than hard limits. A learned positional table would
        /// be a hard one -- indexing past it fails the load -- but no instruct family here has one.
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
            .max_context = 131072,
            .default_context = 4096 };

        constexpr FamilyTraits llama{
            .max_context = 131072,
            .default_context = 4096 };

        // 262144 is the checkpoint's own max_seq_length. It is far past what any card here
        // holds, which is the planner's problem and not this row's: the ceiling says what the
        // architecture can address.
        constexpr FamilyTraits qwen{
            .max_context = 262144,
            .default_context = 4096 };

        switch ( family )
        {
            case ModelType::Gemma: return gemma;
            case ModelType::Qwen:  return qwen;
            default:               return llama;
        }
    }
}
