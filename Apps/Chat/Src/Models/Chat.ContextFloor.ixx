/**
 * @file Chat.ContextFloor.ixx
 * @brief The smallest context a session could hold a turn in, and what makes it up.
 */

export module Chat.ContextFloor;

import std;

namespace Mila::ChatApp
{
    /// Tokens an answer needs after the transcript and the reasoning budget. A policy number, and the only
    /// compiled part of the context floor -- the rest the session measures.
    export inline constexpr std::size_t kAnswerHeadroomTokens = 512;

    /**
     * @brief The floor and its parts.
     *
     * The parts are carried, not just the sum, because a refusal has to be arguable: "8192 is below the minimum"
     * invites the question this answers.
     */
    export struct ContextFloor
    {
        /// Zero when there was no tokenizer to measure with, which reads as "no floor".
        std::size_t minimum{ 0 };

        std::size_t transcript{ 0 };

        /// Zero unless thinking is both switched on and something this model can do.
        std::size_t reasoning{ 0 };
    };
}
