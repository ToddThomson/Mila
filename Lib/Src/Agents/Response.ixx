/**
 * @file Response.ixx
 * @brief What an agent's response to one message came to.
 */

export module Agents.Response;

import std;
import Dnn.GenerateStatus;

namespace Mila::Agents
{
    /**
     * @brief One response, across every round its tool calls took.
     *
     * The text and reasoning are what streamed, every round's joined as it arrived.
     */
    export struct Response
    {
        std::string text;
        std::string reasoning;

        /// Why the last round stopped: Success when the model finished its answer, MaxNewTokensReached when the
        /// response used its reply tokens, ClientCancelled when it was stopped or its time ran out.
        Mila::Dnn::GenerateStatus status{ Mila::Dnn::GenerateStatus::Success };

        /// How many times the model generated: one, and one more after each round of calls.
        int rounds{ 0 };

        /// The response reached its limit of rounds with calls the model still wanted run.
        bool round_limit_reached{ false };

        /// The response's time ran out.
        bool time_limit_reached{ false };

        /// How many times the conversation was compacted while this response ran.
        int compactions{ 0 };
    };
}
