/**
 * @file Response.ixx
 * @brief What an agent's response to one message came to.
 */

export module AI.Response;

import std;
import Dnn.GenerateStatus;

namespace Mila::AI
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
        Dnn::GenerateStatus status{ Dnn::GenerateStatus::Success };

        /// How many times the model generated: one, and one more after each round of calls.
        int rounds{ 0 };

        /// The response reached its limit of rounds with calls the model still wanted run.
        bool round_limit_reached{ false };

        /// The response's time ran out.
        bool time_limit_reached{ false };
    };
}
