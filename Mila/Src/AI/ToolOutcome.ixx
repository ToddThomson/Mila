/**
 * @file ToolOutcome.ixx
 * @brief What became of one tool call at the gate: the decision, the result and how long it ran.
 */

export module AI.ToolOutcome;

import std;
import Dnn.TensorTypes;
import AI.ToolDecision;

namespace Mila::AI
{
    /**
     * @brief One call answered by a ToolGate.
     */
    export struct ToolOutcome
    {
        ToolDecision decision{ ToolDecision::NotOffered };

        /// What the model reads, unless the response was stopped.
        std::string result;

        /// Whether the result is an error: the call did not run, failed, or ran out of its time.
        bool is_error{ true };

        /// Whether the tool's function was called.
        bool ran{ false };

        /// The response was stopped before the call ran or while it ran, so result is empty: what a tool returns
        /// after a stop may be partial and is never given to the model.
        bool stopped{ false };

        /// The call ran out of its time and was asked to stop; result says so.
        bool timed_out{ false };

        /// How long the function ran; zero when it did not.
        std::chrono::nanoseconds duration{};

        /// The size of what the function returned, before anything cut it; empty when it did not return.
        std::optional<std::size_t> returned_bytes;

        /// The most tokens of the result the model may read: the tool's own, else the response's limit.
        std::optional<Dnn::dim_t> result_tokens;
    };
}
