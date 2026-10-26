/**
 * @file AgentEvent.ixx
 * @brief One thing a response did as it ran: a token, text on a channel, a call opening or answered, a round ending.
 */

export module AI.AgentEvent;

import std;
import Dnn.GenerateStatus;
import Dnn.TensorTypes;
import Dnn.Models.Conversation.ToolCall;
import Dnn.Models.Conversation.Reply;
import AI.ToolDecision;

namespace Mila::AI
{
    /**
     * @brief What an agent reports while it responds, in the order it happened.
     *
     * The text of every Answer event, in order, is the response's text, and of every Reasoning event its reasoning.
     */
    export struct AgentEvent
    {
        enum class Kind
        {
            /// The model starts generating: once per round, and a turn has a round per set of calls it makes.
            RoundBegan,

            /// The model generated a token, reported before the text it settles. The text is the token decoded,
            /// the template's own markers included, so every Token event of a round is its output verbatim.
            Token,

            /// A reasoning span starts, or the round starts inside one.
            ReasoningOpened,
            Reasoning,
            ReasoningClosed,
            Answer,

            /// The model began writing a call.
            ToolCallOpened,

            /// A call was answered: run, declined, denied, refused for a tool not offered or arguments that do not
            /// match, or not run because the response reached its limit of rounds.
            ToolCalled,

            /// A call span the template's grammar could not read, as the model wrote it. Nothing ran.
            MalformedToolCall,

            /// The round's generation stopped, for the reason in status.
            RoundEnded,
        };

        Kind kind;

        /// Token: the decoded token. Reasoning and Answer: the text. ToolCalled: the result the model reads.
        /// MalformedToolCall: the span. Valid for the duration of the event only.
        std::string_view text;

        /// ToolCalled: the call answered. Null for every other kind.
        const Dnn::Conversation::ToolCall* call{ nullptr };

        /// ToolCalled: whether the result is an error -- the call did not run, the tool threw, or it ran out of time.
        bool is_error{ false };

        /// ToolCalled: how the call was answered, and who decided. The record of the call (ToolCalling.md 3.3)
        /// is this event; where it is kept is the application's.
        std::optional<ToolDecision> decision;

        /// ToolCalled: how long the tool ran; zero when it did not.
        std::chrono::nanoseconds duration{};

        /// ToolCalled: the size of what the tool returned, before anything cut it; empty when it did not return.
        std::optional<std::size_t> returned_bytes;

        /// RoundBegan: the most tokens the round may generate, the response's reply tokens and the context left
        /// both counted; empty when neither bounds it below the context.
        std::optional<Dnn::dim_t> budget;

        /// RoundEnded: why the round's generation stopped.
        Dnn::GenerateStatus status{ Dnn::GenerateStatus::Success };

        /// RoundEnded: the round's reply as the template read it. Null for every other kind.
        const Dnn::Conversation::Reply* reply{ nullptr };
    };
}
