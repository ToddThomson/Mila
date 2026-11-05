/**
 * @file AgentLimits.ixx
 * @brief How far one response may run: its rounds, its time, the tokens the model may generate in it, and how long
 *        and how large each of its tool calls may be.
 */

export module Agents.AgentLimits;

import std;
import Dnn.TensorTypes;

namespace Mila::Agents
{
    /**
     * @brief The limits on one response (ToolCalling.md sections 3.2 and 4). A limit not set does not apply.
     *
     * With none set a response ends when the model stops calling tools, or when the conversation has grown to its
     * context. A session that compacts removes the second: only a limit here, or stop, ends a model that never stops.
     */
    export struct AgentLimits
    {
        /// The most times the model may generate in one response. The calls of the last round allowed are not run.
        std::optional<int> rounds;

        /// The longest one response may run, its calls included; the time the person takes to answer is not counted.
        std::optional<std::chrono::milliseconds> time;

        /// The most tokens the model may generate in one response, across all its rounds.
        std::optional<Mila::Dnn::dim_t> reply_tokens;

        /// The longest one call of a tool that sets no time of its own may run. The response goes on after it.
        std::optional<std::chrono::milliseconds> tool_time;

        /// The most tokens of a result the model reads from a tool that sets no limit of its own, the rest cut.
        std::optional<Mila::Dnn::dim_t> result_tokens;
    };
}
