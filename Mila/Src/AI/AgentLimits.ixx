/**
 * @file AgentLimits.ixx
 * @brief How far one response may run: its rounds, its time and the tokens the model may generate in it.
 */

export module AI.AgentLimits;

import std;
import Dnn.TensorTypes;

namespace Mila::AI
{
    /**
     * @brief The limits on one response (ToolCalling.md section 4). A limit not set does not apply.
     *
     * With none set a response still ends: the conversation grows each round, and the context bounds it.
     */
    export struct AgentLimits
    {
        /// The most times the model may generate in one response. The calls of the last round allowed are not run.
        std::optional<int> rounds;

        /// The longest one response may run, its calls included; the time the person takes to answer is not counted.
        std::optional<std::chrono::milliseconds> time;

        /// The most tokens the model may generate in one response, across all its rounds.
        std::optional<Dnn::dim_t> reply_tokens;
    };
}
