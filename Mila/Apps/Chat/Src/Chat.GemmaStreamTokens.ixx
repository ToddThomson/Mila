/**
 * @file Chat.GemmaStreamTokens.ixx
 * @brief The Gemma 4 control-token ids that route a streamed reply to its channels.
 */

export module Chat.GemmaStreamTokens;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief Resolved Gemma control-token ids driving the channel router.
     *
     * All four routing ids must resolve (single-token vocab entries) for
     * streaming to be enabled; suppressed holds control tokens with no display
     * form (the live counterpart of the buffered path's special-token strip).
     */
    export struct GemmaStreamTokens
    {
        std::int32_t channel_open = -1;
        std::int32_t channel_close = -1;
        std::int32_t tool_call_open = -1;
        std::int32_t tool_call_close = -1;
        std::vector<std::int32_t> suppressed;
    };
}
