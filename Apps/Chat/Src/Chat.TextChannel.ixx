/**
 * @file Chat.TextChannel.ixx
 * @brief Which part of a reply a run of text belongs to.
 */

export module Chat.TextChannel;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief The model's reasoning, or the answer the person reads.
     *
     * Tool calls are not a channel: the session reads them out of the reply and reports each one as
     * a call, so no front end sees their syntax.
     */
    export enum class TextChannel
    {
        Thought,
        Answer,
    };
}
