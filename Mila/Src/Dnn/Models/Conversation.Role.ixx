/**
 * @file Conversation.Role.ixx
 * @brief Who produced a turn of a conversation.
 */

export module Dnn.Models.Conversation.Role;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief Who produced a turn.
     *
     * Named for the conversation rather than for any template's spelling of it. Llama writes
     * Tool as "ipython" and Qwen renders it inside a user turn carrying a tool_response span --
     * both are renderings of the same role, and each template's protocol owns its own.
     */
    export enum class Role
    {
        System,
        User,
        Assistant,
        Tool
    };
}
