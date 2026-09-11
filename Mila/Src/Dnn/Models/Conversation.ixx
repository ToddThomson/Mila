/**
 * @file Conversation.ixx
 * @brief The conversation a model's prompt template renders, independent of any family.
 *
 * A model's chat template is a property of the checkpoint, so it lives in the library -- and a
 * template needs a conversation to render. This is that conversation, in the one shape every
 * family's protocol module reads, so a host describes its exchange once rather than once per
 * family. The shape is not invented here: it is what the checkpoint's own chat_template.jinja
 * takes, which is why HuggingFace spells the same call apply_chat_template(messages).
 *
 * Deliberately smaller than what an application holds. There is no timestamp, no display state
 * and no correlation id: a template renders none of them, and a field a template cannot reach
 * would only be a second place to keep something correct.
 *
 * Sits beside the family protocol modules the way ITransformerBlock sits beside the family
 * blocks -- the one type they are all written against.
 */

module;
#include <string>
#include <vector>

export module Dnn.Models.Conversation;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief Who produced a turn.
     *
     * Named for the conversation rather than for any template's spelling of it. Llama writes
     * Tool as "ipython" and Qwen renders it as a user turn carrying a tool_response span -- both
     * are renderings of the same role, and each family's protocol module owns its own.
     */
    export enum class Role
    {
        System,
        User,
        Assistant,
        Tool
    };

    /**
     * @brief One call the model asked for, as a name and a JSON object of arguments.
     *
     * Arguments stay text rather than becoming a parsed object: every template renders them
     * back out, every host hands them on, and a round trip through a typed representation can
     * only lose something the model wrote.
     *
     * No id. No template in this library renders one -- Qwen pairs a call to its result
     * positionally and Gemma's grammar has no slot for it -- so correlating a call with its
     * result is the host's business, and a host that needs one mints it where it is needed.
     */
    export struct ToolCall
    {
        std::string name;
        std::string arguments;
    };

    /**
     * @brief A single turn in a conversation.
     *
     * tool_calls is non-empty only on an Assistant turn where the model issued one.
     */
    export struct Turn
    {
        Role role;
        std::string content;
        std::vector<ToolCall> tool_calls;
    };
}
