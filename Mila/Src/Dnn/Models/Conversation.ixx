/**
 * @file Conversation.ixx
 * @brief The conversation a model's chat template renders, and the protocol that renders it.
 *
 * A model's chat template is a property of the checkpoint, so it lives in the library -- and a
 * template needs a conversation to render. The shape is not invented here: it is what the
 * checkpoint's own chat_template.jinja takes, which is why HuggingFace spells the same call
 * apply_chat_template(messages). Re-exports one module per type, so a consumer that only builds a
 * history imports this and not a template's protocol.
 */

export module Dnn.Models.Conversation;

export import Dnn.Models.Conversation.Role;
export import Dnn.Models.Conversation.ToolCall;
export import Dnn.Models.Conversation.Turn;
export import Dnn.Models.Conversation.TemplateVariable;
export import Dnn.Models.Conversation.PromptOptions;
export import Dnn.Models.Conversation.Prompt;
export import Dnn.Models.Conversation.Reply;
export import Dnn.Models.Conversation.ProtocolMarkers;
export import Dnn.Models.Conversation.Protocol;
export import Dnn.Models.Conversation.TemplateJson;
