/**
 * @file Chat.QwenProtocol.ixx
 * @brief RETIRED 2026-08-27. Superseded by the canonical runtime grammar module
 *        Dnn.Models.QwenProtocol (Src/Dnn/Models/Qwen/Qwen.Protocol.ixx).
 *
 * The Qwen 3.8 chat protocol -- the ChatML turn structure, the reasoning gate and the tool-call
 * grammar -- is a property of the model, not of the Chat adaptor, so it was folded DOWN into the
 * runtime where the inference server can share it too. Same rule and same move as
 * Chat.GemmaToolCallParser.ixx records for Gemma.
 *
 * The runtime version is the same code retargeted onto Mila::Dnn::Conversation::Turn, the family-neutral
 * history type, so that a template does not need an adaptor's message class to render. It also
 * absorbed Chat's serializeQwenToolSignatures, which renders the <tools> section's
 * one-object-per-line form -- that shape is the template's rule, not the harness's.
 *
 * Chat now imports Dnn.Models.QwenProtocol:
 *   Mila::ChatApp::Qwen::formatPrompt              -> Mila::Dnn::Qwen::formatPrompt
 *   Mila::ChatApp::Qwen::parseToolCall             -> Mila::Dnn::Qwen::parseToolCall
 *   Mila::ChatApp::Qwen::reasoningEffortFromScale  -> Mila::Dnn::Qwen::reasoningEffortFromScale
 *   Chat::serializeQwenToolSignatures              -> Mila::Dnn::Qwen::serializeToolSignatures
 *
 * This file is out of the ChatApp module set (Mila/Adaptors/Chat/CMakeLists.txt) and is retained
 * only as a retirement marker; it is not compiled.
 */
