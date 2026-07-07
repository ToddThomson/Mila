/**
 * @file Chat.GemmaToolCallParser.ixx
 * @brief RETIRED 2026-07-07. Superseded by the canonical runtime grammar module
 *        Dnn.Components.GemmaProtocol (Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx).
 *
 * The Gemma native token grammar is a property of the model, not of the Chat
 * adaptor, so it was folded DOWN into the runtime where the inference server can
 * share it too. The runtime version closes the drift this file carried: it parses
 * and renders the trained <|"|> string delimiter (this parser handled plain quotes
 * only), keeps integer arguments as integers, and distills tool-response output
 * fields with failed-tool error surfacing.
 *
 * Chat now imports Dnn.Components.GemmaProtocol:
 *   GemmaToolCallParser::parse            -> Mila::Dnn::Gemma::parseToolCall
 *   GemmaToolCallParser::formatToolResponse -> Mila::Dnn::Gemma::formatToolResponse
 *
 * This file is out of the ChatApp module set (Mila/Adaptors/Chat/CMakeLists.txt)
 * and is retained only as a retirement marker; it is not compiled.
 */
