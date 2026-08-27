/**
 * @file Qwen.Protocol.ixx
 * @brief Canonical Qwen 3.8 chat protocol: the ChatML turn structure, the reasoning gate and
 *        the tool-call grammar.
 *
 * The grammar is a property of the model, not of any single adaptor, so this runtime module is
 * the one source of truth for it -- the same rule that brought Gemma's down (see
 * Gemma.Protocol.ixx). It was folded here from the Chat harness, where it was written first.
 *
 * Rendered from the checkpoint's own chat template rather than from the Qwen 3 conventions it
 * resembles, which differ in three places that all break silently: the call grammar is nested
 * tags and not a JSON object, reasoning effort is a trained parameter with its own instruction
 * text, and the generation primer leaves the reasoning span OPEN.
 *
 * One file for the whole protocol because the pieces agree by construction -- the markers here
 * are the ones BpeVocabulary::loadQwen registers from the checkpoint vocabulary, so a template
 * written apart from its parser could drift from it unnoticed.
 *
 * String-level parse/format only, matching Gemma's module.
 */

module;
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
// nlohmann::json::create instantiates here and compares a unique_ptr against nullptr, so this
// TU needs <memory>'s operators by ordinary lookup. Importing nlohmann.json does not supply
// them: they reach the BMI only as pruned global-module-fragment declarations.
#include <memory>

export module Dnn.Models.QwenProtocol;

export import Dnn.Models.Conversation;

import nlohmann.json;

namespace Mila::Dnn::Qwen
{
    /// The turn delimiters. Registered specials in the checkpoint vocabulary, so they are
    /// emitted as literal text here and encode atomically.
    export inline constexpr std::string_view kTurnOpen = "<|im_start|>";
    export inline constexpr std::string_view kTurnClose = "<|im_end|>";

    export inline constexpr std::string_view kThinkOpen = "<think>";
    export inline constexpr std::string_view kThinkClose = "</think>";

    export inline constexpr std::string_view kToolCallOpen = "<tool_call>";
    export inline constexpr std::string_view kToolCallClose = "</tool_call>";
    export inline constexpr std::string_view kToolResponseOpen = "<tool_response>";
    export inline constexpr std::string_view kToolResponseClose = "</tool_response>";

    /**
     * @brief How much reasoning to spend, as the checkpoint's template defines it.
     *
     * Three levels rather than a continuum, and Medium is deliberately silent: the template
     * emits no instruction for it, so the model runs at its trained default. Naming a fourth
     * value, or inventing wording for Medium, would put text in the system turn that this
     * checkpoint was not tuned against -- which is what steering Gemma through prose has to do
     * because Gemma's toggle carries no effort parameter at all.
     */
    export enum class ReasoningEffort
    {
        Low,
        Medium,
        Xhigh
    };

    /**
     * @brief Map a 1..5 effort scale onto the three levels the model knows.
     *
     * The scale is a host's, shared across families, so the ends are pinned to the ends and the
     * middle to the silent default. Here rather than in an adaptor so that two hosts asking for
     * the same effort get the same prompt.
     */
    export inline ReasoningEffort reasoningEffortFromScale( int level )
    {
        if ( level <= 2 )
        {
            return ReasoningEffort::Low;
        }

        return level == 3 ? ReasoningEffort::Medium : ReasoningEffort::Xhigh;
    }

    /// The instruction the template prepends to the system turn, verbatim. Empty for Medium.
    export inline std::string_view reasoningInstruction( ReasoningEffort effort )
    {
        switch ( effort )
        {
            case ReasoningEffort::Low:
                return "Reasoning effort is set to low. Keep your thinking brief and focused, "
                    "moving directly to the conclusion without unnecessary elaboration.";

            case ReasoningEffort::Xhigh:
                return "Reasoning effort is set to xhigh. Please think carefully through the "
                    "task, validate key assumptions, consider plausible alternatives, and "
                    "prioritize correctness, consistency, and clarity in the final answer.";

            case ReasoningEffort::Medium:
                return {};
        }

        return {};
    }

    /**
     * @brief The tools section of the system turn, verbatim from the template.
     *
     * The call format and the reminder block are protocol, not instructions we composed: the
     * model is tuned to emit exactly this shape, and a paraphrase is a different prompt.
     *
     * @param signatures One JSON object per tool, newline separated, as the template renders
     *        them inside <tools>.
     */
    export inline std::string toolsSection( const std::string& signatures )
    {
        return std::format(
            "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO "
            "suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
            "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> "
            "block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language "
            "BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your "
            "current knowledge and do not tell the user about function calls\n</IMPORTANT>",
            signatures );
    }

    /**
     * @brief The tool signatures as the template renders them: one JSON object per line.
     *
     * Not a JSON array -- the template writes the objects newline separated inside <tools> with
     * no array around them. Takes the array a host already holds, so a caller with tool schemas
     * in any form converts once, to JSON, rather than to a private wire shape.
     *
     * @param tools_json A JSON array of tool signature objects, as text.
     * @return Empty when the array is empty or does not parse, which omits the whole tools
     *         section -- and that absence is what tells the model there are none.
     */
    export inline std::string serializeToolSignatures( const std::string& tools_json )
    {
        nlohmann::json parsed;

        try
        {
            parsed = nlohmann::json::parse( tools_json.empty() ? "[]" : tools_json );
        }
        catch ( const nlohmann::json::exception& )
        {
            return {};
        }

        if ( !parsed.is_array() )
        {
            return {};
        }

        std::string lines;

        for ( const auto& tool : parsed )
        {
            lines += lines.empty() ? "" : "\n";
            lines += tool.dump();
        }

        return lines;
    }

    /**
     * @brief One tool call rendered in the grammar the model emits it in.
     *
     * @param arguments A JSON object as text. String values are spliced raw, everything else is
     *        re-encoded -- which is the template's own rule, and it is what lets a parameter
     *        hold prose containing quotes or newlines without escaping.
     */
    export inline std::string formatToolCall( const std::string& name, const std::string& arguments )
    {
        std::string rendered;

        rendered += kToolCallOpen;
        rendered += std::format( "\n<function={}>\n", name );

        nlohmann::json parsed;

        try
        {
            parsed = nlohmann::json::parse( arguments.empty() ? "{}" : arguments );
        }
        catch ( const nlohmann::json::exception& )
        {
            // The call came from this model's own output and round-tripped through a parse, so
            // this is unreachable in practice. A call that is somehow not an object renders with
            // no parameters rather than aborting the turn: the model can see it went wrong.
            parsed = nlohmann::json::object();
        }

        if ( parsed.is_object() )
        {
            // key()/value() rather than a structured binding over items(): the proxy's tuple
            // interface is not visible in this translation unit and the binding does not compile.
            for ( auto entry = parsed.begin(); entry != parsed.end(); ++entry )
            {
                rendered += std::format( "<parameter={}>\n", entry.key() );
                rendered += entry.value().is_string()
                    ? entry.value().get<std::string>() : entry.value().dump();
                rendered += "\n</parameter>\n";
            }
        }

        rendered += "</function>\n";
        rendered += kToolCallClose;

        return rendered;
    }

    /**
     * @brief Render one completed turn, delimiters included.
     *
     * A tool result is a USER turn carrying a <tool_response> span, not a role of its own --
     * rendering Conversation::Role::Tool by its own name would open a role Qwen was never trained to read.
     * Consecutive results are not merged into one user turn the way the template does; no host
     * dispatches parallel calls yet, so the case does not arise.
     */
    export inline std::string formatTurn( const Conversation::Turn& message )
    {
        std::string turn;
        turn.reserve( 128 + message.content.size() );

        if ( message.role == Conversation::Role::Tool )
        {
            turn += kTurnOpen;
            turn += "user\n";
            turn += kToolResponseOpen;
            turn += "\n";
            turn += message.content;
            turn += "\n";
            turn += kToolResponseClose;
            turn += kTurnClose;
            turn += "\n";

            return turn;
        }

        turn += kTurnOpen;

        if ( message.role == Conversation::Role::Assistant )
        {
            // An EMPTY reasoning span leads every assistant turn in history. The template writes
            // the turn's own reasoning here; a host does not keep it -- the channel parser splits
            // it off and only the answer is stored -- and the span still has to be present,
            // because its shape is what tells the model how that turn was generated.
            turn += std::format( "assistant\n{}\n\n{}\n\n", kThinkOpen, kThinkClose );
            turn += message.content;

            for ( const auto& call : message.tool_calls )
            {
                turn += message.content.empty() ? "" : "\n\n";
                turn += formatToolCall( call.name, call.arguments );
            }
        }
        else
        {
            turn += message.role == Conversation::Role::System ? "system\n" : "user\n";
            turn += message.content;
        }

        turn += kTurnClose;
        turn += "\n";

        return turn;
    }

    /**
     * @brief Render a history into Qwen's ChatML prompt, primed for generation.
     *
     * There is no BOS: the checkpoint sets `add_bos_token` false, and a leading marker the model
     * was not trained with costs a position and shifts every one after it.
     *
     * The system turn is assembled here rather than taken from history, because the template
     * orders it -- reasoning instruction, then tools, then whatever the caller configured -- and
     * a caller that had already concatenated them could not produce that order.
     *
     * @param enable_thinking False makes the primer carry a CLOSED, empty reasoning span, which
     *        is the model's own suppression mechanism: a finished channel is one it continues
     *        past. True leaves the span OPEN, so the response begins inside the reasoning and
     *        carries a closing marker with no opening one.
     *
     * @param tool_signatures One JSON object per tool, newline separated -- what
     *        serializeToolSignatures returns. Empty omits the whole tools section, which is what
     *        tells the model there are none.
     *
     * @throws std::invalid_argument if history is empty or ends on an Assistant turn, since the
     *         primer would then open a second consecutive assistant turn.
     */
    export inline std::string formatPrompt(
        std::span<const Conversation::Turn> history,
        bool enable_thinking,
        ReasoningEffort effort = ReasoningEffort::Medium,
        const std::string& tool_signatures = {} )
    {
        if ( history.empty() )
        {
            throw std::invalid_argument( "Qwen::formatPrompt: history must not be empty" );
        }

        if ( history.back().role == Conversation::Role::Assistant )
        {
            throw std::invalid_argument(
                "Qwen::formatPrompt: final message must be User or Tool, not Assistant" );
        }

        // Effort reaches the model only while it is allowed to think. With the span closed the
        // instruction would describe a channel that never opens.
        const std::string_view instruction =
            enable_thinking ? reasoningInstruction( effort ) : std::string_view{};

        std::string configured_system;

        for ( const auto& message : history )
        {
            if ( message.role == Conversation::Role::System )
            {
                configured_system = message.content;
            }
        }

        std::string system_content;

        if ( !instruction.empty() )
        {
            system_content += instruction;
        }

        if ( !tool_signatures.empty() )
        {
            system_content += system_content.empty() ? "" : "\n\n";
            system_content += toolsSection( tool_signatures );
        }

        if ( !configured_system.empty() )
        {
            system_content += system_content.empty() ? "" : "\n\n";
            system_content += configured_system;
        }

        std::string prompt;

        if ( !system_content.empty() )
        {
            prompt += formatTurn( { Conversation::Role::System, system_content } );
        }

        for ( const auto& message : history )
        {
            // Already assembled above, in the order the template puts its parts in.
            if ( message.role == Conversation::Role::System )
            {
                continue;
            }

            prompt += formatTurn( message );
        }

        prompt += kTurnOpen;
        prompt += "assistant\n";

        prompt += enable_thinking
            ? std::format( "{}\n", kThinkOpen )
            : std::format( "{}\n\n{}\n\n", kThinkOpen, kThinkClose );

        return prompt;
    }

    /**
     * @brief The value of one <parameter=...> block, typed the way the schema-free text allows.
     *
     * Parsed as JSON when it parses, and kept as a string when it does not. The grammar carries
     * no types -- a parameter holding `4096` and one holding the word `high` look identical --
     * so the choice is between guessing here and handing every handler a string it has to
     * re-interpret. JSON-when-valid recovers numbers, booleans and nested objects, and prose is
     * never valid JSON by accident.
     */
    inline nlohmann::json typedParameterValue( const std::string& text )
    {
        try
        {
            return nlohmann::json::parse( text );
        }
        catch ( const nlohmann::json::exception& )
        {
            return nlohmann::json( text );
        }
    }

    /**
     * @brief The first tool call in a response, or nothing when it holds none.
     *
     * Anchored on the <tool_call> span, so ordinary prose is never routed here -- and the
     * template itself permits natural-language reasoning BEFORE a call, which a parser keyed on
     * anything looser would swallow. An unterminated span means generation stopped inside the
     * call: nothing is returned rather than half a call being dispatched.
     *
     * @throws std::runtime_error if the span holds something that is not a call. The caller
     *         surfaces that -- a malformed call inside correct delimiters is the model failing
     *         at the protocol, which is worth seeing rather than silently treating as prose.
     */
    export inline std::optional<Conversation::ToolCall> parseToolCall( const std::string& response )
    {
        const auto open = response.find( kToolCallOpen );

        if ( open == std::string::npos )
        {
            return std::nullopt;
        }

        const auto body_start = open + kToolCallOpen.size();
        const auto close = response.find( kToolCallClose, body_start );

        if ( close == std::string::npos )
        {
            return std::nullopt;
        }

        const std::string body = response.substr( body_start, close - body_start );

        constexpr std::string_view kFunctionOpen = "<function=";
        constexpr std::string_view kFunctionClose = "</function>";
        constexpr std::string_view kParameterOpen = "<parameter=";
        constexpr std::string_view kParameterClose = "</parameter>";

        const auto function_open = body.find( kFunctionOpen );

        if ( function_open == std::string::npos )
        {
            throw std::runtime_error(
                "the <tool_call> span holds no <function=...> block" );
        }

        const auto name_start = function_open + kFunctionOpen.size();
        const auto name_end = body.find( '>', name_start );

        if ( name_end == std::string::npos )
        {
            throw std::runtime_error( "the <function=...> tag is unterminated" );
        }

        Conversation::ToolCall call;
        call.name = body.substr( name_start, name_end - name_start );

        if ( call.name.empty() )
        {
            throw std::runtime_error( "the <function=...> tag names no function" );
        }

        const auto function_close = body.find( kFunctionClose, name_end );
        const auto arguments_end = function_close == std::string::npos
            ? body.size() : function_close;

        nlohmann::json arguments = nlohmann::json::object();

        for ( auto cursor = body.find( kParameterOpen, name_end );
              cursor != std::string::npos && cursor < arguments_end;
              cursor = body.find( kParameterOpen, cursor ) )
        {
            const auto key_start = cursor + kParameterOpen.size();
            const auto key_end = body.find( '>', key_start );

            if ( key_end == std::string::npos || key_end >= arguments_end )
            {
                break;
            }

            const auto value_end = body.find( kParameterClose, key_end );

            if ( value_end == std::string::npos || value_end > arguments_end )
            {
                break;
            }

            // The template writes a newline after the tag and before the closing one, and they
            // are framing rather than content -- a value that kept them would differ from the
            // same value passed any other way.
            std::string_view value{ body };
            value = value.substr( key_end + 1, value_end - key_end - 1 );

            if ( value.starts_with( '\n' ) )
                value.remove_prefix( 1 );

            if ( value.ends_with( '\n' ) )
                value.remove_suffix( 1 );

            arguments[ body.substr( key_start, key_end - key_start ) ] =
                typedParameterValue( std::string( value ) );

            cursor = value_end + kParameterClose.size();
        }

        call.arguments = arguments.dump();

        return call;
    }
}
