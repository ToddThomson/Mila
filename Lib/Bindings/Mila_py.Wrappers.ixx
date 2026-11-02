/**
 * @file Mila_py.Wrappers.ixx
 * @brief Opaque, std-only wrapper API over Mila for the MilaPy pybind11 extension.
 *
 * The pybind binding TU (Mila_py.cpp) must not `import Mila;`: the latest VS2026
 * MSVC raises C2079 (basic_istream::sentry undefined) whenever Mila is imported
 * into an ordinary .cpp that also includes std headers such as <string>.
 *
 * This INTERFACE unit exposes Tokenizer / LlamaSession handles with a std-only
 * surface and only FORWARD-DECLARES their PIMPL `struct Impl`. The Impl
 * definitions and all member bodies live in the implementation unit
 * (Mila_py.Wrappers.cpp), which is NOT reachable to importers. That keeps Impl
 * (and the Mila types it holds) incomplete in Mila_py.cpp, so binding the
 * handles with pybind11 never instantiates a Mila template. Defining Impl here
 * instead would make it reachable and force such instantiation. See
 * [[feedback-build-in-vs]].
 */

export module Mila.Bindings;

import std;

export import :LlamaConfigInfo;
export import :GemmaConfigInfo;
export import :QwenConfigInfo;
export import :ToolCallInfo;
export import :TurnInfo;
export import :ProtocolTokens;
export import :Tokenizer;
export import :LlamaSession;
export import :StoredModelInfo;
export import :StoreUsageInfo;
export import :RemovalReportInfo;
export import :HubModelInfo;
export import :HttpHeaderInfo;
export import :HttpResponseInfo;
export import :ModelStoreHandle;
export import :GemmaSession;
export import :QwenSession;
export import :SamplingSettingsInfo;

namespace Mila::Bindings
{
    export void initialize( const std::string& log_level );

    // ========================================================================
    // Chat protocol -- the model's own grammar, projected rather than reimplemented
    // ========================================================================

    /**
     * @brief Render a Qwen 3.8 conversation into the prompt its checkpoint was trained on.
     *
     * The whole template is here rather than in the caller: turn structure, the reasoning gate,
     * the ordering of the system turn's parts, and the tools section's exact wording. A host
     * supplies a conversation and gets a prompt, and reimplements nothing.
     *
     * @param reasoning_effort_scale 1..5. Mapped onto the three levels the checkpoint knows,
     *        whose middle level deliberately emits no instruction at all.
     * @param tools_json A JSON array of tool signature objects, or empty for none. Empty omits
     *        the tools section, and that absence is what tells the model there are none.
     *
     * @throws std::runtime_error if a role is not one of the four, or if the history is empty
     *         or ends on an assistant turn.
     */
    export std::string qwenFormatPrompt(
        const std::vector<TurnInfo>& history,
        bool enable_thinking,
        int reasoning_effort_scale,
        const std::string& tools_json );

    /**
     * @brief The first tool call in a Qwen response, or nothing when it holds none.
     *
     * @throws std::runtime_error if the span holds something that is not a call -- the model
     *         failing at its own protocol, which is worth surfacing rather than reading as prose.
     */
    export std::optional<ToolCallInfo> qwenParseToolCall( const std::string& response );

    /// Qwen's control tokens, as the checkpoint vocabulary registers them.
    export ProtocolTokens qwenProtocolTokens();

    /**
     * @brief Render a Gemma 4 conversation into the prompt its checkpoint was trained on.
     *
     * The whole template is here rather than in the caller: turn structure, the role spellings,
     * the empty-thought prime that opens a fresh model turn, and where declarations attach to
     * the system turn. A host supplies a conversation and gets a prompt.
     *
     * @param tool_declarations What gemmaToolDeclarations returns, or empty for none.
     * @param continue_open Emit the final turn OPEN so the next token continues it, which is
     *        the shape after a tool response. No thought prime is emitted in that case -- the
     *        turn already carries its channel, and a second one mid-turn is off-distribution.
     *
     * @throws std::runtime_error if a role is not one of the four, or if continue_open is set
     *         on an empty history.
     */
    export std::string gemmaFormatPrompt(
        const std::vector<TurnInfo>& history,
        const std::string& tool_declarations,
        bool continue_open );

    /**
     * @brief Tool schemas rendered in Gemma's trained <|tool>declaration:...<tool|> grammar.
     *
     * @param tools_json A JSON array of tool schemas, OpenAI function envelopes or bare
     *        declarations. Which tools to advertise is the host's choice: a harness with
     *        UI-only tools filters them out before calling.
     */
    export std::string gemmaToolDeclarations( const std::string& tools_json );

    /// The most recent tool call in a Gemma response, or nothing when it holds none.
    export std::optional<ToolCallInfo> gemmaParseToolCall( const std::string& response );

    /// One assistant tool call rendered back into the native call grammar, for replay.
    export std::string gemmaFormatToolCall( const std::string& name, const std::string& arguments );

    /**
     * @brief A client-executed tool result in Gemma's <|tool_response> grammar.
     *
     * A JSON envelope surfaces only its primary output field; metadata siblings are dropped, and
     * a failed tool's `error` is surfaced explicitly so the model does not blind-retry.
     */
    export std::string gemmaFormatToolResponse( const std::string& name, const std::string& result );

    /// A channel-structured response reduced to the user-facing answer.
    export std::string gemmaExtractAnswer( const std::string& text );

    /// Every registered control token removed from decoded text.
    export std::string gemmaStripControlTokens( const std::string& text );

    /// Gemma's control tokens, as the checkpoint vocabulary registers them.
    export ProtocolTokens gemmaProtocolTokens();

    // Quantization is named the way the store names it -- "bf16", "fp8", "fp4", and for Qwen
    // "cb2-3" and "fp4-cb3" -- by both session entry points, so a variant means the same thing
    // whether it came from a record or from a caller. "fp32" is rejected rather than ignored:
    // these sessions are BF16 instantiations, and loading FP32 weights at BF16 is a
    // different model than the one asked for.
    //
    // A context length of nullopt is automatic: the library's planner chooses the longest that
    // fits the device (Specifications/Deployment.md). A fixed one that does not fit is refused
    // before anything is allocated, never attempted. kv_cache is "bf16" or "fp8" on every load:
    // the cache is a deployment choice, not a property of the weights.
    //
    // A session's generate takes each sampling knob optionally: one not given is the model's
    // publisher's setting for the thinking the prompt was rendered with (ChatConfiguration.md
    // section 5), and thinking not given is the template's own default.

    /**
     * @brief The sampling an installed model's publisher recommends with thinking on or off.
     *
     * The manifest's settings, or its family's when it states none, with every knob the publisher leaves
     * unstated at the sampler's neutral value.
     *
     * @throws std::runtime_error if no such model is installed.
     */
    export SamplingSettingsInfo recommendedSampling( const std::string& name, bool thinking );
}
