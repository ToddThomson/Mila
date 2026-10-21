namespace Mila;

/// <summary>A reply split into its channels.</summary>
/// <param name="Reasoning">The reasoning channel's text.</param>
/// <param name="Content">The answer.</param>
/// <param name="ToolCalls">The calls the reply asked for.</param>
/// <param name="MalformedToolCalls">Text that opened a tool call but did not parse as one.</param>
public sealed record ParsedReply(
    string Reasoning,
    string Content,
    IReadOnlyList<ToolCall> ToolCalls,
    IReadOnlyList<string> MalformedToolCalls);
