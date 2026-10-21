namespace Mila;

/// <summary>One turn of a conversation.</summary>
/// <param name="Role">"system", "user", "assistant" or "tool".</param>
/// <param name="Content">The turn's text.</param>
/// <param name="ToolCalls">The calls an assistant turn made.</param>
/// <param name="ToolName">The tool a tool turn answers for.</param>
/// <param name="Reasoning">An assistant turn's reasoning, for templates that render it.</param>
public sealed record ChatTurn(
    string Role,
    string Content,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    string? ToolName = null,
    string? Reasoning = null);
