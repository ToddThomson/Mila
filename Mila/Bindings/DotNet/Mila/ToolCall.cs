namespace Mila;

/// <summary>One call the model asked for.</summary>
/// <param name="Name">The tool's name.</param>
/// <param name="Arguments">A JSON object as text, as the model wrote it.</param>
public sealed record ToolCall(string Name, string Arguments);
