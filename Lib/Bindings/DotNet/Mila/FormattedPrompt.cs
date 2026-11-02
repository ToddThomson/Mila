namespace Mila;

/// <summary>A conversation rendered in a model's chat template.</summary>
/// <param name="Prompt">The text to encode and generate from.</param>
/// <param name="BeginsInReasoning">The reply begins inside the reasoning channel; pass it to <see cref="ChatProtocol.ParseReply"/>.</param>
public sealed record FormattedPrompt(string Prompt, bool BeginsInReasoning);
