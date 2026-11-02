using System.Text.Json;

namespace Mila;

/// <summary>A model's chat template: render a conversation as a prompt, and split a reply into its channels.</summary>
public sealed class ChatProtocol : IDisposable
{
    private readonly ChatProtocolHandle handle;

    private ChatProtocol(ChatProtocolHandle handle)
    {
        this.handle = handle;
    }

    /// <summary>The template the installed model of that name runs with.</summary>
    /// <param name="name">The model's name in the store.</param>
    /// <exception cref="MilaException">No installed model has that name (<see cref="MilaStatus.NotFound"/>).</exception>
    public static ChatProtocol ForInstalledModel(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        AbiVersion.EnsureCompatible();
        int status = NativeMethods.mila_chat_protocol_for_installed_model(name, out ChatProtocolHandle protocol);
        NativeResult.ThrowIfFailed(status);

        return new ChatProtocol(protocol);
    }

    /// <summary>Renders the conversation with the template's default variables.</summary>
    /// <param name="history">The turns so far, ending with the one the model answers.</param>
    /// <param name="toolsJson">A JSON array of tool schemas, or null for none.</param>
    /// <exception cref="ArgumentException">The template does not accept a turn or the history's order.</exception>
    public FormattedPrompt FormatPrompt(IReadOnlyList<ChatTurn> history, string? toolsJson = null)
    {
        ArgumentNullException.ThrowIfNull(history);
        string historyJson = JsonSerializer.Serialize(history, MilaJsonContext.Default.IReadOnlyListChatTurn);
        int status = NativeMethods.mila_chat_protocol_format_prompt(handle, historyJson, toolsJson, out nint prompt, out int beginsInReasoning);
        NativeResult.ThrowIfFailed(status);
        string text = NativeResult.TakeString(prompt);
        bool begins = beginsInReasoning != 0;

        return new FormattedPrompt(text, begins);
    }

    /// <summary>Splits a reply into reasoning, answer and tool calls.</summary>
    /// <param name="text">The reply's decoded text.</param>
    /// <param name="beginsInReasoning">The prompt's <see cref="FormattedPrompt.BeginsInReasoning"/>.</param>
    public ParsedReply ParseReply(string text, bool beginsInReasoning)
    {
        ArgumentNullException.ThrowIfNull(text);
        int begins = beginsInReasoning ? 1 : 0;
        int status = NativeMethods.mila_chat_protocol_parse_reply(handle, text, begins, out nint replyJson);
        NativeResult.ThrowIfFailed(status);
        string json = NativeResult.TakeString(replyJson);
        ParsedReply? reply = JsonSerializer.Deserialize(json, MilaJsonContext.Default.ParsedReply);

        return reply ?? throw new MilaException(MilaStatus.Runtime, "The template parsed the reply as nothing.");
    }

    /// <summary>Releases the template.</summary>
    public void Dispose()
    {
        handle.Dispose();
    }
}
