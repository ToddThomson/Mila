// Streaming chat with an installed model, in C#: render the conversation in the model's own template,
// encode it, stream tokens back as text, and hand the turn to the model again.
//
//     dotnet run -- <installed model name>
//
// Ctrl-C stops a reply; an empty line quits.

using System.Runtime.InteropServices;
using System.Text;
using Mila;

MilaRuntime.Initialize(LogLevel.Warning);
IReadOnlyList<StoredModel> installed = ModelStore.List();

if (args.Length != 1 || !installed.Any(stored => stored.Name == args[0] && stored.Complete))
{
    Console.Error.WriteLine("usage: dotnet run -- <installed model name>, one of:");

    foreach (StoredModel stored in installed.Where(stored => stored.Complete))
    {
        Console.Error.WriteLine($"  {stored.Name}");
    }

    return 1;
}

string name = args[0];
using Tokenizer tokenizer = Tokenizer.FromStore(name);
using ChatProtocol protocol = ChatProtocol.ForInstalledModel(name);
using Model model = Model.FromStore(name);
Console.WriteLine($"{name}: {model.Family}, {model.ContextLength} tokens of context. Ctrl-C stops a reply; an empty line quits.");

List<ChatTurn> history = [];
List<int> pending = [];
StringBuilder reply = new StringBuilder();
CancellationTokenSource stop = new CancellationTokenSource();
Console.CancelKeyPress += StopReply;

while (true)
{
    Console.Write("\nyou> ");
    string? message = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(message))
    {
        break;
    }

    ChatTurn turn = new ChatTurn("user", message);
    history.Add(turn);
    FormattedPrompt prompt = protocol.FormatPrompt(history);
    int[] promptTokens = tokenizer.Encode(prompt.Prompt);
    pending.Clear();
    reply.Clear();
    stop = new CancellationTokenSource();
    Console.Write($"\n{name}> ");

    FinishReason reason = model.Generate(promptTokens, Show, null, stop.Token);

    // A reply that ended on its own needs no note; the other reasons all mean it was cut short.
    string note = reason == FinishReason.Stop ? string.Empty : $" [{reason}]";
    Console.WriteLine(note);
    string text = reply.ToString();
    ParsedReply parsed = protocol.ParseReply(text, prompt.BeginsInReasoning);
    ChatTurn answer = new ChatTurn("assistant", parsed.Content);
    history.Add(answer);
}

return 0;

// A token can carry only part of a character, so ids are held until they decode whole.
void Show(int token)
{
    pending.Add(token);
    ReadOnlySpan<int> held = CollectionsMarshal.AsSpan(pending);
    string piece;

    try
    {
        piece = tokenizer.Decode(held);
    }
    catch (DecoderFallbackException)
    {
        return;
    }

    pending.Clear();
    reply.Append(piece);
    Console.Write(piece);
}

void StopReply(object? sender, ConsoleCancelEventArgs press)
{
    press.Cancel = true;
    stop.Cancel();
}
