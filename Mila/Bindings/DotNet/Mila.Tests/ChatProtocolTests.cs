namespace Mila.Tests;

/// <summary>Section 3.3: mila_chat_protocol_for_installed_model, _format_prompt, _parse_reply and _free, both JSON crossings.</summary>
public sealed class ChatProtocolTests
{
    [Fact]
    public void TurnsRenderAsRoleLinesAndHandOverToTheAssistant()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");
        ChatTurn[] history =
        [
            new ChatTurn("system", "Be brief."),
            new ChatTurn("user", "Say \"hi\"\n"),
        ];

        FormattedPrompt prompt = protocol.FormatPrompt(history);

        Assert.Equal("<system>Be brief.</system>\n<user>Say \"hi\"\n</user>\n<assistant>", prompt.Prompt);
        Assert.False(prompt.BeginsInReasoning);
    }

    [Fact]
    public void ToolCallsAndToolsCross()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");
        ToolCall call = new ToolCall("weather", "{\"city\":\"Z\u00fcrich\"}");
        ChatTurn[] history =
        [
            new ChatTurn("user", "Weather?"),
            new ChatTurn("assistant", string.Empty, [call]),
            new ChatTurn("tool", "sunny", ToolName: "weather"),
        ];
        string tools = "[{\"name\":\"weather\"}]";

        FormattedPrompt prompt = protocol.FormatPrompt(history, tools);

        string expected =
            "<tools>[{\"name\":\"weather\"}]</tools>\n" +
            "<user>Weather?</user>\n" +
            "<assistant><tool_call name=\"weather\">{\"city\":\"Z\u00fcrich\"}</tool_call></assistant>\n" +
            "<tool>sunny</tool>\n" +
            "<assistant>";
        Assert.Equal(expected, prompt.Prompt);
    }

    [Fact]
    public void AGemmaReplyBeginsInReasoning()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-gemma-it");
        ChatTurn[] history = [new ChatTurn("user", "Think.")];

        FormattedPrompt prompt = protocol.FormatPrompt(history);

        Assert.True(prompt.BeginsInReasoning);
        Assert.EndsWith("<assistant><think>", prompt.Prompt);
    }

    [Fact]
    public void AnEmptyHistoryIsAnArgumentException()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");

        ChatTurn[] history = [];

        Assert.Throws<ArgumentException>(() => protocol.FormatPrompt(history));
    }

    [Fact]
    public void AnUnknownRoleIsAnArgumentException()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");
        ChatTurn[] history = [new ChatTurn("narrator", "Once upon a time")];

        ArgumentException exception = Assert.Throws<ArgumentException>(() => protocol.FormatPrompt(history));

        Assert.Contains("'narrator'", exception.Message);
    }

    [Fact]
    public void AThinkPrefixParsesAsReasoning()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");

        ParsedReply reply = protocol.ParseReply("<think>plan \"it\"</think>Answer.", beginsInReasoning: false);

        Assert.Equal("plan \"it\"", reply.Reasoning);
        Assert.Equal("Answer.", reply.Content);
        Assert.Empty(reply.ToolCalls);
        Assert.Empty(reply.MalformedToolCalls);
    }

    [Fact]
    public void AReplyThatBeginsInReasoningNeedsNoOpeningMarker()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-gemma-it");

        ParsedReply reply = protocol.ParseReply("weighing it</think>Done.", beginsInReasoning: true);

        Assert.Equal("weighing it", reply.Reasoning);
        Assert.Equal("Done.", reply.Content);
    }

    [Fact]
    public void AReplyWithoutReasoningIsAllContent()
    {
        using ChatProtocol protocol = ChatProtocol.ForInstalledModel("mock-llama-it");

        ParsedReply reply = protocol.ParseReply("Just this.", beginsInReasoning: false);

        Assert.Equal(string.Empty, reply.Reasoning);
        Assert.Equal("Just this.", reply.Content);
    }

    [Fact]
    public void AnUnknownNameIsNotFound()
    {
        MilaException exception = Assert.Throws<MilaException>(() => ChatProtocol.ForInstalledModel("no-such-model"));

        Assert.Equal(MilaStatus.NotFound, exception.Status);
    }
}
