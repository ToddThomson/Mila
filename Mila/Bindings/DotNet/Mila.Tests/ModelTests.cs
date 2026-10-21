namespace Mila.Tests;

/// <summary>Section 3.4: mila_model_* and mila_stop_source_*, and section 4.1's callback and cancellation rules.</summary>
public sealed class ModelTests
{
    private const string MockReply = "Hello from the Mila mock.";
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);
    private static readonly int[] HelloPrompt = ['H' + 3, 'i' + 3];

    [Fact]
    public void FromStoreReportsFamilyAndAutomaticContext()
    {
        using Model model = Model.FromStore("mock-gemma-it");

        Assert.Equal("gemma", model.Family);
        Assert.Equal(4096, model.ContextLength);
    }

    [Fact]
    public void LoadOptionsCross()
    {
        LoadOptions options = new LoadOptions { ContextLength = 1024, DeviceIndex = 0, KvCache = "fp8" };

        using Model model = Model.FromStore("mock-llama-it", options);

        Assert.Equal("llama", model.Family);
        Assert.Equal(1024, model.ContextLength);
    }

    [Fact]
    public void AnUnknownKvCacheIsAnArgumentException()
    {
        LoadOptions options = new LoadOptions { KvCache = "int4" };

        ArgumentException exception = Assert.Throws<ArgumentException>(() => Model.FromStore("mock-llama-it", options));

        Assert.Contains("'int4'", exception.Message);
    }

    [Fact]
    public void AnUnknownNameIsNotFound()
    {
        MilaException exception = Assert.Throws<MilaException>(() => Model.FromStore("no-such-model"));

        Assert.Equal(MilaStatus.NotFound, exception.Status);
        Assert.Contains("'no-such-model'", exception.Message);
    }

    [Fact]
    public void AnIncompleteModelIsRefused()
    {
        MilaException exception = Assert.Throws<MilaException>(() => Model.FromStore("mock-incomplete"));

        Assert.Equal(MilaStatus.Refused, exception.Status);
        Assert.Contains("'mock-incomplete'", exception.Message);
    }

    [Fact]
    public void GenerateStreamsTheReplyAndStops()
    {
        using Model model = Model.FromStore("mock-llama-it");
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-llama-it");
        int[] prompt = tokenizer.Encode("Hello?");
        List<int> tokens = [];

        FinishReason reason = model.Generate(prompt, tokens.Add, null, Cancellation);

        Assert.Equal(FinishReason.Stop, reason);
        int[] generated = tokens.ToArray();
        string text = tokenizer.Decode(generated);
        Assert.Equal(MockReply, text);
    }

    [Fact]
    public void EveryKnobCrosses()
    {
        using Model model = Model.FromStore("mock-gemma-it");
        GenerateOptions options = new GenerateOptions
        {
            MaxNewTokens = 128,
            Temperature = 0.6f,
            TopK = 40,
            TopP = 0.9f,
            PresencePenalty = 0.5f,
            Thinking = true,
            Seed = 42,
        };
        List<int> tokens = [];

        FinishReason reason = model.Generate(HelloPrompt, tokens.Add, options, Cancellation);

        Assert.Equal(FinishReason.Stop, reason);
        Assert.Equal(MockReply.Length, tokens.Count);
    }

    [Fact]
    public void AKnobOutOfRangeIsAnArgumentException()
    {
        using Model model = Model.FromStore("mock-gemma-it");
        GenerateOptions options = new GenerateOptions { TopP = 1.5f };

        ArgumentException exception = Assert.Throws<ArgumentException>(() => model.Generate(HelloPrompt, Ignore, options, Cancellation));

        Assert.Contains("top_p", exception.Message);
    }

    [Fact]
    public void MaxNewTokensEndsTheReplyWithLength()
    {
        using Model model = Model.FromStore("mock-llama-it");
        GenerateOptions options = new GenerateOptions { MaxNewTokens = 5 };
        List<int> tokens = [];

        FinishReason reason = model.Generate(HelloPrompt, tokens.Add, options, Cancellation);

        Assert.Equal(FinishReason.Length, reason);
        Assert.Equal(5, tokens.Count);
    }

    [Fact]
    public void AFullContextEndsTheReplyWithContextLimit()
    {
        LoadOptions load = new LoadOptions { ContextLength = 8 };
        using Model model = Model.FromStore("mock-llama-it", load);
        int[] prompt = [10, 11, 12];
        List<int> tokens = [];

        FinishReason reason = model.Generate(prompt, tokens.Add, null, Cancellation);

        Assert.Equal(FinishReason.ContextLimit, reason);
        Assert.Equal(5, tokens.Count);
    }

    [Fact]
    public void ALibraryFailureIsAMilaExceptionWithItsMessage()
    {
        using Model model = Model.FromStore("mock-llama-it");
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-llama-it");
        int[] prompt = tokenizer.Encode("!fail");

        MilaException exception = Assert.Throws<MilaException>(() => model.Generate(prompt, Ignore, null, Cancellation));

        Assert.Equal(MilaStatus.Runtime, exception.Status);
        Assert.Contains("'!'", exception.Message);
    }

    [Fact]
    public void AnEmptyPromptIsAnArgumentException()
    {
        using Model model = Model.FromStore("mock-llama-it");

        int[] empty = [];

        Assert.Throws<ArgumentException>(() => model.Generate(empty, Ignore, null, Cancellation));
    }

    [Fact]
    public void ACallbackThatThrowsEndsGenerateWithThatException()
    {
        using Model model = Model.FromStore("mock-llama-it");
        InvalidTimeZoneException thrown = new InvalidTimeZoneException("from the callback");
        int calls = 0;

        InvalidTimeZoneException caught = Assert.Throws<InvalidTimeZoneException>(() => model.Generate(HelloPrompt, ThrowOnThirdToken, null, Cancellation));

        Assert.Same(thrown, caught);
        Assert.Equal(3, calls);

        FinishReason after = model.Generate(HelloPrompt, Ignore, null, Cancellation);
        Assert.Equal(FinishReason.Stop, after);

        void ThrowOnThirdToken(int token)
        {
            calls++;

            if (calls == 3)
            {
                throw thrown;
            }
        }
    }

    [Fact]
    public void CancellingFromAnotherThreadReturnsCancelled()
    {
        using Model model = Model.FromStore("mock-llama-it");
        using ManualResetEventSlim firstToken = new ManualResetEventSlim();
        using ManualResetEventSlim cancelled = new ManualResetEventSlim();
        using CancellationTokenSource source = new CancellationTokenSource();
        List<int> tokens = [];
        Thread canceller = new Thread(CancelAfterFirstToken);
        canceller.Start();

        FinishReason reason = model.Generate(HelloPrompt, HoldFirstToken, null, source.Token);
        canceller.Join();

        Assert.Equal(FinishReason.Cancelled, reason);
        Assert.Single(tokens);

        void HoldFirstToken(int token)
        {
            tokens.Add(token);
            firstToken.Set();
            Assert.True(cancelled.Wait(Patience));
        }

        void CancelAfterFirstToken()
        {
            Assert.True(firstToken.Wait(Patience));
            source.Cancel();
            cancelled.Set();
        }
    }

    [Fact]
    public void AnAlreadyCancelledTokenReturnsCancelledBeforeAnyToken()
    {
        using Model model = Model.FromStore("mock-llama-it");
        CancellationToken token = new CancellationToken(canceled: true);
        List<int> tokens = [];

        FinishReason reason = model.Generate(HelloPrompt, tokens.Add, null, token);

        Assert.Equal(FinishReason.Cancelled, reason);
        Assert.Empty(tokens);
    }

    [Fact]
    public void ASecondGenerateFromAnotherThreadIsRefused()
    {
        using Model model = Model.FromStore("mock-llama-it");
        using ManualResetEventSlim firstToken = new ManualResetEventSlim();
        using ManualResetEventSlim attempted = new ManualResetEventSlim();
        Exception? refusal = null;
        Thread second = new Thread(GenerateAlongside);
        second.Start();

        FinishReason reason = model.Generate(HelloPrompt, HoldFirstToken, null, Cancellation);
        second.Join();

        Assert.Equal(FinishReason.Stop, reason);
        Assert.IsType<InvalidOperationException>(refusal);

        void HoldFirstToken(int token)
        {
            firstToken.Set();
            attempted.Wait(Patience);
        }

        void GenerateAlongside()
        {
            firstToken.Wait(Patience);
            refusal = Record.Exception(() => model.Generate(HelloPrompt, Ignore, null, Cancellation));
            attempted.Set();
        }
    }

    [Fact]
    public void GenerateFromItsOwnCallbackIsRefused()
    {
        using Model model = Model.FromStore("mock-llama-it");

        Assert.Throws<InvalidOperationException>(() => model.Generate(HelloPrompt, GenerateAgain, null, Cancellation));

        void GenerateAgain(int token)
        {
            model.Generate(HelloPrompt, Ignore, null, Cancellation);
        }
    }

    [Fact]
    public void TheCallbackMayDecodeWithAnotherHandle()
    {
        using Model model = Model.FromStore("mock-gemma-it");
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");
        List<string> pieces = [];

        FinishReason reason = model.Generate(HelloPrompt, DecodeEach, null, Cancellation);

        Assert.Equal(FinishReason.Stop, reason);
        Assert.Equal(MockReply, string.Concat(pieces));

        void DecodeEach(int token)
        {
            int[] single = [token];
            string piece = tokenizer.Decode(single);
            pieces.Add(piece);
        }
    }

    [Fact]
    public void ADisposedModelRefusesGenerate()
    {
        Model model = Model.FromStore("mock-llama-it");
        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => model.Generate(HelloPrompt, Ignore, null, Cancellation));
    }

    private static CancellationToken Cancellation => TestContext.Current.CancellationToken;


    private static void Ignore(int token)
    {
    }
}
