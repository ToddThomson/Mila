using System.Text;

namespace Mila.Tests;

/// <summary>Section 3.3: mila_tokenizer_from_store, _encode, _decode, _eos_token_id and _free over the byte tokenizer.</summary>
public sealed class TokenizerTests
{
    [Fact]
    public void EncodeGivesATokenPerByteOffsetByThree()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-llama-it");

        int[] expected = ['H' + 3, 'i' + 3, '!' + 3];

        int[] tokens = tokenizer.Encode("Hi!");

        Assert.Equal(expected, tokens);
    }

    [Fact]
    public void EncodeOfEmptyTextIsEmpty()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-llama-it");

        int[] tokens = tokenizer.Encode(string.Empty);

        Assert.Empty(tokens);
    }

    [Theory]
    [InlineData("plain ASCII")]
    [InlineData("h\u00e9llo, w\u00f6rld \u2713 \U0001F600")]
    public void DecodeInvertsEncode(string text)
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");
        int[] tokens = tokenizer.Encode(text);

        string decoded = tokenizer.Decode(tokens);

        int byteCount = Encoding.UTF8.GetByteCount(text);
        Assert.Equal(byteCount, tokens.Length);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void ControlTokensDecodeToNothing()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");

        int[] tokens = [0, 'a' + 3, 1, 2];

        string decoded = tokenizer.Decode(tokens);

        Assert.Equal("a", decoded);
    }

    [Fact]
    public void DecodeOfNoTokensIsEmpty()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");

        int[] tokens = [];

        string decoded = tokenizer.Decode(tokens);

        Assert.Equal(string.Empty, decoded);
    }

    [Fact]
    public void DecodeEndingPartWayThroughACharacterThrows()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");
        int[] tokens = tokenizer.Encode("\u2713");
        int[] partial = tokens[..2];

        Assert.Throws<DecoderFallbackException>(() => tokenizer.Decode(partial));
    }

    [Fact]
    public void AnIdOutsideTheVocabularyIsAnArgumentException()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");

        int[] tokens = [999];

        ArgumentException exception = Assert.Throws<ArgumentException>(() => tokenizer.Decode(tokens));

        Assert.Contains("999", exception.Message);
    }

    [Fact]
    public void EosTokenIdIsOne()
    {
        using Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");

        Assert.Equal(1, tokenizer.EosTokenId);
    }

    [Fact]
    public void AnUnknownNameIsNotFound()
    {
        MilaException exception = Assert.Throws<MilaException>(() => Tokenizer.FromStore("no-such-model"));

        Assert.Equal(MilaStatus.NotFound, exception.Status);
        Assert.Contains("'no-such-model'", exception.Message);
    }

    [Fact]
    public void ADisposedTokenizerRefusesUse()
    {
        Tokenizer tokenizer = Tokenizer.FromStore("mock-gemma-it");
        tokenizer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => tokenizer.Encode("x"));
    }
}
