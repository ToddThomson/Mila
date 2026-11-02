namespace Mila;

/// <summary>An installed model's tokenizer: text to token ids and back.</summary>
public sealed unsafe class Tokenizer : IDisposable
{
    private readonly TokenizerHandle handle;

    private Tokenizer(TokenizerHandle handle)
    {
        this.handle = handle;
    }

    /// <summary>Opens the tokenizer of the installed model of that name.</summary>
    /// <param name="name">The model's name in the store.</param>
    /// <exception cref="MilaException">No installed model has that name (<see cref="MilaStatus.NotFound"/>).</exception>
    public static Tokenizer FromStore(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        AbiVersion.EnsureCompatible();
        int status = NativeMethods.mila_tokenizer_from_store(name, out TokenizerHandle tokenizer);
        NativeResult.ThrowIfFailed(status);

        return new Tokenizer(tokenizer);
    }

    /// <summary>The vocabulary's end-of-sequence id, or null when it has none.</summary>
    public int? EosTokenId
    {
        get
        {
            int status = NativeMethods.mila_tokenizer_eos_token_id(handle, out int id, out int present);
            NativeResult.ThrowIfFailed(status);

            return present != 0 ? id : null;
        }
    }

    /// <summary>The token ids of the text.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        int status = NativeMethods.mila_tokenizer_encode(handle, text, out nint tokens, out long count);
        NativeResult.ThrowIfFailed(status);

        return NativeResult.TakeTokens(tokens, count);
    }

    /// <summary>The text of the token ids.</summary>
    /// <exception cref="System.Text.DecoderFallbackException">
    /// The ids end part-way through a character. A streaming caller holds them and decodes again when the next id arrives.
    /// </exception>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        int status;
        nint text;

        fixed (int* ids = tokens)
        {
            status = NativeMethods.mila_tokenizer_decode(handle, ids, tokens.Length, out text);
        }

        NativeResult.ThrowIfFailed(status);

        return NativeResult.TakeString(text);
    }

    /// <summary>Releases the tokenizer.</summary>
    public void Dispose()
    {
        handle.Dispose();
    }
}
