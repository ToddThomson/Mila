using System.Runtime.InteropServices;

namespace Mila;

/// <summary>mila_generate_options, field for field; sequential layout gives C's natural alignment.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeGenerateOptions
{
    internal const uint SetTemperature = 1;
    internal const uint SetTopK = 2;
    internal const uint SetTopP = 4;
    internal const uint SetPresencePenalty = 8;
    internal const uint SetThinking = 16;
    internal const uint SetSeed = 32;

    public uint StructSize;
    public uint Set;
    public long MaxNewTokens;
    public float Temperature;
    public int TopK;
    public float TopP;
    public float PresencePenalty;
    public int Thinking;
    public ulong Seed;

    /// <summary>The options with a mask bit for each knob given, so a knob left null stays the publisher's.</summary>
    internal static unsafe NativeGenerateOptions From(GenerateOptions options)
    {
        NativeGenerateOptions native = new NativeGenerateOptions();
        native.StructSize = (uint)sizeof(NativeGenerateOptions);
        native.MaxNewTokens = options.MaxNewTokens;

        if (options.Temperature is float temperature)
        {
            native.Set |= SetTemperature;
            native.Temperature = temperature;
        }

        if (options.TopK is int topK)
        {
            native.Set |= SetTopK;
            native.TopK = topK;
        }

        if (options.TopP is float topP)
        {
            native.Set |= SetTopP;
            native.TopP = topP;
        }

        if (options.PresencePenalty is float presencePenalty)
        {
            native.Set |= SetPresencePenalty;
            native.PresencePenalty = presencePenalty;
        }

        if (options.Thinking is bool thinking)
        {
            native.Set |= SetThinking;
            native.Thinking = thinking ? 1 : 0;
        }

        if (options.Seed is ulong seed)
        {
            native.Set |= SetSeed;
            native.Seed = seed;
        }

        return native;
    }
}
