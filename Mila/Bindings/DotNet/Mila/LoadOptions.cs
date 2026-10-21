namespace Mila;

/// <summary>How <see cref="Model.FromStore"/> runs a model on the device.</summary>
public sealed record LoadOptions
{
    /// <summary>Tokens the model holds, prompt and reply together; null for the longest the device can hold.</summary>
    public long? ContextLength { get; init; }

    /// <summary>The CUDA ordinal, as <see cref="CudaDevice.Index"/> reports it.</summary>
    public int DeviceIndex { get; init; }

    /// <summary>"bf16" or "fp8"; null for "bf16". FP8 holds the context in half the memory.</summary>
    public string? KvCache { get; init; }
}
