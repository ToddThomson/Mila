namespace Mila;

/// <summary>
/// How <see cref="Model.Generate"/> samples. Each knob left null is the model publisher's setting for
/// whether the prompt was rendered with thinking on.
/// </summary>
public sealed record GenerateOptions
{
    /// <summary>The most tokens the reply may hold.</summary>
    public long MaxNewTokens { get; init; } = 64;

    /// <summary>0 is greedy.</summary>
    public float? Temperature { get; init; }

    /// <summary>0 turns the filter off.</summary>
    public int? TopK { get; init; }

    /// <summary>1 turns the filter off.</summary>
    public float? TopP { get; init; }

    /// <summary>Subtracted from each token the reply has already used.</summary>
    public float? PresencePenalty { get; init; }

    /// <summary>Whether the prompt was rendered with thinking on; null for the chat template's default.</summary>
    public bool? Thinking { get; init; }

    /// <summary>
    /// Reseeds the sampler and prefills the whole prompt, so the same prompt, settings and seed give the
    /// same reply whatever ran before. Null continues from the last reply.
    /// </summary>
    public ulong? Seed { get; init; }
}
