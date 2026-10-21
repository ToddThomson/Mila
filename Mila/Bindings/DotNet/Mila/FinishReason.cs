namespace Mila;

/// <summary>Why <see cref="Model.Generate"/> stopped, which the tokens alone do not show.</summary>
public enum FinishReason
{
    /// <summary>The model finished on its own.</summary>
    Stop = 0,

    /// <summary><see cref="GenerateOptions.MaxNewTokens"/> was reached.</summary>
    Length = 1,

    /// <summary>The context was full.</summary>
    ContextLimit = 2,

    /// <summary>The cancellation token was cancelled.</summary>
    Cancelled = 3,
}
