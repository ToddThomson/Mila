namespace Mila;

/// <summary>A call into Mila failed for a reason other than an invalid argument.</summary>
public sealed class MilaException : Exception
{
    /// <summary>Creates the exception with the library's own message.</summary>
    /// <param name="status">Why the call failed.</param>
    /// <param name="message">The library's message.</param>
    public MilaException(MilaStatus status, string message)
        : base(message)
    {
        Status = status;
    }

    /// <summary>Why the call failed.</summary>
    public MilaStatus Status { get; }
}
