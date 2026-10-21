namespace Mila;

/// <summary>How much the library logs.</summary>
public enum LogLevel
{
    /// <summary>Everything, for diagnosing the library itself.</summary>
    Trace,

    /// <summary>Progress such as loading.</summary>
    Info,

    /// <summary>Only what may need attention.</summary>
    Warning,

    /// <summary>Only failures.</summary>
    Error,
}
