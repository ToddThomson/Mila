using System.Runtime.ExceptionServices;

namespace Mila;

/// <summary>
/// One <see cref="Model.Generate"/> call as its token callback sees it. An exception from the caller's
/// callback is kept here and the generation stopped, so nothing unwinds through native frames.
/// </summary>
internal sealed class GenerationState
{
    private readonly Action<int> onToken;
    private readonly StopSourceHandle stop;
    private ExceptionDispatchInfo? failure;

    internal GenerationState(Action<int> onToken, StopSourceHandle stop)
    {
        this.onToken = onToken;
        this.stop = stop;
    }

    internal void Deliver(int token)
    {
        if (failure is not null)
        {
            return;
        }

        try
        {
            onToken(token);
        }
        catch (Exception exception)
        {
            failure = ExceptionDispatchInfo.Capture(exception);
            NativeMethods.mila_stop_source_request_stop(stop);
        }
    }

    internal void ThrowIfCallbackFailed()
    {
        failure?.Throw();
    }
}
