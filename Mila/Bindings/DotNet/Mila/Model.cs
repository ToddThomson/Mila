using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;

namespace Mila;

/// <summary>An installed model, loaded on a CUDA device.</summary>
public sealed unsafe class Model : IDisposable
{
    private readonly ModelHandle handle;
    private int generating;

    private Model(ModelHandle handle)
    {
        this.handle = handle;
    }

    /// <summary>Loads the installed model of that name; its record decides the family and the weights' format.</summary>
    /// <param name="name">The model's name in the store.</param>
    /// <param name="options">How it runs on the device; null for every default.</param>
    /// <exception cref="MilaException">
    /// No installed model has that name (<see cref="MilaStatus.NotFound"/>), or it does not fit the device or
    /// cannot be loaded by this build (<see cref="MilaStatus.Refused"/>).
    /// </exception>
    public static Model FromStore(string name, LoadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(name);
        AbiVersion.EnsureCompatible();
        LoadOptions effective = options ?? new LoadOptions();
        byte* kvCache = Utf8StringMarshaller.ConvertToUnmanaged(effective.KvCache);

        try
        {
            NativeLoadOptions native = new NativeLoadOptions();
            native.StructSize = (uint)sizeof(NativeLoadOptions);
            native.ContextLength = effective.ContextLength ?? 0;
            native.DeviceIndex = effective.DeviceIndex;
            native.KvCache = (nint)kvCache;
            int status = NativeMethods.mila_model_from_store(name, in native, out ModelHandle model);
            NativeResult.ThrowIfFailed(status);

            return new Model(model);
        }
        finally
        {
            Utf8StringMarshaller.Free(kvCache);
        }
    }

    /// <summary>The native handle, for tests that cross the ABI with what this class would never send.</summary>
    internal ModelHandle Handle => handle;

    /// <summary>Tokens the model holds, prompt and reply together.</summary>
    public long ContextLength
    {
        get
        {
            int status = NativeMethods.mila_model_context_length(handle, out long contextLength);
            NativeResult.ThrowIfFailed(status);

            return contextLength;
        }
    }

    /// <summary>"gemma", "llama" or "qwen".</summary>
    public string Family
    {
        get
        {
            int status = NativeMethods.mila_model_family(handle, out nint family);
            NativeResult.ThrowIfFailed(status);

            return NativeResult.TakeString(family);
        }
    }

    /// <summary>
    /// Generates a reply, calling <paramref name="onToken"/> on this thread with each token id before it returns.
    /// </summary>
    /// <param name="prompt">The prompt's token ids.</param>
    /// <param name="onToken">Called once per token. An exception it throws stops the reply and is rethrown here.</param>
    /// <param name="options">How to sample; null for the publisher's settings and 64 tokens at most.</param>
    /// <param name="cancellationToken">Stops the reply between tokens, from any thread.</param>
    /// <returns>Why the reply ended. A cancelled reply returns <see cref="FinishReason.Cancelled"/>; it does not throw.</returns>
    /// <exception cref="InvalidOperationException">Another <see cref="Generate"/> on this model is running, including from <paramref name="onToken"/>.</exception>
    public FinishReason Generate(
        ReadOnlySpan<int> prompt,
        Action<int> onToken,
        GenerateOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(onToken);
        ObjectDisposedException.ThrowIf(handle.IsClosed, this);
        int alreadyGenerating = Interlocked.Exchange(ref generating, 1);

        if (alreadyGenerating != 0)
        {
            throw new InvalidOperationException("This model is already generating; a model runs one Generate at a time.");
        }

        try
        {
            GenerateOptions effective = options ?? new GenerateOptions();
            NativeGenerateOptions native = NativeGenerateOptions.From(effective);

            return GenerateAlone(prompt, onToken, native, cancellationToken);
        }
        finally
        {
            Volatile.Write(ref generating, 0);
        }
    }

    /// <summary>Releases the model and the device memory it holds.</summary>
    public void Dispose()
    {
        handle.Dispose();
    }

    private FinishReason GenerateAlone(
        ReadOnlySpan<int> prompt,
        Action<int> onToken,
        in NativeGenerateOptions options,
        CancellationToken cancellationToken)
    {
        int createStatus = NativeMethods.mila_stop_source_create(out StopSourceHandle stop);
        NativeResult.ThrowIfFailed(createStatus);

        using (stop)
        {
            GenerationState state = new GenerationState(onToken, stop);
            GCHandle stateHandle = GCHandle.Alloc(state);

            try
            {
                int status;
                int reason;

                // Disposed before the stop source: Dispose waits for a cancellation callback already running.
                using (CancellationTokenRegistration registration = cancellationToken.Register(RequestStop, stop))
                {
                    nint userData = GCHandle.ToIntPtr(stateHandle);

                    fixed (int* promptTokens = prompt)
                    {
                        status = NativeMethods.mila_model_generate(handle, promptTokens, prompt.Length, in options, &OnToken, userData, stop, out reason);
                    }
                }

                string message = status == (int)MilaStatus.Ok ? string.Empty : NativeResult.LastErrorMessage();
                state.ThrowIfCallbackFailed();

                if (status != (int)MilaStatus.Ok)
                {
                    throw NativeResult.CreateException(status, message);
                }

                return (FinishReason)reason;
            }
            finally
            {
                stateHandle.Free();
            }
        }
    }

    private static void RequestStop(object? stop)
    {
        NativeMethods.mila_stop_source_request_stop((StopSourceHandle)stop!);
    }

    [UnmanagedCallersOnly]
    private static void OnToken(int token, nint userData)
    {
        GCHandle stateHandle = GCHandle.FromIntPtr(userData);
        GenerationState state = (GenerationState)stateHandle.Target!;
        state.Deliver(token);
    }
}
