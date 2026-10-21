using System.Text.Json;

namespace Mila;

/// <summary>The library itself: start it, ask its version, list the devices it can run on.</summary>
public static class MilaRuntime
{
    /// <summary>Starts the library. Call it once, before loading a model.</summary>
    /// <param name="logLevel">How much the library logs.</param>
    public static void Initialize(LogLevel logLevel)
    {
        string level = NameOf(logLevel);
        AbiVersion.EnsureCompatible();
        int status = NativeMethods.mila_initialize(level);
        NativeResult.ThrowIfFailed(status);
    }

    /// <summary>The library's version.</summary>
    public static string Version
    {
        get
        {
            AbiVersion.EnsureCompatible();
            int status = NativeMethods.mila_version(out nint version);
            NativeResult.ThrowIfFailed(status);

            return NativeResult.TakeString(version);
        }
    }

    /// <summary>Every CUDA device Mila can run on, in <see cref="CudaDevice.Index"/> order; empty before <see cref="Initialize"/>.</summary>
    public static IReadOnlyList<CudaDevice> CudaDevices
    {
        get
        {
            AbiVersion.EnsureCompatible();
            int status = NativeMethods.mila_cuda_devices_json(out nint json);
            NativeResult.ThrowIfFailed(status);
            string text = NativeResult.TakeString(json);
            IReadOnlyList<CudaDevice>? devices = JsonSerializer.Deserialize(text, MilaJsonContext.Default.IReadOnlyListCudaDevice);

            return devices ?? throw new MilaException(MilaStatus.Runtime, "The library listed no CUDA devices, not even an empty list.");
        }
    }

    private static string NameOf(LogLevel logLevel) => logLevel switch
    {
        LogLevel.Trace => "trace",
        LogLevel.Info => "info",
        LogLevel.Warning => "warning",
        LogLevel.Error => "error",
        _ => throw new ArgumentOutOfRangeException(nameof(logLevel), logLevel, "Not a log level."),
    };
}
