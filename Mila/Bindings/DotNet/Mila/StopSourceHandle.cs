using System.Runtime.InteropServices;

namespace Mila;

/// <summary>A mila_stop_source, released by mila_stop_source_free.</summary>
internal sealed class StopSourceHandle : SafeHandle
{
    // Public and parameterless: the source-generated marshaller creates an out handle with Activator.CreateInstance.
    public StopSourceHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.mila_stop_source_free(handle);

        return true;
    }
}
