using System.Runtime.InteropServices;

namespace Mila;

/// <summary>A mila_model, released by mila_model_free.</summary>
internal sealed class ModelHandle : SafeHandle
{
    // Public and parameterless: the source-generated marshaller creates an out handle with Activator.CreateInstance.
    public ModelHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.mila_model_free(handle);

        return true;
    }
}
