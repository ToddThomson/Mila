using System.Runtime.InteropServices;

namespace Mila;

/// <summary>A mila_tokenizer, released by mila_tokenizer_free.</summary>
internal sealed class TokenizerHandle : SafeHandle
{
    // Public and parameterless: the source-generated marshaller creates an out handle with Activator.CreateInstance.
    public TokenizerHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.mila_tokenizer_free(handle);

        return true;
    }
}
