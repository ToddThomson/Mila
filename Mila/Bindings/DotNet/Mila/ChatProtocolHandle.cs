using System.Runtime.InteropServices;

namespace Mila;

/// <summary>A mila_chat_protocol, released by mila_chat_protocol_free.</summary>
internal sealed class ChatProtocolHandle : SafeHandle
{
    // Public and parameterless: the source-generated marshaller creates an out handle with Activator.CreateInstance.
    public ChatProtocolHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.mila_chat_protocol_free(handle);

        return true;
    }
}
