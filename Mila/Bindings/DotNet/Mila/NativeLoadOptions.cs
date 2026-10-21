using System.Runtime.InteropServices;

namespace Mila;

/// <summary>mila_load_options, field for field; sequential layout gives C's natural alignment.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeLoadOptions
{
    public uint StructSize;
    public long ContextLength;
    public int DeviceIndex;

    /// <summary>A UTF-8 string the caller owns for the length of the call, or zero for "bf16".</summary>
    public nint KvCache;
}
