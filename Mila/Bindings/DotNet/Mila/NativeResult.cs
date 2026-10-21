using System.Runtime.InteropServices;
using System.Text;

namespace Mila;

/// <summary>Turns statuses into exceptions and takes ownership of what the library hands out.</summary>
internal static unsafe class NativeResult
{
    // Strict, as Python's str is: bytes that end part-way through a character throw rather than becoming U+FFFD.
    private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>Throws for a failed status, with the message read before any other call can replace it.</summary>
    internal static void ThrowIfFailed(int status)
    {
        if (status == (int)MilaStatus.Ok)
        {
            return;
        }

        string message = LastErrorMessage();

        throw CreateException(status, message);
    }

    internal static Exception CreateException(int status, string message)
    {
        if (status == (int)MilaStatus.InvalidArgument)
        {
            return new ArgumentException(message);
        }

        MilaStatus known = (MilaStatus)status;

        return new MilaException(known, message);
    }

    internal static string LastErrorMessage()
    {
        nint message = NativeMethods.mila_last_error_message();

        if (message == nint.Zero)
        {
            return string.Empty;
        }

        return Marshal.PtrToStringUTF8(message) ?? string.Empty;
    }

    /// <summary>Decodes a string the library allocated, then frees it.</summary>
    internal static string TakeString(nint text)
    {
        try
        {
            ReadOnlySpan<byte> bytes = MemoryMarshal.CreateReadOnlySpanFromNullTerminated((byte*)text);

            return StrictUtf8.GetString(bytes);
        }
        finally
        {
            NativeMethods.mila_string_free(text);
        }
    }

    /// <summary>Copies a token array the library allocated, then frees it.</summary>
    internal static int[] TakeTokens(nint tokens, long count)
    {
        try
        {
            int length = checked((int)count);
            ReadOnlySpan<int> source = new ReadOnlySpan<int>((int*)tokens, length);

            return source.ToArray();
        }
        finally
        {
            NativeMethods.mila_tokens_free(tokens);
        }
    }
}
