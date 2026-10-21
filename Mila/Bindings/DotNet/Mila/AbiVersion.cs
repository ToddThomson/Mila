namespace Mila;

/// <summary>Refuses a MilaNative built for another ABI version, checked once, on first use.</summary>
internal static class AbiVersion
{
    /// <summary>MILA_ABI_VERSION in the mila.h this library is written against.</summary>
    internal const int Expected = 0;

    private static volatile bool verified;

    internal static void EnsureCompatible()
    {
        if (verified)
        {
            return;
        }

        int actual = NativeMethods.mila_abi_version();
        Verify(actual);
        verified = true;
    }

    internal static void Verify(int actual)
    {
        if (actual == Expected)
        {
            return;
        }

        string message = $"MilaNative implements ABI version {actual}, and this Mila library is built for version {Expected}.";

        throw new MilaException(MilaStatus.Abi, message);
    }
}
