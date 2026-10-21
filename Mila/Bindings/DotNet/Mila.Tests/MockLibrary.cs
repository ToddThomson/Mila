using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyFixture(typeof(Mila.Tests.MockLibrary))]

namespace Mila.Tests;

/// <summary>
/// Points the Mila assembly's "MilaNative" at the mock (DotNetBinding.md section 5), once, before any test runs.
/// The shipped library has no resolver; this is the test project's.
/// </summary>
public sealed class MockLibrary
{
    public MockLibrary()
    {
        Assembly mila = typeof(Model).Assembly;
        NativeLibrary.SetDllImportResolver(mila, Resolve);
    }

    private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != NativeMethods.Library)
        {
            return nint.Zero;
        }

        string path = Path.Combine(Directory(), FileName());

        if (!File.Exists(path))
        {
            string message = $"The mock MilaNative is not built at {path}. Build it first: cmake -S Mock -B Mock/build && cmake --build Mock/build";

            throw new DllNotFoundException(message);
        }

        return NativeLibrary.Load(path);
    }

    private static string Directory()
    {
        IEnumerable<AssemblyMetadataAttribute> metadata = typeof(MockLibrary).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>();
        AssemblyMetadataAttribute directory = metadata.Single(IsMockDirectory);

        return directory.Value!;
    }

    private static bool IsMockDirectory(AssemblyMetadataAttribute attribute) => attribute.Key == "MilaNativeMockDirectory";

    private static string FileName()
    {
        if (OperatingSystem.IsWindows())
        {
            return "MilaNative.dll";
        }

        if (OperatingSystem.IsMacOS())
        {
            return "libMilaNative.dylib";
        }

        return "libMilaNative.so";
    }
}
