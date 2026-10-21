using System.Reflection;

namespace Mila.Tests;

/// <summary>Section 3.2: mila_abi_version, mila_initialize, mila_version, mila_cuda_devices_json.</summary>
public sealed class RuntimeTests
{
    [Fact]
    public void AbiVersionIsTheOneThisLibraryExpects()
    {
        int actual = NativeMethods.mila_abi_version();

        Assert.Equal(AbiVersion.Expected, actual);
    }

    [Fact]
    public void AnotherAbiVersionIsRefusedNamingBoth()
    {
        MilaException exception = Assert.Throws<MilaException>(() => AbiVersion.Verify(7));

        Assert.Equal(MilaStatus.Abi, exception.Status);
        Assert.Contains("version 7", exception.Message);
        Assert.Contains($"version {AbiVersion.Expected}", exception.Message);
    }

    [Theory]
    [InlineData(LogLevel.Trace)]
    [InlineData(LogLevel.Info)]
    [InlineData(LogLevel.Warning)]
    [InlineData(LogLevel.Error)]
    public void InitializeAcceptsEveryLogLevel(LogLevel logLevel)
    {
        MilaRuntime.Initialize(logLevel);
    }

    [Fact]
    public void InitializeRefusesALevelItDoesNotKnow()
    {
        int status = NativeMethods.mila_initialize("verbose");
        string message = NativeResult.LastErrorMessage();

        Assert.Equal((int)MilaStatus.InvalidArgument, status);
        Assert.Contains("'verbose'", message);
    }

    [Fact]
    public void VersionIsVersionTxtAsTheAssemblyCarriesIt()
    {
        AssemblyInformationalVersionAttribute? expected = typeof(Model).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>();

        Assert.NotNull(expected);
        Assert.Equal(expected.InformationalVersion, MilaRuntime.Version);
    }

    [Fact]
    public void CudaDevicesAreListedOnceInitialized()
    {
        MilaRuntime.Initialize(LogLevel.Warning);

        IReadOnlyList<CudaDevice> devices = MilaRuntime.CudaDevices;

        CudaDevice device = Assert.Single(devices);
        Assert.Equal(0, device.Index);
        Assert.Equal("Mila mock device", device.Name);
        Assert.Equal(12, device.ComputeCapabilityMajor);
        Assert.Equal(0, device.ComputeCapabilityMinor);
        Assert.Equal(17179869184UL, device.TotalMemoryBytes);
        Assert.Equal(1, device.PciBus);
    }

    [Fact]
    public void ErrorMessageIsPerThreadAndClearedBySuccess()
    {
        int failed = NativeMethods.mila_initialize("verbose");
        string otherThreadMessage = "unset";
        Thread other = new Thread(ReadOtherThreadMessage);
        other.Start();
        other.Join();

        Assert.Equal((int)MilaStatus.InvalidArgument, failed);
        Assert.NotEqual(string.Empty, NativeResult.LastErrorMessage());
        Assert.Equal(string.Empty, otherThreadMessage);

        int succeeded = NativeMethods.mila_initialize("warning");

        Assert.Equal((int)MilaStatus.Ok, succeeded);
        Assert.Equal(string.Empty, NativeResult.LastErrorMessage());

        void ReadOtherThreadMessage()
        {
            otherThreadMessage = NativeResult.LastErrorMessage();
        }
    }
}
