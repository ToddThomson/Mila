namespace Mila;

/// <summary>One CUDA device Mila can run on.</summary>
/// <param name="Index">The CUDA ordinal <see cref="LoadOptions.DeviceIndex"/> takes; not the index nvidia-smi prints, which orders by PCI bus.</param>
/// <param name="Name">The device's name.</param>
/// <param name="ComputeCapabilityMajor">The compute capability's major number.</param>
/// <param name="ComputeCapabilityMinor">The compute capability's minor number.</param>
/// <param name="TotalMemoryBytes">The device's memory.</param>
/// <param name="PciDomain">The PCI domain, which identifies the card in either ordering.</param>
/// <param name="PciBus">The PCI bus.</param>
/// <param name="PciDevice">The PCI device.</param>
public sealed record CudaDevice(
    int Index,
    string Name,
    int ComputeCapabilityMajor,
    int ComputeCapabilityMinor,
    ulong TotalMemoryBytes,
    int PciDomain,
    int PciBus,
    int PciDevice);
