/**
 * @file Mila_py.Runtime.ixx
 * @brief What this build of the extension is and which CUDA devices it can run on, with a std-only surface.
 *
 * Kept apart from Mila.Bindings for the same reason that module exists: Mila_py.cpp must not import Mila, so
 * the Mila types stay in the implementation unit (Mila_py.Runtime.cpp).
 */

module;

#include <cstddef>
#include <string>
#include <vector>

export module Mila.Bindings.Runtime;

namespace Mila::Bindings
{
    /**
     * @brief One CUDA device as the runtime sees it.
     *
     * `index` is the CUDA ordinal that `from_store( ..., device_index )` takes. It is not the index nvidia-smi
     * prints, which orders by PCI bus; the PCI fields identify the card in both orderings.
     */
    export struct CudaDeviceInfo
    {
        int index{ 0 };
        std::string name;
        int compute_capability_major{ 0 };
        int compute_capability_minor{ 0 };
        std::size_t total_memory_bytes{ 0 };
        int pci_domain{ 0 };
        int pci_bus{ 0 };
        int pci_device{ 0 };
    };

    /// Every CUDA device the runtime registered, in CUDA ordinal order. Empty before initialize().
    export std::vector<CudaDeviceInfo> cudaDevices();

    /// The version of the library compiled into this extension, as Version.txt writes it.
    export std::string version();
}
