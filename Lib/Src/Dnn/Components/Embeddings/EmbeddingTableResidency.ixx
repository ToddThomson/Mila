/**
 * @file EmbeddingTableResidency.ixx
 * @brief Where an embedding table's bytes live: the device's memory, or host memory read across the bus.
 */

export module Dnn.Components.EmbeddingTableResidency;

namespace Mila::Dnn
{
    /**
     * @brief Where the embedding table's bytes live.
     *
     * Device is what every other parameter in the tree does and stays the default.
     * Host places the table in pinned host memory, which is device-addressable under
     * unified virtual addressing, so the gather kernel reads rows across PCIe and is
     * unchanged -- the table simply costs no VRAM. It is worth doing only for a table
     * large enough to matter and only because a lookup is a gather and never a
     * multiply: the traffic is one row per token, not one pass per matrix. It means
     * something only on a device whose memory is separate from the host's
     * (DeviceTypeTraits::memory_separate_from_host).
     *
     * See Specifications/Qwen3.8.md section 5, which spends this on 1.271 B parameters
     * (2.37 GiB) to bring a 27B model inside a 12 GiB card.
     */
    export enum class EmbeddingTableResidency
    {
        Device,
        Host
    };
}
