/**
 * @file CudaDeviceMemoryResourceTraits.ixx
 * @brief CUDA-specific memory resource traits and specializations
 */

module;
#include <type_traits>

export module Compute.CudaDeviceMemoryResourceTraits;

import Compute.CudaDeviceMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.MemoryResourceTraits;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA device memory resource traits providing detailed GPU backend characteristics
     *
     * Extends base MemoryResourceTraits with CUDA-specific capabilities and optimizations.
     * Used for CUDA backend dispatch and optimization decisions.
     */
    export template<>
        struct MemoryResourceTraits<CudaDeviceMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;

        // CUDA-specific traits
        static constexpr bool supports_unified_memory = false;
        static constexpr bool supports_texture_memory = true;
        static constexpr bool supports_shared_memory = true;
        static constexpr bool supports_constant_memory = true;
        static constexpr bool supports_async_operations = true;
        static constexpr bool supports_peer_access = true;

        // Memory characteristics
        static constexpr size_t preferred_alignment = 256;  // Optimal for coalesced access
        static constexpr size_t warp_size = 32;
        static constexpr bool requires_host_synchronization = true;
        static constexpr bool supports_zero_copy = false;

        // Performance characteristics
        static constexpr bool low_latency_access = false;  // High latency from host
        static constexpr bool high_bandwidth = true;       // Very high bandwidth
        static constexpr bool supports_atomic_operations = true;
        static constexpr bool requires_context_binding = true;

        // Access patterns
        static constexpr bool host_accessible = false;
        static constexpr bool device_accessible = true;
        static constexpr bool supports_concurrent_kernels = true;
    };

    /**
     * @brief CUDA managed memory resource traits providing unified memory characteristics
     */
    export template<>
        struct MemoryResourceTraits<CudaManagedMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;

        // CUDA managed-specific traits
        static constexpr bool supports_unified_memory = true;
        static constexpr bool supports_texture_memory = true;
        static constexpr bool supports_shared_memory = true;
        static constexpr bool supports_constant_memory = true;
        static constexpr bool supports_async_operations = true;
        static constexpr bool supports_peer_access = true;
        static constexpr bool supports_prefetching = true;
        static constexpr bool supports_memory_advise = true;

        // Memory characteristics
        static constexpr size_t preferred_alignment = 256;
        static constexpr size_t warp_size = 32;
        static constexpr bool requires_host_synchronization = false;  // Automatic migration
        static constexpr bool supports_zero_copy = true;             // Unified addressing

        // Performance characteristics
        static constexpr bool low_latency_access = false;  // Migration overhead
        static constexpr bool high_bandwidth = true;
        static constexpr bool supports_atomic_operations = true;
        static constexpr bool requires_context_binding = true;

        // Access patterns
        static constexpr bool host_accessible = true;
        static constexpr bool device_accessible = true;
        static constexpr bool supports_concurrent_kernels = true;
        static constexpr bool automatic_migration = true;
    };

    /**
     * @brief CUDA pinned memory resource traits providing fast transfer characteristics
     */
    export template<>
        struct MemoryResourceTraits<CudaPinnedMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;

        // CUDA pinned-specific traits
        static constexpr bool supports_unified_memory = false;
        static constexpr bool supports_texture_memory = false;  // Host memory
        static constexpr bool supports_shared_memory = false;   // Host memory
        static constexpr bool supports_constant_memory = false; // Host memory
        static constexpr bool supports_async_operations = true;
        static constexpr bool supports_peer_access = false;     // Host memory
        static constexpr bool is_page_locked = true;
        static constexpr bool optimized_for_transfer = true;

        // Memory characteristics
        static constexpr size_t preferred_alignment = 64;   // Host cache line
        static constexpr size_t warp_size = 32;             // For kernel design
        static constexpr bool requires_host_synchronization = false;
        static constexpr bool supports_zero_copy = true;    // Direct GPU access possible

        // Performance characteristics
        static constexpr bool low_latency_access = true;    // From host perspective
        static constexpr bool high_bandwidth = false;       // Host memory bandwidth
        static constexpr bool supports_atomic_operations = true;
        static constexpr bool requires_context_binding = false;  // Host accessible
        static constexpr bool fast_host_device_transfer = true;

        // Access patterns
        static constexpr bool host_accessible = true;
        static constexpr bool device_accessible = true;
        static constexpr bool supports_concurrent_kernels = true;
    };

    // ====================================================================
    // CUDA-Specific Helper Concepts
    // ====================================================================

    /**
     * @brief Concept for CUDA memory resources with unified memory support
     */
    export template<typename TMemoryResource>
        concept SupportsUnifiedMemory =
        MemoryResourceTraits<TMemoryResource>::is_cuda_resource &&
        MemoryResourceTraits<TMemoryResource>::supports_unified_memory;

    /**
     * @brief Concept for memory resources optimized for device-to-device transfers
     */
    export template<typename TMemoryResource>
        concept SupportsPeerAccess =
        MemoryResourceTraits<TMemoryResource>::supports_peer_access;

    /**
     * @brief Concept for memory resources with high bandwidth characteristics
     */
    export template<typename TMemoryResource>
        concept IsHighBandwidth =
        MemoryResourceTraits<TMemoryResource>::high_bandwidth;

    /**
     * @brief Concept for memory resources requiring CUDA context binding
     */
    export template<typename TMemoryResource>
        concept RequiresContextBinding =
        MemoryResourceTraits<TMemoryResource>::requires_context_binding;

    /**
     * @brief Concept for memory resources supporting texture memory access
     */
    export template<typename TMemoryResource>
        concept SupportsTextureMemory =
        MemoryResourceTraits<TMemoryResource>::supports_texture_memory;

    /**
     * @brief Concept for memory resources optimized for coalesced access patterns
     */
    export template<typename TMemoryResource>
        concept OptimizedForCoalescing =
        MemoryResourceTraits<TMemoryResource>::is_cuda_resource &&
        MemoryResourceTraits<TMemoryResource>::preferred_alignment >= 128;

    /**
     * @brief Concept for memory resources supporting concurrent kernel execution
     */
    export template<typename TMemoryResource>
        concept SupportsConcurrentKernels =
        MemoryResourceTraits<TMemoryResource>::supports_concurrent_kernels;
}