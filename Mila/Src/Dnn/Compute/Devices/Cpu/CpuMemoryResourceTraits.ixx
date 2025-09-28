/**
 * @file CpuMemoryResourceTraits.ixx
 * @brief CPU-specific memory resource traits and specializations
 */

module;
#include <type_traits>

export module Compute.CpuMemoryResourceTraits;

import Compute.CpuMemoryResource;
import Compute.MemoryResourceTraits;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU-specific memory resource traits providing detailed CPU backend characteristics
     *
     * Extends base MemoryResourceTraits with CPU-specific capabilities and optimizations.
     * Used for CPU backend dispatch and optimization decisions.
     */
    export template<>
        struct MemoryResourceTraits<CpuMemoryResource> {
        static constexpr bool is_cpu_resource = true;
        static constexpr bool is_cuda_resource = false;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;

        // CPU-specific traits
        static constexpr bool supports_simd = true;
        static constexpr bool supports_threading = true;
        static constexpr bool supports_numa = true;
        static constexpr bool is_cache_coherent = true;
        static constexpr bool supports_aligned_allocation = true;

        // Memory characteristics
        static constexpr size_t preferred_alignment = 64;  // AVX-512 alignment
        static constexpr size_t cache_line_size = 64;
        static constexpr bool requires_host_synchronization = false;
        static constexpr bool supports_zero_copy = false;

        // Performance characteristics
        static constexpr bool low_latency_access = true;
        static constexpr bool high_bandwidth = false;  // Relative to GPU memory
        static constexpr bool supports_atomic_operations = true;
    };

    // ====================================================================
    // CPU-Specific Helper Concepts
    // ====================================================================

    /**
     * @brief Concept for CPU memory resources with SIMD support
     */
    export template<typename TMemoryResource>
        concept SupportsSIMD =
        MemoryResourceTraits<TMemoryResource>::is_cpu_resource &&
        MemoryResourceTraits<TMemoryResource>::supports_simd;

    /**
     * @brief Concept for memory resources with threading support
     */
    export template<typename TMemoryResource>
        concept SupportsThreading =
        MemoryResourceTraits<TMemoryResource>::supports_threading;

    /**
     * @brief Concept for cache-coherent memory resources
     */
    export template<typename TMemoryResource>
        concept IsCacheCoherent =
        MemoryResourceTraits<TMemoryResource>::is_cache_coherent;
}