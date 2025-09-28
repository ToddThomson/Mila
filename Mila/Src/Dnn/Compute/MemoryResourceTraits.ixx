/**
 * @file MemoryResourceTraits.ixx
 * @brief Compute backend memory resource traits for dispatch optimization
 */

module;
#include <type_traits>

export module Compute.MemoryResourceTraits;

import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Memory resource traits for compile-time dispatch optimization
     * 
     * Provides compile-time classification of memory resource types to enable
     * efficient dispatch in tensor initialization and operations. Part of the
     * compute backend abstraction layer.
     */
    export template<typename TMemoryResource>
    struct MemoryResourceTraits {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = false;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;
    };

    /**
     * @brief CPU Memory Resource Specialization
     */
    template<>
    struct MemoryResourceTraits<CpuMemoryResource> {
        static constexpr bool is_cpu_resource = true;
        static constexpr bool is_cuda_resource = false;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;
    };

    /**
     * @brief CUDA Device Memory Resource Specialization
     */
    template<>
    struct MemoryResourceTraits<CudaMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;
    };

    /**
     * @brief CUDA Managed Memory Resource Specialization
     */
    template<>
    struct MemoryResourceTraits<CudaManagedMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;
    };

    /**
     * @brief CUDA Pinned Memory Resource Specialization
     */
    template<>
    struct MemoryResourceTraits<CudaPinnedMemoryResource> {
        static constexpr bool is_cpu_resource = false;
        static constexpr bool is_cuda_resource = true;
        static constexpr bool is_metal_resource = false;
        static constexpr bool is_opencl_resource = false;
        static constexpr bool is_vulkan_resource = false;
    };

    // ====================================================================
    // Convenience Helper Concepts
    // ====================================================================

    /**
     * @brief Concept for identifying device-based memory resources
     */
    export template<typename TMemoryResource>
    concept IsDeviceMemoryResource = 
        MemoryResourceTraits<TMemoryResource>::is_cuda_resource ||
        MemoryResourceTraits<TMemoryResource>::is_metal_resource ||
        MemoryResourceTraits<TMemoryResource>::is_opencl_resource ||
        MemoryResourceTraits<TMemoryResource>::is_vulkan_resource;

    /**
     * @brief Concept for identifying host-accessible memory resources
     */
    export template<typename TMemoryResource>
    concept IsHostMemoryResource = 
        MemoryResourceTraits<TMemoryResource>::is_cpu_resource;

    /**
     * @brief Concept for identifying compute accelerator memory resources
     */
    export template<typename TMemoryResource>
    concept IsAcceleratorMemoryResource = 
        MemoryResourceTraits<TMemoryResource>::is_cuda_resource ||
        MemoryResourceTraits<TMemoryResource>::is_metal_resource ||
        MemoryResourceTraits<TMemoryResource>::is_opencl_resource ||
        MemoryResourceTraits<TMemoryResource>::is_vulkan_resource;
}