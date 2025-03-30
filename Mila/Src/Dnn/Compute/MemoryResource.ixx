/**
 * @file MemoryResource.ixx
 * @brief Defines a common memory resource abstraction for the compute framework.
 */

module;
#include <memory_resource>

export module Compute.MemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief An alias for the standard polymorphic memory resource.
     *
     * This provides a common abstraction for memory allocation and management
     * across different compute devices and memory types. The memory_resource
     * is the foundation for all memory allocations within the compute framework
     * and can be extended for specific devices (CPU, CUDA, etc.).
     *
     * @see std::pmr::memory_resource
     * @see HostMemoryResource
     * @see DeviceMemoryResource
     */
    export using MemoryResource = std::pmr::memory_resource;
}