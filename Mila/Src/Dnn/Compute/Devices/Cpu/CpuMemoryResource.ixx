module;
#include <memory_resource>
#include <exception>
#include <memory>

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#else
#include <cstdlib>   // For std::aligned_alloc and std::free
#endif
#include <cstring>

export module Compute.CpuMemoryResource;

import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    export struct CpuComputeDeviceTag {};

    /**
     * @brief CPU memory resource for host-accessible memory allocation.
     *
     * Provides optimized CPU memory allocation with proper alignment support
     * for tensor data. CPU memory allocation doesn't require device selection,
     * so the device_id parameter is unused but maintained for interface consistency
     * with other memory resource types.
     */
    export class CpuMemoryResource : public MemoryResource {
    public:
        using ComputeDeviceTag = CpuComputeDeviceTag;
        static constexpr DeviceType device_type = DeviceType::Cpu;

        /**
         * @brief Constructs CPU memory resource.
         *
         * @param device_id Device identifier (unused for CPU, maintained for interface consistency)
         *
         * @note CPU memory allocation doesn't require device selection
         * @note Parameter is kept for consistency with CUDA memory resources
         */
        explicit CpuMemoryResource( [[maybe_unused]] int device_id = 0 ) {
            // CPU memory allocation doesn't require device ID
            // Parameter kept for interface consistency with CUDA
        }

        /**
         * @brief Indicates CPU memory is accessible from host code.
         */
        static constexpr bool is_host_accessible = HostAccessible::is_host_accessible;

        /**
         * @brief Indicates CPU memory is not accessible from device code.
         */
        static constexpr bool is_device_accessible = false;

    protected:
        /**
         * @brief Allocates aligned CPU memory.
         *
         * Uses platform-specific aligned allocation functions to ensure
         * proper memory alignment for optimal CPU performance and SIMD operations.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement
         * @return Pointer to allocated memory
         * @throws std::bad_alloc If allocation fails
         */
        void* do_allocate( std::size_t bytes, std::size_t alignment ) override {
            if (bytes == 0) {
                return nullptr;
            }

            // Ensure minimum alignment for the platform
            if (alignment < alignof(std::max_align_t)) {
                alignment = alignof(std::max_align_t);
            }

#ifdef _WIN32
            void* ptr = _aligned_malloc( bytes, alignment );
#else
            // Ensure alignment is power of 2 and >= sizeof(void*)
            if (alignment < sizeof( void* )) {
                alignment = sizeof( void* );
            }

            // aligned_alloc requires size to be multiple of alignment
            std::size_t aligned_size = ((bytes + alignment - 1) / alignment) * alignment;
            void* ptr = std::aligned_alloc( alignment, aligned_size );
#endif

            if (!ptr) {
                throw std::bad_alloc();
            }

            return ptr;
        }

        /**
         * @brief Deallocates CPU memory using platform-specific functions.
         *
         * Properly releases memory allocated with platform-specific aligned
         * allocation functions.
         *
         * @param ptr Pointer to memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate( void* ptr, std::size_t, std::size_t ) override {
            if (ptr) {
#ifdef _WIN32
                _aligned_free( ptr );
#else
                std::free( ptr );
#endif
            }
        }

        /**
         * @brief Compares CPU memory resources for equality.
         *
         * CPU memory resources are equal if they are both CpuMemoryResource
         * instances, since they all manage the same underlying host memory pool.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CpuMemoryResource instances
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return dynamic_cast<const CpuMemoryResource*>(&other) != nullptr;
        }
    };

    /**
     * @brief Alias for CpuMemoryResource that represents host-accessible memory.
     *
     * This alias provides a semantic name that describes the memory's accessibility
     * characteristics rather than its implementation details. Use HostMemoryResource
     * when you need memory that can be directly accessed from host (CPU) code.
     *
     * @see CpuMemoryResource
     */
    export using HostMemoryResource = CpuMemoryResource;
}