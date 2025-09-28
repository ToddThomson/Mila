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

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.CpuDeviceContext;
import Compute.DeviceContext;

namespace Mila::Dnn::Compute
{
	export struct CpuComputeDeviceTag {};

    /**
     * @brief CPU memory resource for host-accessible memory allocation.
     *
     * Provides optimized CPU memory allocation with proper alignment support
     * for tensor data. Focuses purely on memory allocation responsibilities
     * without tensor-specific operations or type conversions.
     */
    export class CpuMemoryResource : public MemoryResource {
    public:

		using ComputeDeviceTag = CpuComputeDeviceTag;
        using CompatibleDeviceContext = CpuDeviceContext;

        static constexpr bool isValidDeviceContext(const DeviceContext& device_context) {
            return dynamic_cast<const CpuDeviceContext*>(&device_context) != nullptr;
        }

        /**
         * @brief Constructor with device context for interface consistency.
         *
         * Accepts a device context for consistency with other memory resource types,
         * though CPU memory resource doesn't require device context functionality.
         * The device context is stored but not actively used for CPU operations.
         *
         * @param device_context Device context (can be any type, stored for consistency)
         */
        explicit CpuMemoryResource(std::shared_ptr<DeviceContext> device_context)
            : device_context_(device_context) {
        }

        /**
         * @brief Gets the device context associated with this memory resource.
         *
         * @return Shared pointer to the device context, may be null if default constructor was used
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Copies memory using optimized CPU memcpy.
         *
         * Performs efficient host-to-host memory copying using standard
         * library functions optimized for the target platform.
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param size_bytes Number of bytes to copy
         */
        void memcpy(void* dst, const void* src, std::size_t size_bytes) override {
            if (size_bytes > 0 && dst && src) {
                std::memcpy(dst, src, size_bytes);
            }
        }

        /**
         * @brief Sets memory to a specific byte value using CPU memset.
         *
         * Efficiently fills memory with the specified byte value using
         * platform-optimized memset implementation.
         *
         * @param ptr Pointer to the memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        void memset(void* ptr, int value, std::size_t size_bytes) override {
            if (size_bytes > 0 && ptr) {
                std::memset(ptr, value, size_bytes);
            }
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
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            if (bytes == 0) {
                return nullptr;
            }

            // Ensure minimum alignment for the platform
            if (alignment < alignof(std::max_align_t)) {
                alignment = alignof(std::max_align_t);
            }

#ifdef _WIN32
            void* ptr = _aligned_malloc(bytes, alignment);
#else
            // Ensure alignment is power of 2 and >= sizeof(void*)
            if (alignment < sizeof(void*)) {
                alignment = sizeof(void*);
            }

            // aligned_alloc requires size to be multiple of alignment
            std::size_t aligned_size = ((bytes + alignment - 1) / alignment) * alignment;
            void* ptr = std::aligned_alloc(alignment, aligned_size);
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
        void do_deallocate(void* ptr, std::size_t, std::size_t) override {
            if (ptr) {
#ifdef _WIN32
                _aligned_free(ptr);
#else
                std::free(ptr); // aligned_alloc memory can be freed with regular free
#endif
            }
        }

        /**
         * @brief Compares CPU memory resources for equality.
         *
         * CPU memory resources are considered equal if they are the same instance,
         * since they all manage the same underlying host memory pool.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CpuMemoryResource instances
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return dynamic_cast<const CpuMemoryResource*>(&other) != nullptr;
        }

    private:
        std::shared_ptr<DeviceContext> device_context_{ nullptr }; ///< Optional device context for interface consistency
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