/**
 * @file MemoryResource.ixx
 * @brief Defines a clean memory resource abstraction focused on allocation responsibilities.
 */

module;
#include <memory_resource>
#include <cstddef>

export module Compute.MemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Clean memory resource abstraction for device-specific memory allocation.
     *
     * This class extends std::pmr::memory_resource with device-awareness while maintaining
     * clear separation of concerns. Memory resources are responsible only for allocation,
     * deallocation, and basic memory operations. Tensor-specific operations like type
     * conversion and fill operations are handled by separate tensor operation classes.
     *
     * Device-specific implementations (CPU, CUDA) override these methods with optimized
     * implementations appropriate for their memory types and access patterns.
     */
    export class MemoryResource : public std::pmr::memory_resource {
    public:
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         */
        virtual ~MemoryResource() = default;

        /**
         * @brief Copies memory between potentially different memory spaces.
         *
         * This method handles copying between host and device memory automatically,
         * using the most efficient transfer method available for the memory resource type.
         * This is a fundamental memory operation that belongs in the memory resource layer.
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param size_bytes Number of bytes to copy
         */
        virtual void memcpy(void* dst, const void* src, std::size_t size_bytes) = 0;

        /**
         * @brief Sets memory to a specific byte value.
         *
         * Fills a block of memory with a specified byte value. This is a basic memory
         * operation that's distinct from tensor-level fill operations with type conversion.
         *
         * @param ptr Pointer to the memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        virtual void memset(void* ptr, int value, std::size_t size_bytes) = 0;

        /**
         * @brief Checks if the memory is accessible from host code.
         *
         * Derived classes should override this to reflect their accessibility characteristics.
         */
        static constexpr bool is_host_accessible = true;

        /**
         * @brief Checks if the memory is accessible from device code.
         *
         * Derived classes should override this to reflect their accessibility characteristics.
         */
        static constexpr bool is_device_accessible = false;

    protected:
        /**
         * @brief Allocates memory with specified size and alignment.
         *
         * Pure virtual function that must be implemented by derived classes to provide
         * device-specific memory allocation. Implementation should handle device-specific
         * allocation strategies and error conditions.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement
         * @return Pointer to allocated memory
         * @throws std::bad_alloc If allocation fails
         */
        virtual void* do_allocate(std::size_t bytes, std::size_t alignment) override = 0;

        /**
         * @brief Deallocates previously allocated memory.
         *
         * Pure virtual function that must be implemented by derived classes to provide
         * device-specific memory deallocation. Implementation should handle proper cleanup
         * and device synchronization if necessary.
         *
         * @param ptr Pointer to memory to deallocate
         * @param bytes Size of memory block (may be used for debugging/validation)
         * @param alignment Alignment that was used during allocation
         */
        virtual void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override = 0;

        /**
         * @brief Compares this memory resource with another for equality.
         *
         * Two memory resources are equal if they can be used interchangeably for
         * allocation and deallocation. Typically this means they are the same type
         * and manage the same underlying memory pool or device.
         *
         * @param other The other memory resource to compare with
         * @return true if the memory resources are equivalent
         */
        virtual bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override = 0;
    };
}