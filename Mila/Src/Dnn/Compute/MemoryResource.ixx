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