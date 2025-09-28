module;
#include <memory_resource>
#include <atomic>
#include <mutex>
#include <iostream>
#include <string_view>
#include <string>

export module Compute.MemoryResourceTracker;

import Compute.MemoryResource;
import Dnn.TensorDataType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Global memory statistics for all TrackedMemoryResource instances.
     */
    export struct MemoryStats {
        static std::atomic<size_t> totalAllocated;
        static std::atomic<size_t> totalDeallocated;
        static std::atomic<size_t> currentUsage;
        static std::atomic<size_t> peakUsage;
        static std::atomic<size_t> allocationCount;
        static std::atomic<size_t> deallocationCount;
        static std::atomic<size_t> memcpyOperationCount;
        static std::atomic<size_t> memsetOperationCount;

        /** @brief Resets all memory statistics to zero */
        static void reset() {
            totalAllocated = 0;
            totalDeallocated = 0;
            currentUsage = 0;
            peakUsage = 0;
            allocationCount = 0;
            deallocationCount = 0;
            memcpyOperationCount = 0;
            memsetOperationCount = 0;
        }

        /** @brief Prints current memory statistics to the specified output stream */
        static void print(std::ostream& os = std::cout) {
            os << "Memory Statistics:\n"
                << "  Total Allocated:    " << totalAllocated << " bytes\n"
                << "  Total Deallocated:  " << totalDeallocated << " bytes\n"
                << "  Current Usage:      " << currentUsage << " bytes\n"
                << "  Peak Usage:         " << peakUsage << " bytes\n"
                << "  Allocation Count:   " << allocationCount << "\n"
                << "  Deallocation Count: " << deallocationCount << "\n"
                << "  Memcpy Operations:  " << memcpyOperationCount << "\n"
                << "  Memset Operations:  " << memsetOperationCount << "\n";
        }
    };

    // Initialize static members
    std::atomic<size_t> MemoryStats::totalAllocated(0);
    std::atomic<size_t> MemoryStats::totalDeallocated(0);
    std::atomic<size_t> MemoryStats::currentUsage(0);
    std::atomic<size_t> MemoryStats::peakUsage(0);
    std::atomic<size_t> MemoryStats::allocationCount(0);
    std::atomic<size_t> MemoryStats::deallocationCount(0);
    std::atomic<size_t> MemoryStats::memcpyOperationCount(0);
    std::atomic<size_t> MemoryStats::memsetOperationCount(0);

    /**
     * @brief A memory resource wrapper that tracks allocation and deallocation statistics.
     *
     * This class wraps another memory resource and intercepts all allocation,
     * deallocation, memcpy, and memset calls to maintain global memory usage statistics.
     */
    export class TrackedMemoryResource : public MemoryResource {
    public:
        /**
         * @brief Constructs a new tracked memory resource.
         *
         * @param underlying The memory resource to track (takes ownership).
         * @param name Optional name for this memory resource for logging purposes.
         */
        explicit TrackedMemoryResource(MemoryResource* underlying,
            std::string_view name = "")
            : underlying_(underlying), name_(name) {
        }

        /**
         * @brief Destructor that properly cleans up the underlying resource.
         */
        ~TrackedMemoryResource() {
            delete underlying_;
        }

        /**
         * @brief Copies memory between potentially different memory spaces, delegating to underlying resource.
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param size_bytes Number of bytes to copy
         */
        void memcpy(void* dst, const void* src, std::size_t size_bytes) override {
            underlying_->memcpy(dst, src, size_bytes);
            MemoryStats::memcpyOperationCount++;
        }

        /**
         * @brief Sets memory to a specific byte value, delegating to underlying resource.
         *
         * @param ptr Pointer to the memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        void memset(void* ptr, int value, std::size_t size_bytes) override {
            underlying_->memset(ptr, value, size_bytes);
            MemoryStats::memsetOperationCount++;
        }

        /**
         * @brief Gets the name of this tracked memory resource.
         */
        std::string_view name() const { return name_; }

        /**
         * @brief Gets access to the underlying memory resource.
         */
        const MemoryResource* getUnderlying() const { return underlying_; }

    protected:
        /**
         * @brief Allocates memory and updates tracking statistics.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement
         * @return Pointer to allocated memory
         */
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            void* ptr = underlying_->allocate(bytes, alignment);

            // Update statistics
            MemoryStats::totalAllocated += bytes;
            MemoryStats::currentUsage += bytes;
            MemoryStats::allocationCount++;

            // Update peak usage atomically
            size_t currentUsage = MemoryStats::currentUsage;
            size_t peakUsage = MemoryStats::peakUsage;
            while (currentUsage > peakUsage) {
                if (MemoryStats::peakUsage.compare_exchange_weak(peakUsage, currentUsage)) {
                    break;
                }
                peakUsage = MemoryStats::peakUsage;
            }

            return ptr;
        }

        /**
         * @brief Deallocates memory and updates tracking statistics.
         *
         * @param p Pointer to memory to deallocate
         * @param bytes Size of memory block
         * @param alignment Alignment used during allocation
         */
        void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
            underlying_->deallocate(p, bytes, alignment);

            // Update statistics
            MemoryStats::totalDeallocated += bytes;
            MemoryStats::currentUsage -= bytes;
            MemoryStats::deallocationCount++;
        }

        /**
         * @brief Checks if this memory resource is equal to another.
         *
         * @param other The other memory resource to compare with
         * @return true if the underlying resources are equal
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            if (auto* tracked = dynamic_cast<const TrackedMemoryResource*>(&other)) {
                return underlying_->is_equal(*tracked->underlying_);
            }
            return false;
        }

    private:
        MemoryResource* underlying_;  ///< The wrapped memory resource (owned)
        std::string name_;           ///< Optional name for debugging/logging
    };
}