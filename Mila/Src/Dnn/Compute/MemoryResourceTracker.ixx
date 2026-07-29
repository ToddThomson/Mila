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
     * @brief Process-wide allocator counters for all TrackedMemoryResource instances.
     *
     * Named MemoryAllocationStats, not MemoryStats, to stay distinct from
     * Mila::Dnn::MemoryStats -- the per-component figure returned by
     * Component::getMemoryStats(). The two are one namespace apart, and every consumer
     * that opens both Mila::Dnn and Mila::Dnn::Compute (which is the conventional pair)
     * saw an ambiguous unqualified name.
     */
    export struct MemoryAllocationStats {
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
    std::atomic<size_t> MemoryAllocationStats::totalAllocated(0);
    std::atomic<size_t> MemoryAllocationStats::totalDeallocated(0);
    std::atomic<size_t> MemoryAllocationStats::currentUsage(0);
    std::atomic<size_t> MemoryAllocationStats::peakUsage(0);
    std::atomic<size_t> MemoryAllocationStats::allocationCount(0);
    std::atomic<size_t> MemoryAllocationStats::deallocationCount(0);
    std::atomic<size_t> MemoryAllocationStats::memcpyOperationCount(0);
    std::atomic<size_t> MemoryAllocationStats::memsetOperationCount(0);

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

        /*
         * Retired in place. Copy memory between potentially different memory spaces,
         * delegating to the underlying resource. Params: dst -- destination pointer;
         * src -- source pointer; size_bytes -- number of bytes to copy.
         */
        /*void memcpy(void* dst, const void* src, std::size_t size_bytes) override {
            underlying_->memcpy(dst, src, size_bytes);
            MemoryAllocationStats::memcpyOperationCount++;
        }*/

        /*
         * Retired in place. Set memory to a specific byte value, delegating to the
         * underlying resource. Params: ptr -- memory block to fill; value -- byte
         * value (0-255); size_bytes -- number of bytes to set.
         */
        /*void memset(void* ptr, int value, std::size_t size_bytes) override {
            underlying_->memset(ptr, value, size_bytes);
            MemoryAllocationStats::memsetOperationCount++;
        }*/

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
            MemoryAllocationStats::totalAllocated += bytes;
            MemoryAllocationStats::currentUsage += bytes;
            MemoryAllocationStats::allocationCount++;

            // Update peak usage atomically
            size_t currentUsage = MemoryAllocationStats::currentUsage;
            size_t peakUsage = MemoryAllocationStats::peakUsage;
            while (currentUsage > peakUsage) {
                if (MemoryAllocationStats::peakUsage.compare_exchange_weak(peakUsage, currentUsage)) {
                    break;
                }
                peakUsage = MemoryAllocationStats::peakUsage;
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
            MemoryAllocationStats::totalDeallocated += bytes;
            MemoryAllocationStats::currentUsage -= bytes;
            MemoryAllocationStats::deallocationCount++;
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