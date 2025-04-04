module;
#include <memory_resource>
#include <atomic>
#include <mutex>
#include <iostream>
#include <string_view>

export module Compute.MemoryResourceTracker;

import Compute.MemoryResource;

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

        /** @brief Resets all memory statistics to zero */
        static void reset() {
            totalAllocated = 0;
            totalDeallocated = 0;
            currentUsage = 0;
            peakUsage = 0;
            allocationCount = 0;
            deallocationCount = 0;
        }

        /** @brief Prints current memory statistics to the specified output stream */
        static void print( std::ostream& os = std::cout ) {
            os << "Memory Statistics:\n"
                << "  Total Allocated:    " << totalAllocated << " bytes\n"
                << "  Total Deallocated:  " << totalDeallocated << " bytes\n"
                << "  Current Usage:      " << currentUsage << " bytes\n"
                << "  Peak Usage:         " << peakUsage << " bytes\n"
                << "  Allocation Count:   " << allocationCount << "\n"
                << "  Deallocation Count: " << deallocationCount << "\n";
        }
    };

    // Initialize static members
    std::atomic<size_t> MemoryStats::totalAllocated( 0 );
    std::atomic<size_t> MemoryStats::totalDeallocated( 0 );
    std::atomic<size_t> MemoryStats::currentUsage( 0 );
    std::atomic<size_t> MemoryStats::peakUsage( 0 );
    std::atomic<size_t> MemoryStats::allocationCount( 0 );
    std::atomic<size_t> MemoryStats::deallocationCount( 0 );

    /**
     * @brief A memory resource wrapper that tracks allocation and deallocation statistics.
     *
     * This class wraps another memory resource and intercepts all allocation and
     * deallocation calls to maintain global memory usage statistics.
     */
    export class TrackedMemoryResource : public MemoryResource {
    public:
        /**
         * @brief Constructs a new tracked memory resource.
         *
         * @param underlying The memory resource to track.
         * @param name Optional name for this memory resource for logging purposes.
         */
        explicit TrackedMemoryResource( std::pmr::memory_resource* underlying,
            std::string_view name = "" )
            : underlying_( underlying ), name_( name ) {}

        /**
         * @brief Gets the name of this tracked memory resource.
         */
        std::string_view name() const { return name_; }

    protected:
        /**
         * @brief Allocates memory and updates tracking statistics.
         */
        void* do_allocate( std::size_t bytes, std::size_t alignment ) override {
            void* ptr = underlying_->allocate( bytes, alignment );

            // Update statistics
            MemoryStats::totalAllocated += bytes;
            MemoryStats::currentUsage += bytes;
            MemoryStats::allocationCount++;

            // Update peak usage
            size_t currentUsage = MemoryStats::currentUsage;
            size_t peakUsage = MemoryStats::peakUsage;
            while ( currentUsage > peakUsage ) {
                if ( MemoryStats::peakUsage.compare_exchange_weak( peakUsage, currentUsage ) ) {
                    break;
                }
                peakUsage = MemoryStats::peakUsage;
            }

            return ptr;
        }

        /**
         * @brief Deallocates memory and updates tracking statistics.
         */
        void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override {
            underlying_->deallocate( p, bytes, alignment );

            // Update statistics
            MemoryStats::totalDeallocated += bytes;
            MemoryStats::currentUsage -= bytes;
            MemoryStats::deallocationCount++;
        }

        /**
         * @brief Checks if this memory resource is equal to another.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            if ( auto* tracked = dynamic_cast<const TrackedMemoryResource*>(&other) ) {
                return underlying_->is_equal( *tracked->underlying_ );
            }
            return false;
        }

    private:
        std::pmr::memory_resource* underlying_;
        std::string name_;
    };
}
