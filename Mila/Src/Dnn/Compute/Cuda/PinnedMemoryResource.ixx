module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>

export module Compute.PinnedMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    /**
     * @brief A memory resource that allocates pinned (page-locked) memory using CUDA.
     */
    export class PinnedMemoryResource : public MemoryResource {

    public:
        static constexpr bool is_cpu_accessible = CpuAccessible::is_cpu_accessible;
        static constexpr bool is_cuda_accessible = CudaAccessible::is_cuda_accessible;

    protected:
        /**
         * @brief Allocates pinned memory.
         * 
         * @param n The size of the memory to allocate.
         * @param alignment The alignment of the memory to allocate.
         * @return void* Pointer to the allocated memory.
         * @throws std::bad_alloc if the allocation fails.
         */
        void* do_allocate( std::size_t n, std::size_t alignment = alignof(std::max_align_t) ) override {
            if ( n == 0 ) {
                return nullptr;
            }

            void* ptr = nullptr;

            if ( cudaMallocHost( &ptr, n ) != cudaSuccess )
                throw std::bad_alloc();

            return ptr;
        }

        /**
         * @brief Deallocates pinned memory.
         * 
         * @param ptr Pointer to the memory to deallocate.
         * @param size The size of the memory to deallocate.
         * @param alignment The alignment of the memory to deallocate.
         */
        void do_deallocate( void* ptr, std::size_t, std::size_t alignment = alignof(std::max_align_t) ) override {
            cudaFreeHost( ptr );
        }

        /**
         * @brief Compares this memory resource with another for equality.
         * 
         * @param other The other memory resource to compare with.
         * @return true if the other memory resource is also a PinnedMemoryResource.
         * @return false otherwise.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return dynamic_cast<const PinnedMemoryResource*>(&other) != nullptr;
        }
    };
}
