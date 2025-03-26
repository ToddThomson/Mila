module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>

export module Compute.CudaMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    /**
     * @brief A memory resource that allocates memory on a CUDA device.
     */
    export class DeviceMemoryResource : public MemoryResource {

    public:
        static constexpr bool is_host_accessible = false;
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

    protected:
        /**
         * @brief Allocates memory on the CUDA device.
         * 
         * @param n The size of the memory to allocate.
         * @param alignment The alignment of the memory to allocate.
         * @return void* A pointer to the allocated memory.
         * @throws std::bad_alloc if the allocation fails.
         */
        [[nodiscard]] void* do_allocate( std::size_t n, std::size_t alignment ) override {
            if ( n == 0 ) return nullptr;
            void* ptr = nullptr;
            if ( cudaMalloc( &ptr, n ) != cudaSuccess )
                throw std::bad_alloc();
            return ptr;
        }

        /**
         * @brief Deallocates memory on the CUDA device.
         * 
         * @param ptr A pointer to the memory to deallocate.
         * @param n The size of the memory to deallocate.
         * @param alignment The alignment of the memory to deallocate.
         */
        void do_deallocate( void* ptr, std::size_t, std::size_t ) override {
            cudaFree( ptr );
        }

        /**
         * @brief Checks if this memory resource is equal to another memory resource.
         * 
         * @param other The other memory resource to compare to.
         * @return true if the other memory resource is a DeviceMemoryResource.
         * @return false otherwise.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return dynamic_cast<const DeviceMemoryResource*>(&other) != nullptr;
        }
    };
}