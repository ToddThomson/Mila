module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <exception>

export module Compute.CudaPinnedMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    /**
     * @brief A memory resource that allocates pinned (page-locked) memory using CUDA.
     */
    export class CudaPinnedMemoryResource : public MemoryResource {

    public:
        static constexpr bool is_host_accessible = HostAccessible::is_host_accessible;
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

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
            cudaError_t error = cudaMallocHost( &ptr, n );

            if ( error != cudaSuccess ) {
                std::string errorMsg = "CUDA pinned memory allocation failed: " +
                    std::string( cudaGetErrorString( error ) ) +
                    " (size: " + std::to_string( n ) + " bytes)";
                throw std::bad_alloc();
            }

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
            if ( ptr != nullptr ) {
                cudaFreeHost( ptr );
            }
        }

        /**
         * @brief Compares this memory resource with another for equality.
         * 
         * @param other The other memory resource to compare with.
         * @return true if the other memory resource is also a PinnedMemoryResource.
         * @return false otherwise.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return dynamic_cast<const CudaPinnedMemoryResource*>(&other) != nullptr;
        }
    };
}
