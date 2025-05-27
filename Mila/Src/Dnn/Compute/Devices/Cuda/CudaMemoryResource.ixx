module;
#include <cuda_runtime.h>
//#include <format>
#include <memory_resource>
#include <string>
#include <source_location>
#include <iostream>
#include <sstream>
#include <cassert>

export module Compute.CudaMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Cuda.BadAlloc;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief A memory resource that allocates memory on a CUDA device.
     */
    export class CudaMemoryResource : public MemoryResource {

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

            if ( cudaMalloc( &ptr, n ) != cudaSuccess ) {
                std::string errorMsg = "CUDA device memory allocation failed: " +
                    std::string( cudaGetErrorString( cudaGetLastError() ) ) +
                    " (size: " + std::to_string( n ) + " bytes)";
                throw CudaBadAlloc( errorMsg );
            }
            
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
            assert( ptr != nullptr );

            // DEBUG: Check for any previous CUDA errors
            cudaCheckLastError( std::source_location::current() );

            cudaError_t status = cudaFree( ptr );
            try {
                cudaCheckStatus( status, std::source_location::current() );
            }
            catch ( const CudaError& e ) {
                std::ostringstream ss;
                ss << e.what() << " (ptr: 0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr) << ")";
                std::cerr << ss.str() << std::endl;
                throw;
            }
        }

        /**
         * @brief Checks if this memory resource is equal to another memory resource.
         * 
         * @param other The other memory resource to compare to.
         * @return true if the other memory resource is a CudaMemoryResource.
         * @return false otherwise.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return dynamic_cast<const CudaMemoryResource*>(&other) != nullptr;
        }
    };

    /**
    * @brief Alias for CudaMemoryResource that represents device-accessible memory.
    *
    * This alias provides a semantic name that describes the memory's accessibility
    * characteristics rather than its implementation details. Use DeviceMemoryResource
    * when you need memory that can be accessed by CUDA device code and operations.
    *
    * This naming follows CUDA conventions where "device" refers to GPU memory,
    * while maintaining consistency with the architecture's naming pattern.
    *
    * @see CudaMemoryResource
    */
	export using DeviceMemoryResource = CudaMemoryResource;
}