module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <string>
#include <source_location>
#include <iostream>
#include <sstream>
#include <cassert>

export module Compute.CudaDeviceMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.DeviceType;
import Compute.CudaExecutionContext;
import Compute.ExecutionContext;
import Cuda.Helpers;
import Cuda.BadAlloc;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA device memory resource for GPU-accessible memory allocation.
     *
     * Provides optimized CUDA device memory allocation with proper device binding
     * through device context integration. Focuses purely on memory allocation
     * responsibilities without tensor-specific operations or type conversions.
     */
    export class CudaDeviceMemoryResource : public MemoryResource {
    public:
        /**
         * @brief Device type constant for CUDA memory resources.
         */
        static constexpr DeviceType device_type = DeviceType::Cuda;

        /**
         * @brief Indicates CUDA device memory is not accessible from host code.
         */
        static constexpr bool is_host_accessible = false;

        /**
         * @brief Indicates CUDA device memory is accessible from device code.
         */
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

        /**
         * @brief Constructs CUDA managed memory resource with device ID
         *
         * @param device_id CUDA device ID (0, 1, 2, etc.)
         * @throws std::invalid_argument If device_id is invalid
         */
        explicit CudaDeviceMemoryResource( int device_id )
            : device_id_( device_id ) {

            if (device_id_ < 0) {
                throw std::invalid_argument( "Device ID must be non-negative" );
            }

            // Validate device exists
            int device_count = 0;
            cudaGetDeviceCount( &device_count );
            if (device_id_ >= device_count) {
                throw std::invalid_argument(
                    "Device ID " + std::to_string( device_id_ ) +
                    " exceeds available devices (" + std::to_string( device_count ) + ")"
                );
            }
        }
        
    protected:
        /**
         * @brief Allocates memory on the CUDA device.
         *
         * Ensures allocation occurs on the correct device by activating the
         * device context before calling cudaMalloc. Provides detailed error
         * information on allocation failure.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement (ignored by CUDA)
         * @return Pointer to allocated device memory
         * @throws CudaBadAlloc If allocation fails
         */
        void* do_allocate( std::size_t bytes, std::size_t alignment ) override {
            if (bytes == 0) return nullptr;

            Cuda::setCurrentDevice( device_id_ );

            void* ptr = nullptr;
            cudaError_t result = cudaMalloc( &ptr, bytes );

            if (result != cudaSuccess) {
                std::string errorMsg = "CUDA device memory allocation failed: " +
                    std::string( cudaGetErrorString( result ) ) +
                    " (size: " + std::to_string( bytes ) + " bytes)" +
                    " (device: " + std::to_string( device_id_ ) + ")";
                throw CudaBadAlloc( errorMsg );
            }

            return ptr;
        }

        /**
         * @brief Deallocates CUDA device memory.
         *
         * Ensures deallocation occurs on the correct device and provides
         * detailed error information if deallocation fails.
         *
         * @param ptr Pointer to device memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate(void* ptr, std::size_t, std::size_t) override {
            if (!ptr) return;

            assert(ptr != nullptr);
            
            Cuda::setCurrentDevice( device_id_ );

            // Check for any previous CUDA errors before deallocation
            cudaCheckLastError(std::source_location::current());

            cudaError_t status = cudaFree(ptr);
            try {
                cudaCheckStatus(status, std::source_location::current());
            }
            catch (const CudaError& e) {
                std::ostringstream ss;
                ss << e.what() << " (ptr: 0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr) << ")"
                    << " (device: " << device_id_ << ")";
                std::cerr << ss.str() << std::endl;
                throw;
            }
        }

        /**
         * @brief Compares CUDA memory resources for equality.
         *
         * CUDA memory resources are equal if they are both CudaDeviceMemoryResource
         * instances. Device binding is handled at the tensor level through
         * device context management.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CudaDeviceMemoryResource instances
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return dynamic_cast<const CudaDeviceMemoryResource*>(&other) != nullptr;
        }

    private:
        int device_id_;
    };
}