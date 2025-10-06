module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <source_location>

export module Compute.CudaPinnedMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.DeviceType;
import Cuda.Utils;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    export struct CudaComputeDeviceTag {};

    /**
     * @brief CUDA pinned memory resource for fast host/device transfer memory.
     *
     * Provides CUDA pinned (page-locked) memory allocation that is accessible from
     * host code but provides faster transfers to/from GPU devices. Focuses purely
     * on memory allocation responsibilities without tensor-specific operations.
     */
    export class CudaPinnedMemoryResource : public MemoryResource {
    public:
        using ComputeDeviceTag = CudaComputeDeviceTag;
        static constexpr DeviceType device_type = DeviceType::Cuda;

        /**
         * @brief Constructs CUDA pinned memory resource with device ID.
         *
         * @param device_id CUDA device ID (0, 1, 2, etc.)
         * @throws std::invalid_argument If device_id is invalid
         */
        explicit CudaPinnedMemoryResource( int device_id )
            : device_id_( device_id ) {

            if (device_id_ < 0) {
                throw std::invalid_argument( "Device ID must be non-negative" );
            }

            // Validate device exists
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            if (error != cudaSuccess || device_id_ >= device_count) {
                throw std::invalid_argument(
                    "Invalid device ID " + std::to_string( device_id_ ) +
                    ": " + (error != cudaSuccess ? cudaGetErrorString( error ) :
                        "exceeds device count " + std::to_string( device_count ))
                );
            }
        }

        /**
         * @brief Indicates pinned memory is accessible from host code.
         */
        static constexpr bool is_host_accessible = HostAccessible::is_host_accessible;

        /**
         * @brief Indicates pinned memory is accessible from device code.
         */
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

    protected:
        /**
         * @brief Allocates CUDA pinned memory.
         *
         * Uses cudaMallocHost to allocate page-locked memory that provides
         * faster transfers between host and device while remaining host-accessible.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement (ignored by CUDA)
         * @return Pointer to allocated pinned memory
         * @throws std::bad_alloc If allocation fails
         */
        void* do_allocate( std::size_t bytes, std::size_t alignment = alignof(std::max_align_t) ) override {
            if (bytes == 0) {
                return nullptr;
            }

            // Set device before allocation
            cudaError_t set_error = cudaSetDevice( device_id_ );
            if (set_error != cudaSuccess) {
                throw std::runtime_error(
                    "Failed to set device " + std::to_string( device_id_ ) +
                    ": " + cudaGetErrorString( set_error )
                );
            }

            void* ptr = nullptr;
            cudaError_t error = cudaMallocHost( &ptr, bytes );

            if (error != cudaSuccess) {
                throw std::bad_alloc();
            }

            return ptr;
        }

        /**
         * @brief Deallocates CUDA pinned memory.
         *
         * Uses cudaFreeHost to properly release page-locked memory.
         * Ensures operation occurs on the correct device.
         *
         * @param ptr Pointer to pinned memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate( void* ptr, std::size_t, std::size_t alignment = alignof(std::max_align_t) ) override {
            if (ptr) {
                cudaSetDevice( device_id_ ); // Best effort - ignore errors in destructor path
                cudaFreeHost( ptr );
            }
        }

        /**
         * @brief Compares pinned memory resources for equality.
         *
         * Pinned memory resources are equal if they are both CudaPinnedMemoryResource
         * instances with the same device ID.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CudaPinnedMemoryResource with same device ID
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            if (auto* cuda_mr = dynamic_cast<const CudaPinnedMemoryResource*>(&other)) {
                return device_id_ == cuda_mr->device_id_;
            }
            return false;
        }

    private:
        int device_id_;
    };
}