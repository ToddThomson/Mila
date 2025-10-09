module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>

export module Compute.CudaManagedMemoryResource;

import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA managed memory resource for unified host/device accessible memory.
     *
     * Provides CUDA managed (unified) memory allocation that is accessible from both
     * host and device code with automatic migration. Focuses purely on memory allocation
     * responsibilities without tensor-specific operations or type conversions.
     */
    export class CudaManagedMemoryResource : public MemoryResource {
    public:
        static constexpr DeviceType device_type = DeviceType::Cuda;

        /**
         * @brief Constructs CUDA managed memory resource with device ID.
         *
         * @param device_id CUDA device ID (0, 1, 2, etc.)
         * @throws std::invalid_argument If device_id is invalid
         */
        explicit CudaManagedMemoryResource(int device_id)
            : device_id_(device_id) {
            
            if (device_id_ < 0) {
                throw std::invalid_argument("Device ID must be non-negative");
            }

            // Validate device exists
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount(&device_count);
            if (error != cudaSuccess || device_id_ >= device_count) {
                throw std::invalid_argument(
                    "Invalid device ID " + std::to_string(device_id_) +
                    ": " + (error != cudaSuccess ? cudaGetErrorString(error) :
                        "exceeds device count " + std::to_string(device_count))
                );
            }
        }

        /**
         * @brief Indicates managed memory is accessible from host code.
         */
        static constexpr bool is_host_accessible = HostAccessible::is_host_accessible;

        /**
         * @brief Indicates managed memory is accessible from device code.
         */
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

    protected:
        /**
         * @brief Allocates CUDA managed memory.
         *
         * Uses cudaMallocManaged to allocate unified memory that can be accessed
         * from both host and device code with automatic migration.
         *
         * @param bytes Number of bytes to allocate
         * @param alignment Memory alignment requirement (ignored by CUDA)
         * @return Pointer to allocated managed memory
         * @throws std::bad_alloc If allocation fails
         */
        void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override {
            if (bytes == 0) return nullptr;

            Cuda::setCurrentDevice(device_id_);

            void* ptr = nullptr;
            cudaError_t result = cudaMallocManaged(&ptr, bytes);

            if (result != cudaSuccess) {
                std::string errorMsg = "CUDA managed memory allocation failed: " +
                    std::string(cudaGetErrorString(result)) +
                    " (size: " + std::to_string(bytes) + " bytes)" +
                    " (device: " + std::to_string(device_id_) + ")";
                throw std::bad_alloc();
            }

            return ptr;
        }

        /**
         * @brief Deallocates CUDA managed memory.
         *
         * Uses cudaFree to release managed memory. Ensures operation occurs
         * on the correct device.
         *
         * @param ptr Pointer to managed memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate(void* ptr, std::size_t, std::size_t alignment = alignof(std::max_align_t)) override {
            if (ptr) {
                Cuda::setCurrentDevice(device_id_);
                cudaFree(ptr);
            }
        }

        /**
         * @brief Compares managed memory resources for equality.
         *
         * Managed memory resources are equal if they are both CudaManagedMemoryResource
         * instances with the same device ID.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CudaManagedMemoryResource with same device ID
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            if (auto* cuda_mr = dynamic_cast<const CudaManagedMemoryResource*>(&other)) {
                return device_id_ == cuda_mr->device_id_;
            }
            return false;
        }

    private:
        int device_id_{-1};
    };
}