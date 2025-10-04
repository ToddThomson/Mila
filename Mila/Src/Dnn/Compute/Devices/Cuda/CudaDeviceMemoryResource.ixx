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
import Compute.CudaDeviceContext;
import Compute.DeviceContext;
import Cuda.Utils;
import Cuda.BadAlloc;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    export struct CudaComputeDeviceTag {};

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
         * @brief Indicates CUDA device memory is not accessible from host code.
         */
        static constexpr bool is_host_accessible = false;

        /**
         * @brief Indicates CUDA device memory is accessible from device code.
         */
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

        /**
         * @brief Device type constant for CUDA memory resources.
         */
        static constexpr DeviceType device_type = DeviceType::Cuda;

		using ComputeDeviceTag = CudaComputeDeviceTag;
        using CompatibleDeviceContext = CudaDeviceContext;

        /**
         * @brief Constructs CUDA memory resource with device context.
         *
         * @param device_context Device context for proper device binding and stream coordination
         * @throws std::invalid_argument If device_context is null or not a CUDA device
         */
        explicit CudaDeviceMemoryResource(std::shared_ptr<DeviceContext> device_context)
            : device_context_(device_context) {
            if (!device_context_) {
                throw std::invalid_argument("Device context cannot be null");
            }
            if (!device_context_->isCudaDevice()) {
                throw std::invalid_argument("CudaDeviceMemoryResource requires CUDA device context");
            }
        }

        /**
         * @brief Gets the device context associated with this memory resource.
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Copies memory using CUDA memcpy with automatic transfer type detection.
         *
         * Uses cudaMemcpyDefault to automatically detect the appropriate transfer
         * direction (host-to-device, device-to-host, device-to-device) based on
         * pointer locations.
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param size_bytes Number of bytes to copy
         */
        void memcpy(void* dst, const void* src, std::size_t size_bytes) override {
            if (size_bytes == 0) {
                return;
            }

            device_context_->makeCurrent();
            cudaError_t status = cudaMemcpy(dst, src, size_bytes, cudaMemcpyDefault);
            cudaCheckStatus(status, std::source_location::current());
        }

        /**
         * @brief Sets CUDA device memory to a specific byte value.
         *
         * Uses cudaMemset to efficiently fill device memory with the specified
         * byte value. Ensures operations occur on the correct device.
         *
         * @param ptr Pointer to device memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        void memset(void* ptr, int value, std::size_t size_bytes) override {
            if (size_bytes == 0 || !ptr) {
                return;
            }

            device_context_->makeCurrent();
            cudaError_t status = cudaMemset(ptr, value, size_bytes);
            cudaCheckStatus(status, std::source_location::current());
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
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            if (bytes == 0) return nullptr;

            device_context_->makeCurrent();

            void* ptr = nullptr;
            cudaError_t result = cudaMalloc(&ptr, bytes);

            if (result != cudaSuccess) {
                std::string errorMsg = "CUDA device memory allocation failed: " +
                    std::string(cudaGetErrorString(result)) +
                    " (size: " + std::to_string(bytes) + " bytes)" +
                    " (device: " + std::to_string(device_context_->getDeviceId()) + ")";
                throw CudaBadAlloc(errorMsg);
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
            device_context_->makeCurrent();

            // Check for any previous CUDA errors before deallocation
            cudaCheckLastError(std::source_location::current());

            cudaError_t status = cudaFree(ptr);
            try {
                cudaCheckStatus(status, std::source_location::current());
            }
            catch (const CudaError& e) {
                std::ostringstream ss;
                ss << e.what() << " (ptr: 0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr) << ")"
                    << " (device: " << device_context_->getDeviceId() << ")";
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
        std::shared_ptr<DeviceContext> device_context_;
    };

    /**
     * @brief Alias for CudaDeviceMemoryResource that represents device-accessible memory.
     *
     * This alias provides a semantic name that describes the memory's accessibility
     * characteristics rather than its implementation details. Use DeviceMemoryResource
     * when you need memory that can be accessed by CUDA device code and operations.
     *
     * @see CudaDeviceMemoryResource
     */
    export using DeviceMemoryResource = CudaDeviceMemoryResource;
}