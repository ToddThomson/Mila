module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <source_location>

export module Compute.CudaPinnedMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.DeviceContext;
import Compute.CudaDeviceContext;
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
        using CompatibleDeviceContext = CudaDeviceContext;

        static constexpr bool isValidDeviceContext(const DeviceContext& device_context) {
            return dynamic_cast<const CudaDeviceContext*>(&device_context) != nullptr;
        }

        /**
         * @brief Constructs CUDA pinned memory resource with device context.
         *
         * @param device_context Device context for proper device binding and stream coordination
         * @throws std::invalid_argument If device_context is null or not a CUDA device
         */
        explicit CudaPinnedMemoryResource(std::shared_ptr<DeviceContext> device_context)
            : device_context_(device_context) {
            if (!device_context_) {
                throw std::invalid_argument("Device context cannot be null");
            }
            if (!device_context_->isCudaDevice()) {
                throw std::invalid_argument("CudaPinnedMemoryResource requires CUDA device context");
            }
        }

        /**
         * @brief Gets the device context associated with this memory resource.
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Copies memory using CUDA memcpy optimized for pinned memory.
         *
         * Since pinned memory enables faster host-device transfers, this uses
         * cudaMemcpy with automatic transfer type detection for optimal performance.
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
         * @brief Sets pinned memory to a specific byte value.
         *
         * Since pinned memory is host-accessible, can use either host-side memset
         * or CUDA memset for efficiency. Uses CUDA memset for consistency with
         * other CUDA memory resources.
         *
         * @param ptr Pointer to pinned memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        void memset(void* ptr, int value, std::size_t size_bytes) override {
            if (size_bytes == 0 || !ptr) {
                return;
            }

            // For pinned memory, we could use either std::memset or cudaMemset
            // Using std::memset since it's host-accessible and avoids device context
            std::memset(ptr, value, size_bytes);
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
        void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override {
            if (bytes == 0) {
                return nullptr;
            }

            device_context_->makeCurrent();

            void* ptr = nullptr;
            cudaError_t error = cudaMallocHost(&ptr, bytes);

            if (error != cudaSuccess) {
                std::string errorMsg = "CUDA pinned memory allocation failed: " +
                    std::string(cudaGetErrorString(error)) +
                    " (size: " + std::to_string(bytes) + " bytes)" +
                    " (device: " + std::to_string(device_context_->getDeviceId()) + ")";
                throw std::bad_alloc();
            }

            return ptr;
        }

        /**
         * @brief Deallocates CUDA pinned memory.
         *
         * Uses cudaFreeHost to properly release page-locked memory.
         * Ensures operation occurs on the correct device context.
         *
         * @param ptr Pointer to pinned memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate(void* ptr, std::size_t, std::size_t alignment = alignof(std::max_align_t)) override {
            if (ptr) {
                device_context_->makeCurrent();
                cudaFreeHost(ptr);
            }
        }

        /**
         * @brief Compares pinned memory resources for equality.
         *
         * Pinned memory resources are equal if they are both CudaPinnedMemoryResource
         * instances. Device binding is handled through device context management.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CudaPinnedMemoryResource instances
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return dynamic_cast<const CudaPinnedMemoryResource*>(&other) != nullptr;
        }

    private:
        std::shared_ptr<DeviceContext> device_context_;
    };
}