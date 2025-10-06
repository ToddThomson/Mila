module;
#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <source_location>
#include <string>

export module Compute.CudaManagedMemoryResource;

import Compute.DeviceType;
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
     * @brief CUDA managed memory resource for unified host/device accessible memory.
     *
     * Provides CUDA managed (unified) memory allocation that is accessible from both
     * host and device code with automatic migration. Focuses purely on memory allocation
     * responsibilities without tensor-specific operations or type conversions.
     */
    export class CudaManagedMemoryResource : public MemoryResource {
    public:
        static constexpr DeviceType device_type = DeviceType::Cuda;
		using ComputeDeviceTag = CudaComputeDeviceTag;
        using CompatibleDeviceContext = CudaDeviceContext;

        /*static constexpr bool isValidDeviceContext(const DeviceContext& device_context) {
            return dynamic_cast<const CudaDeviceContext*>(&device_context) != nullptr;
        }*/

        /**
         * @brief Constructs CUDA managed memory resource with device context.
         *
         * @param device_context Device context for proper device binding and stream coordination
         * @throws std::invalid_argument If device_context is null or not a CUDA device
         */
        explicit CudaManagedMemoryResource(std::shared_ptr<DeviceContext> device_context)
            : device_context_(device_context) {
            if (!device_context_) {
                throw std::invalid_argument("Device context cannot be null");
            }
            if (!device_context_->isCudaDevice()) {
                throw std::invalid_argument("CudaManagedMemoryResource requires CUDA device context");
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
         * For managed memory, uses cudaMemcpyDefault which automatically handles
         * the unified memory addressing and migration as needed.
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param size_bytes Number of bytes to copy
         */
        /*void memcpy(void* dst, const void* src, std::size_t size_bytes) override {
            if (size_bytes == 0) {
                return;
            }

            device_context_->makeCurrent();
            cudaError_t status = cudaMemcpy(dst, src, size_bytes, cudaMemcpyDefault);
            cudaCheckStatus(status, std::source_location::current());
        }*/

        /**
         * @brief Sets managed memory to a specific byte value.
         *
         * Uses cudaMemset to efficiently fill managed memory. Since managed memory
         * is accessible from both host and device, this operation is efficient
         * regardless of current memory location.
         *
         * @param ptr Pointer to managed memory block to fill
         * @param value Byte value to set (0-255)
         * @param size_bytes Number of bytes to set
         */
        /*void memset(void* ptr, int value, std::size_t size_bytes) override {
            if (size_bytes == 0 || !ptr) {
                return;
            }

            device_context_->makeCurrent();
            cudaError_t status = cudaMemset(ptr, value, size_bytes);
            cudaCheckStatus(status, std::source_location::current());
        }*/

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

            device_context_->makeCurrent();

            void* ptr = nullptr;
            cudaError_t result = cudaMallocManaged(&ptr, bytes);

            if (result != cudaSuccess) {
                std::string errorMsg = "CUDA managed memory allocation failed: " +
                    std::string(cudaGetErrorString(result)) +
                    " (size: " + std::to_string(bytes) + " bytes)" +
                    " (device: " + std::to_string(device_context_->getDeviceId()) + ")";
                throw std::bad_alloc();
            }

            return ptr;
        }

        /**
         * @brief Deallocates CUDA managed memory.
         *
         * Uses cudaFree to release managed memory. Ensures operation occurs
         * on the correct device context.
         *
         * @param ptr Pointer to managed memory to deallocate
         * @param bytes Size of memory block (unused, kept for interface compatibility)
         * @param alignment Alignment used during allocation (unused, kept for interface compatibility)
         */
        void do_deallocate(void* ptr, std::size_t, std::size_t alignment = alignof(std::max_align_t)) override {
            if (ptr) {
                device_context_->makeCurrent();
                cudaFree(ptr);
            }
        }

        /**
         * @brief Compares managed memory resources for equality.
         *
         * Managed memory resources are equal if they are both CudaManagedMemoryResource
         * instances. Device binding is handled through device context management.
         *
         * @param other The other memory resource to compare with
         * @return true if both are CudaManagedMemoryResource instances
         */
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return dynamic_cast<const CudaManagedMemoryResource*>(&other) != nullptr;
        }

    private:
        std::shared_ptr<DeviceContext> device_context_;
    };
}