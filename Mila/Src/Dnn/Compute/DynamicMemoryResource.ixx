module;
#include <variant>
#include <memory_resource>
#include <type_traits>

export module Compute.DynamicMemoryResource;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
    * @brief A class that represents a dynamically-determined memory resource.
    *
    * This class serves as an adapter between the runtime selection of memory resources
    * (via DeviceContext) and the compile-time requirements of the Tensor class, which
    * requires a specific memory resource type rather than a variant.
    */
    export class DynamicMemoryResource : public MemoryResource {
    public:
        /**
         * @brief Constructs a DynamicMemoryResource based on the device type.
         *
         * @param device_type The type of device to create the memory resource for.
         */
        explicit DynamicMemoryResource( Compute::DeviceType device_type = Compute::DeviceType::Cuda ) {
            if ( device_type == Compute::DeviceType::Cuda ) {
                resource_variant_ = Compute::CudaMemoryResource{};
            }
            else {
                resource_variant_ = Compute::CpuMemoryResource{};
            }
        }

    protected:
        /**
         * @brief Allocates memory of the specified size with the given alignment.
         *
         * This delegates to the appropriate memory resource based on the device type.
         *
         * @param size The size in bytes to allocate.
         * @param alignment The alignment requirement for the allocation.
         * @return void* Pointer to the allocated memory.
         */
        void* do_allocate( std::size_t size, std::size_t alignment ) override {
            return std::visit( [size, alignment]( auto& resource ) -> void* {
                return resource.allocate( size, alignment );
                }, resource_variant_ );
        }

        /**
         * @brief Deallocates previously allocated memory.
         *
         * This delegates to the appropriate memory resource based on the device type.
         *
         * @param ptr Pointer to the memory to deallocate.
         * @param size The size in bytes of the allocation.
         * @param alignment The alignment of the allocation.
         */
        void do_deallocate( void* ptr, std::size_t size, std::size_t alignment ) override {
            std::visit( [ptr, size, alignment]( auto& resource ) {
                resource.deallocate( ptr, size, alignment );
                }, resource_variant_ );
        }

        /**
         * @brief Checks if this memory resource is equal to another memory resource.
         *
         * @param other The other memory resource to compare with.
         * @return bool True if the memory resources are equal, false otherwise.
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            return this == &other;
        }

    public:
        /**
         * @brief Checks if the memory resource is host-accessible.
         *
         * @return bool True if the memory is accessible from the host (CPU).
         */
        bool is_host_accessible() const noexcept {
            return std::visit( []( const auto& resource ) -> bool {
                using ResourceType = std::decay_t<decltype(resource)>;
                return ResourceType::is_host_accessible;
                }, resource_variant_ );
        }

        /**
         * @brief Checks if the memory resource is device-accessible.
         *
         * @return bool True if the memory is accessible from the device (GPU).
         */
        bool is_device_accessible() const noexcept {
            return std::visit( []( const auto& resource ) -> bool {
                using ResourceType = std::decay_t<decltype(resource)>;
                return ResourceType::is_device_accessible;
                }, resource_variant_ );
        }

    private:
        std::variant<Compute::CpuMemoryResource, Compute::CudaMemoryResource> resource_variant_;
    };
}