/**
 * @file MetalMemoryResource.ixx
 * @brief Tensor memory on the Metal device.
 */

export module Compute.MetalMemoryResource;

import std;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.MetalDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Shared Metal buffers, one per allocation, made by the device.
     *
     * Unlike CUDA's resources, this one does not allocate itself. A Metal kernel binds a buffer and an offset, not a
     * pointer, so every allocation has to be in the device's map (`MetalDevice::locate`); making a buffer and recording it
     * are one call on the device, so that no allocation can be missing from it (MetalBackend.md section 5).
     *
     * The host can reach this memory, and `is_host_accessible` is still false (MetalBackend.md section 5). A host read
     * has to wait for the GPU first, and `Tensor`'s host accessors neither wait nor know FP16 and BF16 elements, so they
     * are taken away at compile time and a read goes through `copy`, which waits.
     */
    export class MetalMemoryResource : public MemoryResource
    {
    public:

        static constexpr DeviceType device_type = DeviceType::Metal;
        static constexpr bool is_host_accessible = false;
        static constexpr bool is_device_accessible = DeviceAccessible::is_device_accessible;

        /// @throws std::runtime_error When this Mac cannot run Mila on its GPU, or has no such device.
        explicit MetalMemoryResource( int device_index )
            : device_( MetalDevice::at( device_index ) )
        {
        }

    protected:

        void* do_allocate( std::size_t bytes, std::size_t /*alignment: a buffer's contents are page aligned*/ ) override
        {
            return bytes == 0 ? nullptr : device_->allocate( bytes );
        }

        void do_deallocate( void* contents, std::size_t, std::size_t ) noexcept override
        {
            if ( contents )
            {
                device_->deallocate( contents );
            }
        }

        [[nodiscard]] bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override
        {
            return dynamic_cast<const MetalMemoryResource*>( &other ) != nullptr;
        }

    private:

        // Shared, so that a tensor outliving the registry's teardown still frees into a live device.
        std::shared_ptr<MetalDevice> device_;
    };
}
