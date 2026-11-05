/**
 * @file MetalDevice.ixx
 * @brief The GPU of an Apple M-series chip: its checks, its allocations and its kernels.
 */

module;

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

export module Compute.MetalDevice;

import std;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Compute.MetalAllocationMap;
import Compute.MetalBadAlloc;
import Compute.MetalBufferLocation;
import Compute.MetalCapabilities;
import Compute.MetalKernelLibrary;

namespace Mila::Dnn::Compute
{
    namespace
    {
        int highestAppleFamily( MTL::Device* device )
        {
            for ( int family = kNewestKnownAppleGpuFamily; family >= kMinimumAppleGpuFamily; --family )
            {
                const auto named = static_cast<MTL::GPUFamily>( static_cast<int>( MTL::GPUFamilyApple1 ) + family - 1 );

                if ( device->supportsFamily( named ) )
                {
                    return family;
                }
            }

            return 0;
        }

        MetalCapabilities readCapabilities( MTL::Device* device )
        {
            const NS::OperatingSystemVersion version = NS::ProcessInfo::processInfo()->operatingSystemVersion();

            MetalCapabilities capabilities;
            capabilities.device_name = device->name()->utf8String();
            capabilities.architecture_name = device->architecture()->name()->utf8String();
            capabilities.macos_major_version = static_cast<int>( version.majorVersion );
            capabilities.macos_minor_version = static_cast<int>( version.minorVersion );
            capabilities.macos_patch_version = static_cast<int>( version.patchVersion );
            capabilities.apple_gpu_family = highestAppleFamily( device );
            capabilities.has_unified_memory = device->hasUnifiedMemory();
            capabilities.recommended_max_working_set_bytes = device->recommendedMaxWorkingSetSize();
            capabilities.max_buffer_bytes = device->maxBufferLength();
            capabilities.max_threadgroup_memory_bytes = device->maxThreadgroupMemoryLength();

            return capabilities;
        }
    }

    /**
     * @brief The one Metal device of a Mac.
     *
     * Created by DeviceRegistry and shared. Constructing it is what refuses a Mac Mila cannot run on, so that a build
     * with Metal in it still starts there and everything that needs no GPU keeps working.
     */
    export class MetalDevice : public Device
    {
    public:

        /// @throws std::runtime_error In words for the person running it, when this Mac cannot run Mila on its GPU.
        explicit MetalDevice( [[maybe_unused]] DeviceConstructionKey key )
        {
            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
            device_ = MTL::CreateSystemDefaultDevice();

            if ( !device_ )
            {
                pool->release();

                throw std::runtime_error( "This Mac has no GPU that Mila can use." );
            }

            capabilities_ = readCapabilities( device_ );
            pool->release();

            if ( const std::optional<std::string> refusal = metalRefusalReason( capabilities_ ) )
            {
                device_->release();
                device_ = nullptr;

                throw std::runtime_error( *refusal );
            }

            kernels_ = std::make_unique<MetalKernelLibrary>( device_ );
        }

        ~MetalDevice() override
        {
            kernels_.reset();

            if ( device_ )
            {
                device_->release();
            }
        }

        MetalDevice( const MetalDevice& ) = delete;
        MetalDevice& operator=( const MetalDevice& ) = delete;

        /// Whether this Mac has a Metal device at all, which registration asks.
        [[nodiscard]] static bool isPresent()
        {
            MTL::Device* const device = MTL::CreateSystemDefaultDevice();

            if ( !device )
            {
                return false;
            }

            device->release();

            return true;
        }

        /// The shared device. @throws std::runtime_error As the constructor, or when no such device is registered.
        [[nodiscard]] static std::shared_ptr<MetalDevice> at( int device_index )
        {
            return std::static_pointer_cast<MetalDevice>(
                DeviceRegistry::instance().getDevice( DeviceId{ DeviceType::Metal, device_index } ) );
        }

        [[nodiscard]] DeviceId getDeviceId() const override
        {
            return Device::Metal( 0 );
        }

        [[nodiscard]] constexpr DeviceType getDeviceType() const override
        {
            return DeviceType::Metal;
        }

        /// "Metal:0", as a device is named everywhere in Mila. The chip's own name is in capabilities().
        [[nodiscard]] std::string getDeviceName() const override
        {
            return getDeviceId().toString();
        }

        /**
         * @brief What is left of the working set Apple recommends, and that recommendation.
         *
         * The total is `recommendedMaxWorkingSetSize`, not the Mac's memory: unified memory is shared with everything
         * else the Mac is doing, and Apple calls this figure an approximation.
         */
        [[nodiscard]] DeviceMemoryInfo getMemoryInfo() const override
        {
            const std::size_t total = capabilities_.recommended_max_working_set_bytes;
            const std::size_t used = device_->currentAllocatedSize();

            return DeviceMemoryInfo{ used < total ? total - used : 0, total };
        }

        [[nodiscard]] const MetalCapabilities& capabilities() const noexcept
        {
            return capabilities_;
        }

        /// The device, for the execution context of this backend.
        [[nodiscard]] MTL::Device* handle() const noexcept
        {
            return device_;
        }

        /**
         * @brief A shared buffer of this many bytes, as the address of its contents.
         *
         * @throws MetalBadAlloc When it is larger than the largest buffer this device allows, or the device cannot make it.
         */
        [[nodiscard]] void* allocate( std::size_t bytes )
        {
            if ( bytes > capabilities_.max_buffer_bytes )
            {
                throw MetalBadAlloc( std::format(
                    "A Metal buffer of {} bytes was asked for, and this GPU allows at most {}.",
                    bytes, capabilities_.max_buffer_bytes ) );
            }

            MTL::Buffer* const buffer = device_->newBuffer( bytes, MTL::ResourceStorageModeShared );

            if ( !buffer )
            {
                throw MetalBadAlloc( std::format( "The GPU could not make a Metal buffer of {} bytes.", bytes ) );
            }

            void* const contents = buffer->contents();
            allocations_.add( contents, bytes, buffer );

            return contents;
        }

        /// Free what allocate returned. Work already encoded on it completes first: a command buffer keeps its buffers.
        void deallocate( void* contents ) noexcept
        {
            if ( MTL::Buffer* const buffer = allocations_.remove( contents ) )
            {
                buffer->release();
            }
        }

        /// @copydoc MetalAllocationMap::locate
        [[nodiscard]] MetalBufferLocation locate( const void* address ) const
        {
            return allocations_.locate( address );
        }

        [[nodiscard]] MetalKernelLibrary& kernels() noexcept
        {
            return *kernels_;
        }

    private:

        MTL::Device* device_{ nullptr };
        MetalCapabilities capabilities_;
        MetalAllocationMap allocations_;
        std::unique_ptr<MetalKernelLibrary> kernels_;
    };
}
