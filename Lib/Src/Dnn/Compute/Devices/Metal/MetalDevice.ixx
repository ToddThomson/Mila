/**
 * @file MetalDevice.ixx
 * @brief The GPU of an Apple M-series chip: its checks, its allocations, how firmly it holds them, and its kernels.
 */

module;

namespace MTL
{
    class Device;
    class ResidencySet;
}

export module Compute.MetalDevice;

import std;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Compute.MemoryHold;
import Compute.MetalAllocationMap;
import Compute.MetalBufferLocation;
import Compute.MetalCapabilities;
import Compute.MetalKernelLibrary;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The one Metal device of a Mac.
     *
     * Created by DeviceRegistry and shared. Constructing it is what refuses a Mac Mila cannot run on, so that a build
     * with Metal in it still starts there and everything that needs no GPU keeps working. Its member bodies are in
     * Implementation/MetalDevice.cpp.
     */
    export class MetalDevice : public Device
    {
    public:

        /// @throws std::runtime_error In words for the person running it, when this Mac cannot run Mila on its GPU.
        explicit MetalDevice( DeviceConstructionKey key );

        ~MetalDevice() override;

        MetalDevice( const MetalDevice& ) = delete;
        MetalDevice& operator=( const MetalDevice& ) = delete;

        /// Whether this Mac has a Metal device at all, which registration asks.
        [[nodiscard]] static bool isPresent();

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

        /// "Metal:0", as a device is named everywhere in Mila. The chip's own name is getHardwareName.
        [[nodiscard]] std::string getDeviceName() const override
        {
            return getDeviceId().toString();
        }

        /// The chip's name as Metal reports it, such as "Apple M1 Pro".
        [[nodiscard]] std::string getHardwareName() const override
        {
            return capabilities_.device_name;
        }

        /// A Mac has one Metal device, so its type names it, and a setting saved on one Mac means the GPU of another.
        [[nodiscard]] std::string getStableIdentity() const override
        {
            return "metal";
        }

        /**
         * @brief What is left of the working set Apple recommends, and that recommendation.
         *
         * The total is `recommendedMaxWorkingSetSize`, not the Mac's memory: unified memory is shared with everything
         * else the Mac is doing, and Apple calls this figure an approximation. Free is the total less what Mila has
         * allocated, less what the GPU driver holds for other applications, as far as the driver says
         * (MetalBackend.md section 5, W3d). For a second or two after Mila uses the GPU without a requested hold, the
         * driver's figure includes Mila's own memory and free reads that much low.
         */
        [[nodiscard]] DeviceMemoryInfo getMemoryInfo() const override;

        /**
         * @brief Keep Mila's memory as firmly as `hold` says (MemoryHold), now and for what is allocated later.
         *
         * Locking first reads every page on eight threads, which is the fastest way back for memory macOS compressed
         * while nothing held it (MetalBackend.md section 5, W3b). A lock macOS refuses is reported and leaves the rest of
         * the memory as it was: Mila still runs, with less of its memory held.
         */
        void setMemoryHold( MemoryHold hold ) override;

        [[nodiscard]] MemoryHold getMemoryHold() const override
        {
            return hold_.load();
        }

        /**
         * @brief The bytes of Mila's buffers that are not in RAM, page by page (`mincore`).
         *
         * macOS counts a shared buffer's pages as the process's own, and compresses them like any other memory
         * (MetalBackend.md section 5, W3f). Cheap enough to read while a hold is bringing them back, as its progress.
         */
        [[nodiscard]] std::size_t getPagedOutBytes() const override;

        [[nodiscard]] const MetalCapabilities& capabilities() const noexcept
        {
            return capabilities_;
        }

        /// The device, for the execution context of this backend.
        [[nodiscard]] MTL::Device* handle() const noexcept
        {
            return device_;
        }

        /// Every allocation while the hold is `LockedAndRequested`, and nothing otherwise. Each command queue attaches it.
        [[nodiscard]] MTL::ResidencySet* residencySet() const noexcept
        {
            return residency_;
        }

        /**
         * @brief A shared buffer of this many bytes, as the address of its contents, held as the device's hold says.
         *
         * Every Metal allocation in Mila is made here, so that each is in the map a kernel binds through and none is
         * missed by the hold (MetalBackend.md section 5, *Binding by pointer*).
         *
         * @throws MetalBadAlloc When it is larger than the largest buffer this device allows, or the device cannot make it.
         */
        [[nodiscard]] void* allocate( std::size_t bytes );

        /// Free what allocate returned. Work already encoded on it completes first: a command buffer keeps its buffers.
        void deallocate( void* contents ) noexcept;

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

        void lockAllocations();
        void unlockAllocations() noexcept;
        void requestAllocations();
        void endRequest() noexcept;
        void renewRequest( std::stop_token stop );

        MTL::Device* device_{ nullptr };
        MTL::ResidencySet* residency_{ nullptr };
        MetalCapabilities capabilities_;
        MetalAllocationMap allocations_;
        std::unique_ptr<MetalKernelLibrary> kernels_;

        // Taken by allocate, deallocate and a change of hold, so that an allocation is held exactly as the hold says.
        mutable std::mutex hold_mutex_;
        std::atomic<MemoryHold> hold_{ MemoryHold::None };
        std::condition_variable_any renewal_wait_;
        std::jthread renewer_;
        bool lock_refusal_reported_{ false };
    };
}
