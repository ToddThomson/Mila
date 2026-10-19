/**
 * @file DeviceArena.ixx
 * @brief One device allocation that a block's parameter tensors are placed in, so the block rounds once.
 *
 * See Specifications/MemoryFootprint.md 11.8.1.
 */

export module Compute.DeviceArena;

import std;
import Compute.MemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Where each tensor placed in an arena starts: cudaMalloc's own alignment, so a placed tensor's
     *        address is as aligned as one allocated alone.
     */
    export inline constexpr std::size_t kArenaPlacementAlignmentBytes = 256;

    /**
     * @brief Bytes a tensor of `storage_bytes` occupies once placed in an arena.
     *
     * The pricing twin of DeviceArena::place(): a prediction sums this over a block's parameters and rounds the
     * sum once, where a tensor allocated alone is rounded by itself (occupiedDeviceBytes).
     */
    export constexpr std::size_t arenaPlacedBytes( std::size_t storage_bytes ) noexcept
    {
        return ( storage_bytes + kArenaPlacementAlignmentBytes - 1 )
            / kArenaPlacementAlignmentBytes * kArenaPlacementAlignmentBytes;
    }

    /**
     * @brief One allocation of a fixed capacity, handed out front to back and never reused.
     *
     * Its capacity is the block's priced parameter bytes, so a build that places what it priced fills it exactly;
     * the owner checks that (isFull) and a mismatch is a pricing defect, not a sizing choice. Placed tensors hold
     * the arena alive and never free their part of it: parameters live as long as the block that owns them.
     *
     * Not synchronized; used only during the single-threaded build phase.
     */
    export class DeviceArena
    {
    public:

        /**
         * @brief Allocate `capacity` bytes from `resource`, which the arena owns from here on.
         *
         * @throws std::invalid_argument when resource is null.
         * @throws std::bad_alloc        when the allocation fails.
         */
        DeviceArena( std::unique_ptr<MemoryResource> resource, std::size_t capacity, std::string owner )
            : resource_( std::move( resource ) ), capacity_( capacity ), owner_( std::move( owner ) )
        {
            if ( !resource_ )
            {
                throw std::invalid_argument( std::format( "DeviceArena '{}': no memory resource", owner_ ) );
            }

            if ( capacity_ > 0 )
            {
                base_ = static_cast<std::byte*>( resource_->allocate( capacity_, kArenaPlacementAlignmentBytes ) );
            }
        }

        ~DeviceArena()
        {
            if ( base_ != nullptr )
            {
                resource_->deallocate( base_, capacity_, kArenaPlacementAlignmentBytes );
            }
        }

        DeviceArena( const DeviceArena& ) = delete;
        DeviceArena& operator=( const DeviceArena& ) = delete;
        DeviceArena( DeviceArena&& ) = delete;
        DeviceArena& operator=( DeviceArena&& ) = delete;

        /**
         * @brief Place `storage_bytes` at the next 256-byte boundary and return its address.
         *
         * @throws std::logic_error when the arena cannot hold it: the owner priced fewer bytes than its children
         *         allocate, which names the block whose pricing and build disagree.
         */
        void* place( std::size_t storage_bytes, std::string_view tensor_name )
        {
            const std::size_t placed = arenaPlacedBytes( storage_bytes );

            if ( placed > capacity_ - used_ )
            {
                throw std::logic_error( std::format(
                    "DeviceArena '{}': '{}' needs {} bytes and {} of {} remain -- the block priced fewer "
                    "parameter bytes than it allocates", owner_, tensor_name, placed, capacity_ - used_, capacity_ ) );
            }

            std::byte* address = base_ + used_;
            used_ += placed;

            return address;
        }

        std::size_t capacity() const noexcept
        {
            return capacity_;
        }

        std::size_t usedBytes() const noexcept
        {
            return used_;
        }

        bool isFull() const noexcept
        {
            return used_ == capacity_;
        }

        const std::string& owner() const noexcept
        {
            return owner_;
        }

    private:

        std::unique_ptr<MemoryResource> resource_;
        std::size_t capacity_{ 0 };
        std::size_t used_{ 0 };
        std::byte* base_{ nullptr };
        std::string owner_;
    };
}
