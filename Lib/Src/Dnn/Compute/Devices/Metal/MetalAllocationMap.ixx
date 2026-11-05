/**
 * @file MetalAllocationMap.ixx
 * @brief From an address a tensor holds to the Metal buffer it lies in.
 */

module;

namespace MTL
{
    class Buffer;
}

export module Compute.MetalAllocationMap;

import std;
import Compute.MetalBufferLocation;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The live allocations of one device, by address range.
     *
     * A tensor holds a pointer and Metal binds a buffer and an offset. A view holds an address inside an allocation, so
     * a lookup finds the range an address lies in, not only an allocation's base. One map per device, because a
     * `TensorBuffer` makes a memory resource of its own and a kernel binds buffers from many of them.
     */
    export class MetalAllocationMap
    {
    public:

        /// Record an allocation. It is the caller's that the range does not overlap a live one.
        void add( const void* base, std::size_t bytes, MTL::Buffer* buffer )
        {
            std::unique_lock lock( mutex_ );

            ranges_.insert_or_assign( address( base ), Range{ bytes, buffer } );
        }

        /// Forget an allocation by its base, returning its buffer, or null when no allocation starts there.
        MTL::Buffer* remove( const void* base ) noexcept
        {
            std::unique_lock lock( mutex_ );

            const auto found = ranges_.find( address( base ) );

            if ( found == ranges_.end() )
            {
                return nullptr;
            }

            MTL::Buffer* const buffer = found->second.buffer;
            ranges_.erase( found );

            return buffer;
        }

        /**
         * @brief The buffer an address lies in, and how far into it.
         *
         * @throws std::invalid_argument When the address lies in no live allocation: an operation was handed memory that
         *         did not come from this device, or that has been freed.
         */
        [[nodiscard]] MetalBufferLocation locate( const void* pointer ) const
        {
            const std::uintptr_t wanted = address( pointer );
            std::shared_lock lock( mutex_ );

            auto after = ranges_.upper_bound( wanted );

            if ( after != ranges_.begin() )
            {
                const auto& [base, range] = *std::prev( after );

                if ( wanted - base < range.bytes )
                {
                    return MetalBufferLocation{ range.buffer, wanted - base };
                }
            }

            throw std::invalid_argument( std::format(
                "MetalAllocationMap: address {:#x} lies in no live Metal allocation", wanted ) );
        }

        [[nodiscard]] std::size_t count() const
        {
            std::shared_lock lock( mutex_ );

            return ranges_.size();
        }

    private:

        struct Range
        {
            std::size_t bytes;
            MTL::Buffer* buffer;
        };

        [[nodiscard]] static std::uintptr_t address( const void* pointer ) noexcept
        {
            return reinterpret_cast<std::uintptr_t>( pointer );
        }

        mutable std::shared_mutex mutex_;
        std::map<std::uintptr_t, Range> ranges_;
    };
}
