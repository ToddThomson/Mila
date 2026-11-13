/**
 * @file MetalDevice.cpp
 * @brief The bodies of MetalDevice's members; the interface is MetalDevice.ixx.
 */

module;

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

extern "C" long long milaMetalGpuDriverMemoryInUse();

module Compute.MetalDevice;

import std;
import Compute.DeviceMemoryInfo;
import Compute.MemoryHold;
import Compute.MetalBadAlloc;
import Compute.MetalCapabilities;
import Logging.Logger;

namespace Mila::Dnn::Compute
{
    namespace
    {
        constexpr auto kRenewalInterval = std::chrono::milliseconds( 1000 );
        constexpr unsigned kPageReadingThreads = 8;

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

        struct Region
        {
            const std::byte* base;
            std::size_t bytes;
        };

        // A read of one byte in every page brings back what macOS compressed or paged out, on several threads at once,
        // before the lock wires it: the order that brought 7.5 GiB back in about a second (MetalBackend.md 5, W3b).
        void readEveryPage( const std::vector<Region>& regions )
        {
            const std::size_t page = static_cast<std::size_t>( ::getpagesize() );
            std::vector<const std::byte*> pages;

            for ( const Region& region : regions )
            {
                for ( std::size_t offset = 0; offset < region.bytes; offset += page )
                {
                    pages.push_back( region.base + offset );
                }
            }

            const std::size_t share = ( pages.size() + kPageReadingThreads - 1 ) / kPageReadingThreads;
            std::vector<std::jthread> readers;

            for ( std::size_t first = 0; first < pages.size(); first += share )
            {
                readers.emplace_back( [&pages, first, last = std::min( first + share, pages.size() )]
                {
                    for ( std::size_t i = first; i < last; ++i )
                    {
                        static_cast<void>( *static_cast<const volatile std::byte*>( pages[ i ] ) );
                    }
                } );
            }
        }
    }

    MetalDevice::MetalDevice( [[maybe_unused]] DeviceConstructionKey key )
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

        MTL::ResidencySetDescriptor* const descriptor = MTL::ResidencySetDescriptor::alloc()->init();
        NS::Error* error = nullptr;
        residency_ = device_->newResidencySet( descriptor, &error );
        descriptor->release();

        if ( !residency_ )
        {
            device_->release();
            device_ = nullptr;

            throw std::runtime_error( "The GPU could not make the set that keeps Mila's memory ready for it." );
        }

        kernels_ = std::make_unique<MetalKernelLibrary>( device_ );
    }

    MetalDevice::~MetalDevice()
    {
        renewer_ = {};
        kernels_.reset();

        if ( residency_ )
        {
            residency_->endResidency();
            residency_->release();
        }

        if ( device_ )
        {
            device_->release();
        }
    }

    bool MetalDevice::isPresent()
    {
        MTL::Device* const device = MTL::CreateSystemDefaultDevice();

        if ( !device )
        {
            return false;
        }

        device->release();

        return true;
    }

    DeviceMemoryInfo MetalDevice::getMemoryInfo() const
    {
        const std::size_t total = capabilities_.recommended_max_working_set_bytes;
        const std::size_t mine = device_->currentAllocatedSize();

        // Mila's own memory is in the driver's figure for certain only while it is requested; otherwise only for a
        // second or two after a use, when what is counted as another application's is Mila's too, which errs towards
        // less free.
        const std::size_t mine_held_by_driver = hold_.load() == MemoryHold::LockedAndRequested ? mine : 0;
        const long long driver = milaMetalGpuDriverMemoryInUse();
        const std::size_t others = driver > 0 && static_cast<std::size_t>( driver ) > mine_held_by_driver
            ? static_cast<std::size_t>( driver ) - mine_held_by_driver : 0;
        const std::size_t used = mine + others;

        return DeviceMemoryInfo{ used < total ? total - used : 0, total };
    }

    void MetalDevice::setMemoryHold( MemoryHold hold )
    {
        std::jthread ended_renewer;

        {
            std::lock_guard lock( hold_mutex_ );
            const MemoryHold current = hold_.load();

            if ( hold == current )
            {
                return;
            }

            if ( current == MemoryHold::LockedAndRequested )
            {
                ended_renewer = std::move( renewer_ );
                endRequest();
            }

            if ( current == MemoryHold::None )
            {
                lockAllocations();
            }

            if ( hold == MemoryHold::None )
            {
                unlockAllocations();
            }

            if ( hold == MemoryHold::LockedAndRequested )
            {
                requestAllocations();
                renewer_ = std::jthread( [this]( std::stop_token stop ) { renewRequest( stop ); } );
            }

            hold_.store( hold );
        }

        // The renewer takes the lock between waits, so it is ended once the lock is let go.
    }

    void* MetalDevice::allocate( std::size_t bytes )
    {
        if ( bytes > capabilities_.max_buffer_bytes )
        {
            throw MetalBadAlloc( std::format(
                "A Metal buffer of {} bytes was asked for, and this GPU allows at most {}.",
                bytes, capabilities_.max_buffer_bytes ) );
        }

        std::lock_guard lock( hold_mutex_ );
        MTL::Buffer* const buffer = device_->newBuffer( bytes, MTL::ResourceStorageModeShared );

        if ( !buffer )
        {
            throw MetalBadAlloc( std::format( "The GPU could not make a Metal buffer of {} bytes.", bytes ) );
        }

        void* const contents = buffer->contents();
        allocations_.add( contents, bytes, buffer );

        const MemoryHold hold = hold_.load();

        if ( hold != MemoryHold::None && ::mlock( contents, bytes ) != 0 && !lock_refusal_reported_ )
        {
            lock_refusal_reported_ = true;
            Logging::Logger::warning( std::format(
                "MetalDevice: macOS refused to lock a buffer of {} bytes ({}); it is not held", bytes, std::strerror( errno ) ) );
        }

        if ( hold == MemoryHold::LockedAndRequested )
        {
            residency_->addAllocation( buffer );
            residency_->commit();
        }

        return contents;
    }

    void MetalDevice::deallocate( void* contents ) noexcept
    {
        std::lock_guard lock( hold_mutex_ );
        MTL::Buffer* const buffer = allocations_.remove( contents );

        if ( !buffer )
        {
            return;
        }

        const MemoryHold hold = hold_.load();

        if ( hold == MemoryHold::LockedAndRequested )
        {
            residency_->removeAllocation( buffer );
            residency_->commit();
        }

        if ( hold != MemoryHold::None )
        {
            ::munlock( contents, buffer->length() );
        }

        buffer->release();
    }

    void MetalDevice::lockAllocations()
    {
        std::vector<Region> regions;
        allocations_.forEach( [&regions]( void* base, std::size_t bytes, MTL::Buffer* )
        {
            regions.push_back( Region{ static_cast<const std::byte*>( base ), bytes } );
        } );

        readEveryPage( regions );

        std::size_t total = 0;

        for ( const Region& region : regions )
        {
            total += region.bytes;
        }

        std::size_t locked = 0;

        for ( const Region& region : regions )
        {
            if ( ::mlock( region.base, region.bytes ) != 0 )
            {
                Logging::Logger::warning( std::format(
                    "MetalDevice: macOS refused to lock Mila's memory after {} of {} bytes ({}); the rest is not held",
                    locked, total, std::strerror( errno ) ) );
                lock_refusal_reported_ = true;

                return;
            }

            locked += region.bytes;
        }
    }

    void MetalDevice::unlockAllocations() noexcept
    {
        allocations_.forEach( []( void* base, std::size_t bytes, MTL::Buffer* )
        {
            ::munlock( base, bytes );
        } );

        lock_refusal_reported_ = false;
    }

    void MetalDevice::requestAllocations()
    {
        allocations_.forEach( [this]( void*, std::size_t, MTL::Buffer* buffer )
        {
            residency_->addAllocation( buffer );
        } );

        residency_->commit();
        residency_->requestResidency();
    }

    void MetalDevice::endRequest() noexcept
    {
        residency_->endResidency();
        residency_->removeAllAllocations();
        residency_->commit();
    }

    void MetalDevice::renewRequest( std::stop_token stop )
    {
        std::unique_lock lock( hold_mutex_ );

        // The GPU driver lets a request go after a second or two idle; renewed, it keeps every use at once (W3a).
        while ( !renewal_wait_.wait_for( lock, stop, kRenewalInterval, [] { return false; } ) && !stop.stop_requested() )
        {
            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
            residency_->requestResidency();
            pool->release();
        }
    }
}
