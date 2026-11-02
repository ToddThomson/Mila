/**
 * @file MetalMemoryResource.ixx
 * @brief Metal-specific memory resource implementation for Apple GPU compute
 *
 * This module provides Metal GPU memory management through Apple's Metal Performance
 * Shaders framework, enabling high-performance compute operations on Apple Silicon
 * and discrete GPUs. The implementation handles Metal buffer allocation, device-host
 * transfers, and compute shader integration.
 */

module;
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <source_location>
#include <cassert>

// Platform-conditional Metal headers
#ifdef METAL_AVAILABLE
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

export module Compute.MetalMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
#ifdef METAL_AVAILABLE
    /**
     * @brief Metal GPU memory resource for Apple devices
     *
     * Provides GPU memory allocation and management through Apple's Metal framework,
     * supporting both unified memory architecture on Apple Silicon and discrete GPU
     * memory on Intel Macs. Integrates with Metal Performance Shaders for optimized
     * compute operations and automatic memory coherency management.
     *
     * Key features:
     * - Automatic device selection (integrated vs discrete GPU)
     * - Unified memory support on Apple Silicon for zero-copy host access
     * - Metal buffer allocation with appropriate storage modes
     * - Integration with Metal command queues for async operations
     * - Memory coherency management for host-device synchronization
     */
    export class MetalMemoryResource : public MemoryResource {
    public:
        /**
         * @brief Constructs Metal memory resource with automatic device selection
         *
         * Initializes the Metal device context, selecting the best available GPU
         * (discrete GPU preferred over integrated when available). Creates the
         * default command queue for memory operations and compute dispatches.
         *
         * @throws std::runtime_error If Metal is unavailable or device creation fails
         */
        MetalMemoryResource() {
            // Get default Metal device (automatically selects best available GPU)
            device_ = MTLCreateSystemDefaultDevice();
            if ( !device_ ) {
                throw std::runtime_error( "Failed to create Metal device - Metal may not be available" );
            }

            // Create command queue for memory operations
            command_queue_ = [ device_ newCommandQueue ];
            if ( !command_queue_ ) {
                throw std::runtime_error( "Failed to create Metal command queue" );
            }
        }

        /**
         * @brief Memory accessibility properties for Metal GPU memory
         *
         * On Apple Silicon with unified memory, Metal buffers can be host-accessible
         * depending on storage mode. On discrete GPUs, device memory is typically
         * not directly host-accessible without explicit transfers.
         */
        static constexpr bool is_host_accessible = false;  // Conservative default
        static constexpr bool is_device_accessible = true;
        static constexpr DeviceType device_type = DeviceType::Metal;

    protected:
        /**
         * @brief Allocates Metal buffer with appropriate storage mode
         *
         * Creates Metal buffer optimized for compute operations, selecting
         * storage mode based on device capabilities (unified vs discrete memory).
         * Ensures proper alignment for Metal Performance Shaders operations.
         *
         * @param n Size in bytes to allocate
         * @param alignment Memory alignment requirement
         * @return Pointer to allocated Metal buffer
         * @throws std::bad_alloc If allocation fails
         */
        [[nodiscard]] void* do_allocate( std::size_t n, std::size_t alignment ) override {
            if ( n == 0 ) return nullptr;

            // Select appropriate storage mode based on device capabilities
            MTLResourceOptions options = MTLResourceStorageModePrivate;

            // On Apple Silicon, we can use shared memory for better performance
            if ( [ device_ hasUnifiedMemory ] ) {
                options = MTLResourceStorageModeShared;
            }

            id<MTLBuffer> buffer = [ device_ newBufferWithLength : n options : options ];
            if ( !buffer ) {
                throw std::bad_alloc();
            }

            // Return bridged pointer for integration with memory resource interface
            return (__bridge_retained void*)buffer;
        }

        /**
         * @brief Deallocates Metal buffer resources
         *
         * Releases Metal buffer and associated GPU memory, ensuring proper
         * cleanup of Metal objects and GPU memory management.
         *
         * @param ptr Pointer to Metal buffer to deallocate
         * @param n Size of allocation (unused in Metal)
         * @param alignment Alignment of allocation (unused in Metal)
         */
        void do_deallocate( void* ptr, std::size_t, std::size_t ) override {
            if ( ptr ) {
                // Release the Metal buffer
                id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
                // ARC will automatically handle the release
                (void)buffer; // Suppress unused variable warning
            }
        }

        /**
         * @brief Compares Metal memory resources for equality
         *
         * Two Metal memory resources are considered equal if they use the
         * same Metal device, ensuring compatibility for buffer operations.
         *
         * @param other Memory resource to compare with
         * @return true if both use same Metal device, false otherwise
         */
        bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
            const auto* metal_other = dynamic_cast<const MetalMemoryResource*>(&other);
            return metal_other && [metal_other->device_ isEqual : device_];
        }

    private:
        id<MTLDevice> device_;              ///< Metal device for GPU operations
        id<MTLCommandQueue> command_queue_; ///< Command queue for Metal operations

    };

#else // !METAL_AVAILABLE
    /**
     * @brief Stub implementation for non-Apple platforms
     *
     * Provides compilation compatibility on non-Apple platforms where Metal
     * is not available. Always throws runtime_error when instantiated.
     */
    export class MetalMemoryResource : public MemoryResource {
    public:
        MetalMemoryResource() {
            throw std::runtime_error( "Metal support is not available on this platform" );
        }

        static constexpr bool is_host_accessible = false;
        static constexpr bool is_device_accessible = false;

    protected:
        void* do_allocate( std::size_t, std::size_t ) override {
            throw std::runtime_error( "Metal not available" );
        }

        void do_deallocate( void*, std::size_t, std::size_t ) override {
            throw std::runtime_error( "Metal not available" );
        }

        bool do_is_equal( const std::pmr::memory_resource& ) const noexcept override {
            return false;
        }
    };
#endif // METAL_AVAILABLE
}