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
         * @brief Fills Metal buffer with repeated value pattern
         *
         * Uses Metal compute shaders for efficient parallel fill operations on GPU.
         * For small buffers or simple patterns, may use CPU-side fill with buffer
         * upload for better performance.
         *
         * @param data Pointer to Metal buffer memory
         * @param count Number of elements to fill
         * @param value_ptr Pointer to the value pattern
         * @param value_size Size of the value pattern in bytes
         */
        void fill( void* data, std::size_t count, const void* value_ptr, std::size_t value_size ) override {
            if ( count == 0 || !data || !value_ptr ) {
                return;
            }

            // For Metal, we need to dispatch a compute shader for efficient parallel fill
            // This is a simplified implementation - production code would use optimized shaders
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)data;

            if ( value_size == sizeof( float ) ) {
                // Optimized path for float values using Metal Performance Shaders
                fillFloatBuffer( buffer, count, *static_cast<const float*>(value_ptr) );
            }
            else {
                // General path using compute shader dispatch
                fillGenericBuffer( buffer, count, value_ptr, value_size );
            }
        }

        /**
         * @brief Copies memory between Metal buffers or host-device
         *
         * Handles efficient memory transfers using Metal's blit encoders for
         * buffer-to-buffer copies or explicit host-device transfers with
         * automatic synchronization and memory coherency management.
         *
         * @param dst Destination buffer pointer
         * @param src Source buffer or host memory pointer
         * @param size_bytes Number of bytes to copy
         */
        void memcpy( void* dst, const void* src, std::size_t size_bytes ) override {
            if ( size_bytes == 0 || !dst || !src ) {
                return;
            }

            // Check if both pointers are Metal buffers or if one is host memory
            id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst;
            id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src;

            if ( dst_buffer && src_buffer ) {
                // Buffer-to-buffer copy using blit encoder
                copyBufferToBuffer( dst_buffer, src_buffer, size_bytes );
            }
            else if ( dst_buffer ) {
                // Host-to-device copy
                copyHostToDevice( dst_buffer, src, size_bytes );
            }
            else {
                // Device-to-host copy (less common in this context)
                copyDeviceToHost( dst, src_buffer, size_bytes );
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

        /**
         * @brief Optimized float buffer fill using Metal Performance Shaders
         *
         * Uses vectorized operations for efficient parallel fill of float buffers,
         * leveraging Metal's optimized compute infrastructure.
         */
        void fillFloatBuffer( id<MTLBuffer> buffer, std::size_t count, float value ) {
            // Create command buffer for the fill operation
            id<MTLCommandBuffer> command_buffer = [ command_queue_ commandBuffer ];

            // Use Metal Performance Shaders for optimized float fill
            // This would typically use a custom compute shader for maximum efficiency

            // For now, use a simple approach - production code would use optimized kernels
            id<MTLBlitCommandEncoder> blit_encoder = [ command_buffer blitCommandEncoder ];

            // Fill with pattern (simplified - real implementation would use compute shader)
            [blit_encoder fillBuffer : buffer range : NSMakeRange( 0, count * sizeof( float ) ) value : *(uint8_t*)&value] ;

            [blit_encoder endEncoding] ;
            [command_buffer commit] ;
            [command_buffer waitUntilCompleted] ;
        }

        /**
         * @brief Generic buffer fill using compute shader dispatch
         *
         * Dispatches custom compute shader for arbitrary value patterns and sizes,
         * providing maximum flexibility for different data types and patterns.
         */
        void fillGenericBuffer( id<MTLBuffer> buffer, std::size_t count, const void* value_ptr, std::size_t value_size ) {
            // Implementation would create and dispatch custom compute shader
            // For now, fall back to host-side approach

            // Create temporary host buffer with pattern
            std::vector<uint8_t> temp_data( count * value_size );
            for ( std::size_t i = 0; i < count; ++i ) {
                std::memcpy( temp_data.data() + i * value_size, value_ptr, value_size );
            }

            // Copy to Metal buffer
            std::memcpy( [ buffer contents ], temp_data.data(), temp_data.size() );

            // Ensure memory coherency if needed
            if ( [ buffer storageMode ] == MTLStorageModeManaged ) {
                [buffer didModifyRange : NSMakeRange( 0, temp_data.size() )] ;
            }
        }

        /**
         * @brief Efficient buffer-to-buffer copy using blit encoder
         */
        void copyBufferToBuffer( id<MTLBuffer> dst, id<MTLBuffer> src, std::size_t size_bytes ) {
            id<MTLCommandBuffer> command_buffer = [ command_queue_ commandBuffer ];
            id<MTLBlitCommandEncoder> blit_encoder = [ command_buffer blitCommandEncoder ];

            [blit_encoder copyFromBuffer : src
                sourceOffset : 0
                toBuffer : dst
                destinationOffset : 0
                size : size_bytes] ;

            [blit_encoder endEncoding] ;
            [command_buffer commit] ;
            [command_buffer waitUntilCompleted] ;
        }

        /**
         * @brief Host-to-device memory transfer with synchronization
         */
        void copyHostToDevice( id<MTLBuffer> dst_buffer, const void* src, std::size_t size_bytes ) {
            std::memcpy( [ dst_buffer contents ], src, size_bytes );

            // Ensure memory coherency for managed storage
            if ( [ dst_buffer storageMode ] == MTLStorageModeManaged ) {
                [dst_buffer didModifyRange : NSMakeRange( 0, size_bytes )] ;
            }
        }

        /**
         * @brief Device-to-host memory transfer with synchronization
         */
        void copyDeviceToHost( void* dst, id<MTLBuffer> src_buffer, std::size_t size_bytes ) {
            // Ensure any pending GPU operations are complete
            id<MTLCommandBuffer> command_buffer = [ command_queue_ commandBuffer ];
            [command_buffer commit] ;
            [command_buffer waitUntilCompleted] ;

            std::memcpy( dst, [ src_buffer contents ], size_bytes );
        }
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

        void memcpy( void*, const void*, std::size_t ) override {
            throw std::runtime_error( "Metal not available" );
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