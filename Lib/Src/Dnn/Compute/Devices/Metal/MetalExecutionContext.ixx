/**
 * @file MetalExecutionContext.ixx
 * @brief One queue of Metal work, encoded in order and run when synchronized.
 */

module;

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

export module Compute.MetalExecutionContext;

import std;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.MetalBufferLocation;
import Compute.MetalDevice;

namespace Mila::Dnn::Compute
{
    namespace
    {
        // What Metal hands back unowned goes to the innermost pool, and there is none unless the caller is an
        // Objective-C program. The objects this file keeps are retained first, so each call can drain its own.
        class LocalPool
        {
        public:

            LocalPool()
                : pool_( NS::AutoreleasePool::alloc()->init() )
            {
            }

            ~LocalPool()
            {
                pool_->release();
            }

            LocalPool( const LocalPool& ) = delete;
            LocalPool& operator=( const LocalPool& ) = delete;

        private:

            NS::AutoreleasePool* pool_;
        };

        DeviceId validated( DeviceId device_id )
        {
            if ( device_id.type != DeviceType::Metal )
            {
                throw std::invalid_argument( std::format(
                    "MetalExecutionContext requires a Metal device, got '{}'", device_id.toString() ) );
            }

            return device_id;
        }
    }

    /**
     * @brief A command queue, with one open command buffer that work is encoded into.
     *
     * Nothing runs until `synchronize()`, which commits and waits. Work on one context runs in the order it was encoded,
     * and Metal tracks what each step reads and writes across compute and blit. Like a CUDA stream, a context is used
     * from one thread at a time. Its work is ordered against nothing on any other context.
     */
    export class MetalExecutionContext : public IExecutionContext
    {
    public:

        /// @throws std::invalid_argument For a device that is not Metal.
        /// @throws std::runtime_error When this Mac cannot run Mila on its GPU.
        explicit MetalExecutionContext( DeviceId device_id )
            : device_id_( validated( device_id ) ),
              device_( MetalDevice::at( device_id.index ) ),
              queue_( device_->handle()->newCommandQueue() )
        {
            if ( !queue_ )
            {
                throw std::runtime_error( "The GPU could not make a Metal command queue." );
            }
        }

        ~MetalExecutionContext() override
        {
            try
            {
                synchronize();
            }
            catch ( const std::exception& error )
            {
                std::println( std::cerr, "MetalExecutionContext: work still open at destruction failed: {}", error.what() );
            }

            queue_->release();
        }

        MetalExecutionContext( const MetalExecutionContext& ) = delete;
        MetalExecutionContext& operator=( const MetalExecutionContext& ) = delete;

        [[nodiscard]] DeviceId getDeviceId() const noexcept override
        {
            return device_id_;
        }

        /// @throws std::runtime_error When the GPU reports that the work failed, saying why.
        void synchronize() override
        {
            if ( !commands_ )
            {
                return;
            }

            LocalPool pool;

            endEncoding();
            commands_->commit();
            commands_->waitUntilCompleted();

            const bool failed = commands_->status() == MTL::CommandBufferStatusError;
            const std::string reason = failed && commands_->error()
                ? commands_->error()->localizedDescription()->utf8String() : "no reason given";

            commands_->release();
            commands_ = nullptr;

            if ( failed )
            {
                throw std::runtime_error( std::format( "The GPU failed to run Mila's work: {}", reason ) );
            }
        }

        [[nodiscard]] MetalDevice& device() const noexcept
        {
            return *device_;
        }

        /**
         * @brief Encode a kernel over `elements` threads.
         *
         * Buffers are bound in order from index 0, each by an address inside a live allocation of this device, and the
         * constants follow them at the next index. A `narrowToKernelIndex` at the caller is what bounds `elements`.
         */
        void dispatch(
            MTL::ComputePipelineState* pipeline,
            std::span<const void* const> buffers,
            std::span<const std::byte> constants,
            std::size_t elements )
        {
            LocalPool pool;

            MTL::ComputeCommandEncoder* const encoder = computeEncoder();
            encoder->setComputePipelineState( pipeline );

            NS::UInteger index = 0;

            for ( const void* const address : buffers )
            {
                const MetalBufferLocation location = device_->locate( address );
                encoder->setBuffer( location.buffer, location.offset, index++ );
            }

            if ( !constants.empty() )
            {
                encoder->setBytes( constants.data(), constants.size(), index );
            }

            encoder->dispatchThreads( MTL::Size( elements, 1, 1 ), MTL::Size( pipeline->threadExecutionWidth(), 1, 1 ) );
        }

        /// Encode a copy between two addresses inside live allocations of this device.
        void copyBytes( const void* source, void* destination, std::size_t bytes )
        {
            LocalPool pool;

            const MetalBufferLocation from = device_->locate( source );
            const MetalBufferLocation to = device_->locate( destination );

            blitEncoder()->copyFromBuffer( from.buffer, from.offset, to.buffer, to.offset, bytes );
        }

        /// Encode writing zero to `bytes` bytes at an address inside a live allocation of this device.
        void zeroBytes( void* destination, std::size_t bytes )
        {
            LocalPool pool;

            const MetalBufferLocation location = device_->locate( destination );

            blitEncoder()->fillBuffer( location.buffer, NS::Range( location.offset, bytes ), 0 );
        }

    private:

        /// The open command buffer, starting one when none is open.
        MTL::CommandBuffer* openCommands()
        {
            if ( !commands_ )
            {
                commands_ = queue_->commandBuffer()->retain();
            }

            return commands_;
        }

        /// The compute encoder, ending a blit one so that order is kept. Blit is the same the other way.
        MTL::ComputeCommandEncoder* computeEncoder()
        {
            if ( blit_encoder_ )
            {
                endEncoding();
            }

            if ( !compute_encoder_ )
            {
                compute_encoder_ = openCommands()->computeCommandEncoder()->retain();
            }

            return compute_encoder_;
        }

        MTL::BlitCommandEncoder* blitEncoder()
        {
            if ( compute_encoder_ )
            {
                endEncoding();
            }

            if ( !blit_encoder_ )
            {
                blit_encoder_ = openCommands()->blitCommandEncoder()->retain();
            }

            return blit_encoder_;
        }

        void endEncoding() noexcept
        {
            if ( compute_encoder_ )
            {
                compute_encoder_->endEncoding();
                compute_encoder_->release();
                compute_encoder_ = nullptr;
            }

            if ( blit_encoder_ )
            {
                blit_encoder_->endEncoding();
                blit_encoder_->release();
                blit_encoder_ = nullptr;
            }
        }

        DeviceId device_id_;
        std::shared_ptr<MetalDevice> device_;
        MTL::CommandQueue* queue_{ nullptr };
        MTL::CommandBuffer* commands_{ nullptr };
        MTL::ComputeCommandEncoder* compute_encoder_{ nullptr };
        MTL::BlitCommandEncoder* blit_encoder_{ nullptr };
    };
}
