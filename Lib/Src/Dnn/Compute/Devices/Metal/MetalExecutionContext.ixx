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
import Compute.MetalMatrixView;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

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

            if ( decode_position_ )
            {
                device_->deallocate( decode_position_ );
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

            for ( MTL::Tensor* const tensor : tensors_in_flight_ )
            {
                tensor->release();
            }

            tensors_in_flight_.clear();

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
         * @brief Write the position the next decode step runs at, where its kernels read it.
         *
         * Open work finishes first, since the host writes this memory directly and that work may read the old position. A
         * decode step ends by reading its token back, so at a step's start there is normally none.
         */
        void setDecodePosition( dim_t position ) override
        {
            synchronize();

            if ( !decode_position_ )
            {
                decode_position_ = static_cast<int*>( device_->allocate( sizeof( int ) ) );
            }

            *decode_position_ = narrowToKernelIndex( position );
        }

        /// Where a decode kernel binds the position. @throws std::logic_error Before the position has been set.
        [[nodiscard]] const void* decodePositionAddress() const
        {
            if ( !decode_position_ )
            {
                throw std::logic_error( "A Metal decode ran before its position was set on the execution context." );
            }

            return decode_position_;
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

        /**
         * @brief Encode a kernel over a grid of threadgroups, each of `simdgroups` SIMD-groups.
         *
         * Buffers and constants bind as `dispatch` binds them. A kernel written for one SIMD-group per threadgroup reads
         * its position in this grid from `threadgroup_position_in_grid`.
         */
        void dispatchThreadgroups(
            MTL::ComputePipelineState* pipeline,
            std::span<const void* const> buffers,
            std::span<const std::byte> constants,
            std::size_t width,
            std::size_t height,
            std::size_t simdgroups = 1 )
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

            encoder->dispatchThreadgroups(
                MTL::Size( width, height, 1 ), MTL::Size( pipeline->threadExecutionWidth() * simdgroups, 1, 1 ) );
        }

        /// As `dispatchThreadgroups` with one SIMD-group, and two separate constants at the indices after the buffers.
        void dispatchThreadgroupsWithTwoConstants(
            MTL::ComputePipelineState* pipeline,
            std::span<const void* const> buffers,
            std::span<const std::byte> first_constants,
            std::span<const std::byte> second_constants,
            std::size_t width,
            std::size_t height )
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

            encoder->setBytes( first_constants.data(), first_constants.size(), index++ );
            encoder->setBytes( second_constants.data(), second_constants.size(), index );

            encoder->dispatchThreadgroups( MTL::Size( width, height, 1 ), MTL::Size( pipeline->threadExecutionWidth(), 1, 1 ) );
        }

        /**
         * @brief Encode a kernel that takes its matrices as Metal tensors, gathered in one struct at index 0.
         *
         * The tensors live until the work has finished. The last matrix is the one written; a measured trap is that a
         * tensor bound alone gives all-zero output with no error, which is why these go through a struct of resource ids
         * (MetalBackend.md section 4.2).
         *
         * @throws std::runtime_error When Metal refuses a tensor, saying why.
         */
        void dispatchMatrices(
            MTL::ComputePipelineState* pipeline,
            std::span<const MetalMatrixView> matrices,
            std::size_t width,
            std::size_t height,
            std::size_t simdgroups )
        {
            LocalPool pool;

            MTL::ComputeCommandEncoder* const encoder = computeEncoder();
            encoder->setComputePipelineState( pipeline );

            std::vector<MTL::ResourceID> identifiers;
            identifiers.reserve( matrices.size() );

            for ( std::size_t i = 0; i < matrices.size(); ++i )
            {
                const MetalBufferLocation location = device_->locate( matrices[i].address );
                MTL::Tensor* const tensor = makeTensor( matrices[i], location );
                const MTL::ResourceUsage usage = i + 1 == matrices.size() ? MTL::ResourceUsageWrite : MTL::ResourceUsageRead;

                tensors_in_flight_.push_back( tensor );
                identifiers.push_back( tensor->gpuResourceID() );

                encoder->useResource( tensor, usage );
                encoder->useResource( location.buffer, usage );
            }

            encoder->setBytes( identifiers.data(), identifiers.size() * sizeof( MTL::ResourceID ), 0 );
            encoder->dispatchThreadgroups(
                MTL::Size( width, height, 1 ), MTL::Size( pipeline->threadExecutionWidth() * simdgroups, 1, 1 ) );
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

        /// A new tensor over a view. The caller owns the one reference it returns.
        static MTL::Tensor* makeTensor( const MetalMatrixView& view, const MetalBufferLocation& location )
        {
            if ( view.data_type != TensorDataType::FP16 && view.data_type != TensorDataType::BF16 )
            {
                throw std::logic_error( "A Metal matrix multiply takes FP16 or BF16." );
            }

            // Metal lists a tensor's axes from the fastest varying, the opposite of a row and column.
            NS::Integer extents[2] = { view.columns, view.rows };
            NS::Integer strides[2] = { 1, view.row_stride };

            MTL::TensorExtents* const dimensions = MTL::TensorExtents::alloc()->init( 2, extents );
            MTL::TensorExtents* const steps = MTL::TensorExtents::alloc()->init( 2, strides );
            MTL::TensorDescriptor* const descriptor = MTL::TensorDescriptor::alloc()->init();

            descriptor->setDataType( view.data_type == TensorDataType::FP16 ? MTL::TensorDataTypeFloat16 : MTL::TensorDataTypeBFloat16 );
            descriptor->setUsage( MTL::TensorUsageCompute );
            descriptor->setDimensions( dimensions );
            descriptor->setStrides( steps );

            NS::Error* error = nullptr;
            MTL::Tensor* const tensor = location.buffer->newTensor( descriptor, location.offset, &error );

            const std::string reason = tensor || !error ? std::string{} : error->localizedDescription()->utf8String();

            descriptor->release();
            steps->release();
            dimensions->release();

            if ( !tensor )
            {
                throw std::runtime_error( std::format( "Metal refused a matrix of {} by {}: {}", view.rows, view.columns, reason ) );
            }

            return tensor;
        }

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
        std::vector<MTL::Tensor*> tensors_in_flight_;
        int* decode_position_{ nullptr };
    };
}
