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
import Compute.DispatchTime;
import Compute.IExecutionContext;
import Compute.MetalBufferLocation;
import Compute.MetalDevice;
import Compute.MetalKernelLibrary;
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

            if ( scratch_ )
            {
                device_->deallocate( scratch_ );
            }

            releaseLoadStaging();
            queue_->release();
        }

        MetalExecutionContext( const MetalExecutionContext& ) = delete;
        MetalExecutionContext& operator=( const MetalExecutionContext& ) = delete;

        [[nodiscard]] DeviceId getDeviceId() const noexcept override
        {
            return device_id_;
        }

        /**
         * @brief Time each dispatch by the GPU's own timestamps: each in an encoder of its own, named for its kernel, with a
         *        timestamp at its start and end. Turning it on starts afresh.
         *
         * @throws std::runtime_error When this GPU cannot sample timestamps at an encoder's boundaries.
         */
        void setDispatchProfiling( bool enabled ) override
        {
            if ( enabled && !device_->handle()->supportsCounterSampling( MTL::CounterSamplingPointAtStageBoundary ) )
            {
                throw std::runtime_error( "This GPU cannot time each Metal dispatch: it does not sample timestamps at an encoder's boundaries." );
            }

            if ( enabled )
            {
                profile_.clear();
            }

            dispatch_profiling_ = enabled;
        }

        [[nodiscard]] std::vector<DispatchTime> dispatchProfile() const override
        {
            std::vector<DispatchTime> result;
            result.reserve( profile_.size() );

            for ( const auto& [kernel, time] : profile_ )
            {
                result.push_back( time );
            }

            std::ranges::sort( result, std::ranges::greater{}, &DispatchTime::gpu_seconds );

            return result;
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

            collectDispatchTimes( failed ? 0.0 : commands_->GPUEndTime() - commands_->GPUStartTime() );
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

        /// @copydoc IExecutionContext::reserveScratch
        void reserveScratch( std::size_t bytes ) override
        {
            replaceScratch( bytes );
            scratch_reserved_ = true;
        }

        /**
         * @brief The scratch buffer every operation on this context shares.
         *
         * Work on one context runs in the order it was encoded, so an operation's use of this buffer ends before the next one's starts.
         * Fetch it once per use and never keep the address: until a network reserves it, a larger request replaces it. Work already
         * encoded on the old buffer still completes, since a command buffer keeps its buffers.
         *
         * @throws std::logic_error For a request past the reservation, which is a defect in an operation's `getRequiredScratchBytes`.
         * @throws MetalBadAlloc When the GPU cannot make the buffer.
         */
        [[nodiscard]] void* getDeviceScratchBuffer( std::size_t required_bytes )
        {
            if ( required_bytes <= scratch_bytes_ )
            {
                return scratch_;
            }

            if ( scratch_reserved_ )
            {
                throw std::logic_error( std::format(
                    "MetalExecutionContext: a scratch request of {} bytes exceeds the {} bytes the network reserved; "
                    "an operation's getRequiredScratchBytes() understates what it requests",
                    required_bytes, scratch_bytes_ ) );
            }

            replaceScratch( required_bytes );

            return scratch_;
        }

        /// The most a load stages at once. A tensor larger than this is quantized in row blocks that fit, as on CUDA.
        [[nodiscard]] static constexpr std::size_t getLoadStagingLimitBytes() noexcept
        {
            return kLoadStagingLimitBytes;
        }

        /**
         * @brief The buffer a load quantizes full-precision weights through, grown on demand.
         *
         * Separate from the scratch buffer: it lives only while a load runs, which frees it through `releaseLoadStaging()`. The
         * host writes it directly, so work that reads it must have finished first. A larger request replaces it.
         *
         * @throws MetalBadAlloc When the GPU cannot make the buffer.
         */
        [[nodiscard]] void* getLoadStagingBuffer( std::size_t required_bytes )
        {
            if ( required_bytes <= load_staging_bytes_ )
            {
                return load_staging_;
            }

            releaseLoadStaging();

            load_staging_ = device_->allocate( required_bytes );
            load_staging_bytes_ = required_bytes;

            return load_staging_;
        }

        void releaseLoadStaging() noexcept override
        {
            if ( load_staging_ )
            {
                device_->deallocate( load_staging_ );
                load_staging_ = nullptr;
                load_staging_bytes_ = 0;
            }
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

            MTL::ComputeCommandEncoder* const encoder = computeEncoder( pipeline );

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

            MTL::ComputeCommandEncoder* const encoder = computeEncoder( pipeline );

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

            MTL::ComputeCommandEncoder* const encoder = computeEncoder( pipeline );

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

            MTL::ComputeCommandEncoder* const encoder = computeEncoder( pipeline );

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

        void replaceScratch( std::size_t bytes )
        {
            if ( scratch_ )
            {
                device_->deallocate( scratch_ );
                scratch_ = nullptr;
                scratch_bytes_ = 0;
            }

            if ( bytes > 0 )
            {
                scratch_ = device_->allocate( bytes );
                scratch_bytes_ = bytes;
            }
        }

        /// FP16 and BF16 are a product's activations and weights, FP8 E4M3 a weight's bare codes, and FP32 the sums they make.
        static MTL::TensorDataType tensorDataTypeOf( TensorDataType data_type )
        {
            switch ( data_type )
            {
                case TensorDataType::FP16:
                    return MTL::TensorDataTypeFloat16;
                case TensorDataType::BF16:
                    return MTL::TensorDataTypeBFloat16;
                case TensorDataType::FP8_E4M3:
                    return MTL::TensorDataTypeMetalFloat8E4M3;
                case TensorDataType::FP32:
                    return MTL::TensorDataTypeFloat32;
                default:
                    throw std::logic_error( "A Metal matrix multiply takes FP16, BF16, FP8 E4M3 codes or FP32 sums." );
            }
        }

        /// A new tensor over a view. The caller owns the one reference it returns.
        static MTL::Tensor* makeTensor( const MetalMatrixView& view, const MetalBufferLocation& location )
        {
            const MTL::TensorDataType data_type = tensorDataTypeOf( view.data_type );

            // Metal lists a tensor's axes from the fastest varying, the opposite of a row and column.
            NS::Integer extents[2] = { view.columns, view.rows };
            NS::Integer strides[2] = { 1, view.row_stride };

            MTL::TensorExtents* const dimensions = MTL::TensorExtents::alloc()->init( 2, extents );
            MTL::TensorExtents* const steps = MTL::TensorExtents::alloc()->init( 2, strides );
            MTL::TensorDescriptor* const descriptor = MTL::TensorDescriptor::alloc()->init();

            descriptor->setDataType( data_type );
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

        /// The compute encoder, set to run `pipeline`, ending a blit one so that order is kept; blit is the same the other way.
        /// Profiling dispatches, each gets an encoder of its own named for its kernel.
        MTL::ComputeCommandEncoder* computeEncoder( MTL::ComputePipelineState* pipeline )
        {
            if ( blit_encoder_ || dispatch_profiling_ )
            {
                endEncoding();
            }

            if ( !compute_encoder_ )
            {
                compute_encoder_ = dispatch_profiling_ ? timedComputeEncoder( pipeline ) : openCommands()->computeCommandEncoder()->retain();
            }

            compute_encoder_->setComputePipelineState( pipeline );

            return compute_encoder_;
        }

        /// A compute encoder for one dispatch of `pipeline`, with a GPU timestamp at its start and its end.
        MTL::ComputeCommandEncoder* timedComputeEncoder( MTL::ComputePipelineState* pipeline )
        {
            if ( sample_buffers_.empty() || sample_buffers_.back().used + 2 > kSamplesPerBuffer )
            {
                sample_buffers_.push_back( SampleBuffer{ newTimestampBuffer(), 0 } );
            }

            SampleBuffer& samples = sample_buffers_.back();
            std::string kernel = device_->kernels().nameOf( pipeline );

            MTL::ComputePassDescriptor* const descriptor = MTL::ComputePassDescriptor::computePassDescriptor();
            MTL::ComputePassSampleBufferAttachmentDescriptor* const attachment = descriptor->sampleBufferAttachments()->object( 0 );
            attachment->setSampleBuffer( samples.buffer );
            attachment->setStartOfEncoderSampleIndex( samples.used );
            attachment->setEndOfEncoderSampleIndex( samples.used + 1 );

            MTL::ComputeCommandEncoder* const encoder = openCommands()->computeCommandEncoder( descriptor )->retain();
            encoder->setLabel( NS::String::string( kernel.c_str(), NS::UTF8StringEncoding ) );

            timed_dispatches_.push_back( TimedDispatch{ std::move( kernel ), sample_buffers_.size() - 1, samples.used } );
            samples.used += 2;

            return encoder;
        }

        MTL::CounterSampleBuffer* newTimestampBuffer()
        {
            MTL::CounterSet* timestamps = nullptr;
            const NS::Array* const sets = device_->handle()->counterSets();

            for ( NS::UInteger i = 0; sets && i < sets->count(); ++i )
            {
                MTL::CounterSet* const set = sets->object<MTL::CounterSet>( i );

                if ( set->name()->isEqualToString( MTL::CommonCounterSetTimestamp ) )
                {
                    timestamps = set;
                }
            }

            if ( !timestamps )
            {
                throw std::runtime_error( "This GPU has no timestamp counters to time each Metal dispatch with." );
            }

            MTL::CounterSampleBufferDescriptor* const descriptor = MTL::CounterSampleBufferDescriptor::alloc()->init();
            descriptor->setCounterSet( timestamps );
            descriptor->setStorageMode( MTL::StorageModeShared );
            descriptor->setSampleCount( kSamplesPerBuffer );

            NS::Error* error = nullptr;
            MTL::CounterSampleBuffer* const buffer = device_->handle()->newCounterSampleBuffer( descriptor, &error );
            descriptor->release();

            if ( !buffer )
            {
                throw std::runtime_error( std::format( "Metal refused a buffer of GPU timestamps: {}",
                    error ? error->localizedDescription()->utf8String() : "no reason given" ) );
            }

            return buffer;
        }

        /**
         * @brief Add the finished command buffer's timed dispatches to the profile.
         *
         * Timestamps are GPU ticks; the command buffer's own GPU time, over the span from its first timed start to its last
         * timed end, converts them to seconds. Zero time, as a failed buffer gives, adds nothing.
         */
        void collectDispatchTimes( double command_buffer_seconds )
        {
            if ( !timed_dispatches_.empty() && command_buffer_seconds > 0.0 )
            {
                std::vector<const MTL::CounterResultTimestamp*> resolved;

                for ( const SampleBuffer& samples : sample_buffers_ )
                {
                    NS::Data* const data = samples.buffer->resolveCounterRange( NS::Range( 0, samples.used ) );
                    resolved.push_back( data ? static_cast<const MTL::CounterResultTimestamp*>( data->bytes() ) : nullptr );
                }

                std::uint64_t first = std::numeric_limits<std::uint64_t>::max();
                std::uint64_t last = 0;

                auto valid = []( std::uint64_t tick ) { return tick != MTL::CounterErrorValue && tick != 0; };

                for ( const TimedDispatch& dispatch : timed_dispatches_ )
                {
                    const MTL::CounterResultTimestamp* const ticks = resolved[dispatch.buffer];

                    if ( ticks && valid( ticks[dispatch.index].timestamp ) && valid( ticks[dispatch.index + 1].timestamp ) )
                    {
                        first = std::min( first, ticks[dispatch.index].timestamp );
                        last = std::max( last, ticks[dispatch.index + 1].timestamp );
                    }
                }

                const double seconds_per_tick = last > first ? command_buffer_seconds / static_cast<double>( last - first ) : 0.0;

                for ( const TimedDispatch& dispatch : timed_dispatches_ )
                {
                    const MTL::CounterResultTimestamp* const ticks = resolved[dispatch.buffer];

                    if ( !ticks || !valid( ticks[dispatch.index].timestamp ) || !valid( ticks[dispatch.index + 1].timestamp ) )
                    {
                        continue;
                    }

                    DispatchTime& time = profile_[dispatch.kernel];
                    time.kernel = dispatch.kernel;
                    time.dispatches += 1;
                    time.gpu_seconds += static_cast<double>( ticks[dispatch.index + 1].timestamp - ticks[dispatch.index].timestamp ) * seconds_per_tick;
                }
            }

            for ( SampleBuffer& samples : sample_buffers_ )
            {
                samples.buffer->release();
            }

            sample_buffers_.clear();
            timed_dispatches_.clear();
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
        void* scratch_{ nullptr };
        std::size_t scratch_bytes_{ 0 };
        bool scratch_reserved_{ false };
        void* load_staging_{ nullptr };
        std::size_t load_staging_bytes_{ 0 };
        // Dispatch profiling: the timestamp buffers of the open command buffer, each dispatch's place in them, and the totals.
        struct SampleBuffer
        {
            MTL::CounterSampleBuffer* buffer;
            NS::UInteger used;
        };

        struct TimedDispatch
        {
            std::string kernel;
            std::size_t buffer;
            NS::UInteger index;
        };

        static constexpr NS::UInteger kSamplesPerBuffer = 4096;

        bool dispatch_profiling_{ false };
        std::vector<SampleBuffer> sample_buffers_;
        std::vector<TimedDispatch> timed_dispatches_;
        std::unordered_map<std::string, DispatchTime> profile_;

        // As CUDA's: it bounds the load's peak, and Qwen 3.8's head is 2.54 GiB of BF16.
        static constexpr std::size_t kLoadStagingLimitBytes = std::size_t{ 256 } * 1024 * 1024;
    };
}
