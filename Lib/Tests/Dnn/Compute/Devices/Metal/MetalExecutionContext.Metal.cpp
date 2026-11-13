/**
 * @file MetalExecutionContext.Metal.cpp
 * @brief One queue of Metal work: what it binds, the order it runs in, and what it refuses.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;

namespace Mila::Tests::Dnn::Compute
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr std::size_t kElements = 4096;

        /// Encode the FP32 fill kernel over part of an allocation.
        void encodeFill( MetalExecutionContext& context, const void* destination, float value, std::size_t elements )
        {
            context.dispatch(
                context.device().kernels().pipeline( "fill_float" ),
                std::span<const void* const>( &destination, 1 ),
                std::as_bytes( std::span<const float>( &value, 1 ) ),
                elements );
        }

        // The free figure subtracts what the GPU driver holds for other work, which for a second or more after a use
        // includes this process's own (MetalDevice::getMemoryInfo). Read once it has not moved for two seconds, which is
        // past the driver's release after the tests before, at most ten seconds.
        std::size_t settledFree( const MetalDevice& device )
        {
            std::size_t last = device.getMemoryInfo().free_bytes;
            int unchanged = 0;

            for ( int reading = 0; reading < 40 && unchanged < 8; ++reading )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 250 ) );
                const std::size_t now = device.getMemoryInfo().free_bytes;

                unchanged = now == last ? unchanged + 1 : 0;
                last = now;
            }

            return last;
        }

        class Allocation
        {
        public:

            Allocation()
                : device_( MetalDevice::at( 0 ) ),
                  values_( static_cast<float*>( device_->allocate( kElements * sizeof( float ) ) ) )
            {
                std::fill_n( values_, kElements, -1.0f );
            }

            ~Allocation()
            {
                device_->deallocate( values_ );
            }

            [[nodiscard]] float* values() const noexcept
            {
                return values_;
            }

        private:

            std::shared_ptr<MetalDevice> device_;
            float* values_;
        };
    }

    TEST( MetalExecutionContext, IsWhatTheFactoryAndTheTraitsGiveForMetal )
    {
        const std::unique_ptr<IExecutionContext> context = createExecutionContext( Device::Metal( 0 ) );

        EXPECT_EQ( context->getDeviceId(), Device::Metal( 0 ) );
        EXPECT_NE( dynamic_cast<MetalExecutionContext*>( context.get() ), nullptr );
        EXPECT_TRUE( ( std::is_same_v<ExecutionContext<DeviceType::Metal>, MetalExecutionContext> ) );
    }

    TEST( MetalExecutionContext, ADeviceThatIsNotMetalIsRefused )
    {
        EXPECT_THROW( MetalExecutionContext context( Device::Cpu() ), std::invalid_argument );
    }

    TEST( MetalExecutionContext, SynchronizingWithNothingEncodedDoesNothing )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );

        EXPECT_NO_THROW( context.synchronize() );
        EXPECT_NO_THROW( context.synchronize() );
    }

    TEST( MetalExecutionContext, NothingRunsUntilSynchronized )
    {
        Allocation allocation;
        MetalExecutionContext context( Device::Metal( 0 ) );

        encodeFill( context, allocation.values(), 3.0f, kElements );

        EXPECT_EQ( allocation.values()[0], -1.0f );

        context.synchronize();

        EXPECT_EQ( allocation.values()[0], 3.0f );
        EXPECT_EQ( allocation.values()[kElements - 1], 3.0f );
    }

    TEST( MetalExecutionContext, ScratchGrowsOnDemandUntilItIsReserved )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );

        void* const first = context.getDeviceScratchBuffer( 100 );

        EXPECT_NE( first, nullptr );
        EXPECT_EQ( context.getDeviceScratchBuffer( 100 ), first );
        EXPECT_EQ( context.getDeviceScratchBuffer( 50 ), first );
        EXPECT_NO_THROW( (void)context.getDeviceScratchBuffer( 4096 ) );

        context.reserveScratch( 300 );

        EXPECT_NE( context.getDeviceScratchBuffer( 300 ), nullptr );
        EXPECT_THROW( (void)context.getDeviceScratchBuffer( 301 ), std::logic_error );
    }

    TEST( MetalExecutionContext, ReservingNoScratchRefusesEveryRequest )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );

        context.reserveScratch( 0 );

        EXPECT_THROW( (void)context.getDeviceScratchBuffer( 1 ), std::logic_error );
    }

    TEST( MetalExecutionContext, WorkEncodedOnScratchStillRunsAfterItIsReplaced )
    {
        Allocation allocation;
        MetalExecutionContext context( Device::Metal( 0 ) );

        auto* const scratch = static_cast<float*>( context.getDeviceScratchBuffer( kElements * sizeof( float ) ) );
        std::fill_n( scratch, kElements, 9.0f );

        context.copyBytes( scratch, allocation.values(), kElements * sizeof( float ) );

        EXPECT_NE( context.getDeviceScratchBuffer( 2 * kElements * sizeof( float ) ), scratch );

        context.synchronize();

        EXPECT_EQ( allocation.values()[0], 9.0f );
        EXPECT_EQ( allocation.values()[kElements - 1], 9.0f );
    }

    TEST( MetalExecutionContext, ScratchIsFreedWhenItIsReplacedAndWithItsContext )
    {
        constexpr std::size_t kBytes = std::size_t{ 64 } * 1024 * 1024;

        // Nothing here uses the GPU, so once the figure has settled only these allocations move it.
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );
        const std::size_t free_before = settledFree( *device );

        {
            MetalExecutionContext context( Device::Metal( 0 ) );

            (void)context.getDeviceScratchBuffer( kBytes );

            EXPECT_LE( device->getMemoryInfo().free_bytes + kBytes, free_before );

            // Still large, so that freeing it with the context shows.
            context.reserveScratch( kBytes / 2 );

            EXPECT_GT( device->getMemoryInfo().free_bytes + kBytes, free_before );
            EXPECT_LE( device->getMemoryInfo().free_bytes + kBytes / 2, free_before );
        }

        EXPECT_EQ( device->getMemoryInfo().free_bytes, free_before );
    }

    TEST( MetalExecutionContext, AKernelBindsAnInteriorAddressAndWritesOnlyItsElements )
    {
        Allocation allocation;
        MetalExecutionContext context( Device::Metal( 0 ) );

        encodeFill( context, allocation.values() + 100, 5.0f, 200 );
        context.synchronize();

        EXPECT_EQ( allocation.values()[99], -1.0f );
        EXPECT_EQ( allocation.values()[100], 5.0f );
        EXPECT_EQ( allocation.values()[299], 5.0f );
        EXPECT_EQ( allocation.values()[300], -1.0f );
    }

    TEST( MetalExecutionContext, ComputeAndBlitRunInTheOrderEncoded )
    {
        Allocation first;
        Allocation second;
        MetalExecutionContext context( Device::Metal( 0 ) );

        constexpr std::size_t bytes = kElements * sizeof( float );

        encodeFill( context, first.values(), 1.0f, kElements );
        context.copyBytes( first.values(), second.values(), bytes );
        encodeFill( context, first.values(), 2.0f, kElements );
        context.synchronize();

        EXPECT_EQ( second.values()[kElements - 1], 1.0f );
        EXPECT_EQ( first.values()[kElements - 1], 2.0f );

        context.zeroBytes( second.values(), bytes );
        context.copyBytes( first.values(), second.values(), bytes / 2 );
        context.synchronize();

        EXPECT_EQ( second.values()[0], 2.0f );
        EXPECT_EQ( second.values()[kElements / 2 - 1], 2.0f );
        EXPECT_EQ( second.values()[kElements / 2], 0.0f );
    }

    TEST( MetalExecutionContext, AnAddressNoAllocationHoldsIsRefused )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        float not_a_buffer[4]{};

        EXPECT_THROW( encodeFill( context, not_a_buffer, 1.0f, 4 ), std::invalid_argument );
    }

    TEST( MetalExecutionContext, ABufferFreedAfterEncodingStaysAliveForItsWork )
    {
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );
        MetalExecutionContext context( Device::Metal( 0 ) );

        void* const freed = device->allocate( kElements * sizeof( float ) );
        encodeFill( context, freed, 1.0f, kElements );
        device->deallocate( freed );

        EXPECT_NO_THROW( context.synchronize() );
    }

    TEST( MetalExecutionContext, WorkLeftOpenRunsWhenTheContextEnds )
    {
        Allocation allocation;

        {
            MetalExecutionContext context( Device::Metal( 0 ) );
            encodeFill( context, allocation.values(), 7.0f, kElements );
        }

        EXPECT_EQ( allocation.values()[kElements - 1], 7.0f );
    }
}
