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
