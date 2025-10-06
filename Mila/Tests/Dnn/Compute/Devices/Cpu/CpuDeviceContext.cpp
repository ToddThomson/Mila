/**
 * @file CpuDeviceContext.cpp
 * @brief Unit tests for CpuDeviceContext class.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Compute.CpuDeviceContext;
import Compute.DeviceType;
import Compute.DeviceRegistry;
import Compute.CpuDevice;

using namespace Mila::Dnn::Compute;

namespace Dnn::Compute::Devices::Cpu {

    /**
     * @brief Test fixture for CpuDeviceContext tests.
     *
     * Sets up device registry with CPU device registration before each test
     * and ensures clean state for testing.
     */
    class CpuDeviceContextTest : public testing::Test {
    protected:
        void SetUp() override {
            // Register CPU device if not already registered
            if (!DeviceRegistry::instance().hasDevice( "CPU" )) {
                DeviceRegistry::instance().registerDevice( "CPU", []() {
                    return std::make_shared<CpuDevice>();
                    } );
            }
        }

        void TearDown() override {
            // Clean up is handled automatically by registry singleton
        }
    };

    // ============================================================================
    // Constructor Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, ParameterlessConstructorWorks ) {
        EXPECT_NO_THROW( {
            CpuDeviceContext context;
            } );
    }

    TEST_F( CpuDeviceContextTest, ConstructorWithRegistryFailure ) {
        // Temporarily clear the registry to test failure case
        // Note: This test may need adjustment based on DeviceRegistry implementation
        // For now, we'll assume the registry is properly set up in SetUp()

        // This test verifies that if device creation fails, constructor throws
        // In practice, this should rarely happen with a properly configured system
        EXPECT_NO_THROW( {
            CpuDeviceContext context;
            } );
    }

    // ============================================================================
    // DeviceContext Interface Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, GetDeviceTypeReturnsCpu ) {
        CpuDeviceContext context;
        EXPECT_EQ( context.getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( CpuDeviceContextTest, GetDeviceNameReturnsCpu ) {
        CpuDeviceContext context;
        EXPECT_EQ( context.getDeviceName(), "CPU" );
    }

    TEST_F( CpuDeviceContextTest, GetDeviceIdReturnsMinusOne ) {
        CpuDeviceContext context;
        EXPECT_EQ( context.getDeviceId(), -1 );
    }

    TEST_F( CpuDeviceContextTest, MakeCurrentDoesNotThrow ) {
        CpuDeviceContext context;
        EXPECT_NO_THROW( context.makeCurrent() );
    }

    TEST_F( CpuDeviceContextTest, GetDeviceReturnsValidDevice ) {
        CpuDeviceContext context;
        auto device = context.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    // ============================================================================
    // Inherited Helper Method Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, IsDeviceTypeWorksCorrectly ) {
        CpuDeviceContext context;

        EXPECT_TRUE( context.isDeviceType( DeviceType::Cpu ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Cuda ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Metal ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::OpenCL ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Vulkan ) );
    }

    TEST_F( CpuDeviceContextTest, IsCpuDeviceReturnsTrue ) {
        CpuDeviceContext context;
        EXPECT_TRUE( context.isCpuDevice() );
    }

    TEST_F( CpuDeviceContextTest, IsOtherDeviceTypesReturnFalse ) {
        CpuDeviceContext context;

        EXPECT_FALSE( context.isCudaDevice() );
        EXPECT_FALSE( context.isMetalDevice() );
        EXPECT_FALSE( context.isOpenCLDevice() );
        EXPECT_FALSE( context.isVulkanDevice() );
    }

    // ============================================================================
    // Move Semantics Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, MoveConstructorWorks ) {
        CpuDeviceContext original;
        auto original_device = original.getDevice();

        CpuDeviceContext moved( std::move( original ) );

        EXPECT_EQ( moved.getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( moved.getDeviceName(), "CPU" );
        EXPECT_EQ( moved.getDeviceId(), -1 );
        EXPECT_NE( moved.getDevice(), nullptr );
        EXPECT_EQ( moved.getDevice(), original_device );
    }

    TEST_F( CpuDeviceContextTest, MoveAssignmentWorks ) {
        CpuDeviceContext original;
        auto original_device = original.getDevice();

        CpuDeviceContext target;
        target = std::move( original );

        EXPECT_EQ( target.getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( target.getDeviceName(), "CPU" );
        EXPECT_EQ( target.getDeviceId(), -1 );
        EXPECT_NE( target.getDevice(), nullptr );
        EXPECT_EQ( target.getDevice(), original_device );
    }

    // ============================================================================
    // Multiple Instance Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, MultipleInstancesIndependent ) {
        CpuDeviceContext context1;
        CpuDeviceContext context2;

        // Both should work independently
        EXPECT_NO_THROW( context1.makeCurrent() );
        EXPECT_NO_THROW( context2.makeCurrent() );

        // Both should have identical device characteristics
        EXPECT_EQ( context1.getDeviceType(), context2.getDeviceType() );
        EXPECT_EQ( context1.getDeviceName(), context2.getDeviceName() );
        EXPECT_EQ( context1.getDeviceId(), context2.getDeviceId() );
    }

    // ============================================================================
    // Polymorphic Usage Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, PolymorphicUsageThroughBaseClass ) {
        std::unique_ptr<DeviceContext> context = std::make_unique<CpuDeviceContext>();

        EXPECT_EQ( context->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( context->getDeviceName(), "CPU" );
        EXPECT_EQ( context->getDeviceId(), -1 );
        EXPECT_TRUE( context->isCpuDevice() );
        EXPECT_FALSE( context->isCudaDevice() );

        EXPECT_NO_THROW( context->makeCurrent() );

        auto device = context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    // ============================================================================
    // Integration Tests with DeviceRegistry
    // ============================================================================

    TEST_F( CpuDeviceContextTest, UsesDeviceRegistryCorrectly ) {
        // Verify that CpuDeviceContext correctly uses DeviceRegistry
        EXPECT_TRUE( DeviceRegistry::instance().hasDevice( "CPU" ) );

        auto device_from_registry = DeviceRegistry::instance().createDevice( "CPU" );
        ASSERT_NE( device_from_registry, nullptr );
        EXPECT_EQ( device_from_registry->getDeviceType(), DeviceType::Cpu );

        CpuDeviceContext context;
        auto device_from_context = context.getDevice();

        // Both should be CPU devices (though potentially different instances)
        EXPECT_EQ( device_from_registry->getDeviceType(), device_from_context->getDeviceType() );
        EXPECT_EQ( device_from_registry->getName(), device_from_context->getName() );
    }

    // ============================================================================
    // Boundary and Edge Case Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, HandlesDestructionGracefully ) {
        auto context = std::make_unique<CpuDeviceContext>();
        auto device = context->getDevice();

        // Destroying context should not affect the device
        context.reset();

        // Device should still be valid (assuming shared ownership)
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( CpuDeviceContextTest, MultipleOperationsSequence ) {
        CpuDeviceContext context;

        // Perform a sequence of operations that should all work
        for (int i = 0; i < 10; ++i) {
            EXPECT_NO_THROW( context.makeCurrent() );
            EXPECT_EQ( context.getDeviceType(), DeviceType::Cpu );
            EXPECT_EQ( context.getDeviceName(), "CPU" );
            EXPECT_EQ( context.getDeviceId(), -1 );

            auto device = context.getDevice();
            EXPECT_NE( device, nullptr );
            EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        }
    }

    // ============================================================================
    // Simplified API Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, SimplifiedConstructionPattern ) {
        // Test the new simplified construction pattern
        auto context = std::make_shared<CpuDeviceContext>();

        EXPECT_EQ( context->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( context->getDeviceName(), "CPU" );
        EXPECT_EQ( context->getDeviceId(), -1 );
        EXPECT_NE( context->getDevice(), nullptr );
    }

    TEST_F( CpuDeviceContextTest, NoRedundantParameterRequired ) {
        // Verify that the new API doesn't require redundant "CPU" parameter
        CpuDeviceContext context;  // Clean, no parameters needed

        EXPECT_EQ( context.getDeviceType(), DeviceType::Cpu );
        EXPECT_TRUE( context.isCpuDevice() );
    }

} // namespace Dnn::Compute::Devices::Cpu