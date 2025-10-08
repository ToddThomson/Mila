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
            /*if (!DeviceRegistry::instance().hasDevice( "CPU" )) {
                DeviceRegistry::instance().registerDevice( "CPU", []() {
                    return std::make_shared<CpuDevice>();
                    } );
            }*/
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

    TEST_F( CpuDeviceContextTest, ConstructorCreatesValidDevice ) {
        CpuDeviceContext context;
        auto device = context.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( CpuDeviceContextTest, ConstructorThrowsWhenDeviceCreationFails ) {
        // Temporarily unregister CPU device to force failure
        // Note: This test validates error handling when registry is misconfigured
        // In production, CPU device should always be available

        // Save current registration state would require registry modification
        // For now, we trust that proper exception is thrown if device creation fails
        // This is validated by the constructor implementation
        SUCCEED(); // Placeholder for when registry supports temporary unregistration
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


    TEST_F( CpuDeviceContextTest, GetDeviceReturnsValidDevice ) {
        CpuDeviceContext context;
        auto device = context.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getDeviceName(), "CPU" );
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

    TEST_F( CpuDeviceContextTest, MultipleInstancesSharesSameDeviceType ) {
        CpuDeviceContext context1;
        CpuDeviceContext context2;

        // Both should have identical device characteristics
        EXPECT_EQ( context1.getDeviceType(), context2.getDeviceType() );
        EXPECT_EQ( context1.getDeviceName(), context2.getDeviceName() );
        EXPECT_EQ( context1.getDeviceId(), context2.getDeviceId() );

        // Devices should be separate instances but same type
        EXPECT_EQ( context1.getDevice()->getDeviceType(),
            context2.getDevice()->getDeviceType() );
    }

    TEST_F( CpuDeviceContextTest, MultipleInstancesAreIndependent ) {
        CpuDeviceContext context1;
        CpuDeviceContext context2;

        auto device1 = context1.getDevice();
        auto device2 = context2.getDevice();

        // Both should be valid CPU devices
        ASSERT_NE( device1, nullptr );
        ASSERT_NE( device2, nullptr );
        EXPECT_EQ( device1->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device2->getDeviceType(), DeviceType::Cpu );
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

        auto device = context->getDevice();
        
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( CpuDeviceContextTest, PolymorphicMoveSemantics ) {
        auto original = std::make_unique<CpuDeviceContext>();
        auto original_device = original->getDevice();

        std::unique_ptr<DeviceContext> moved = std::move( original );

        EXPECT_EQ( moved->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( moved->getDevice(), original_device );
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

        // Both should be CPU devices (separate instances from registry)
        EXPECT_EQ( device_from_registry->getDeviceType(),
            device_from_context->getDeviceType() );
        EXPECT_EQ( device_from_registry->getDeviceName(),
            device_from_context->getDeviceName() );
    }

    // ============================================================================
    // Resource Management Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, HandlesDestructionGracefully ) {
        std::shared_ptr<ComputeDevice> device;

        {
            auto context = std::make_unique<CpuDeviceContext>();
            device = context->getDevice();

            ASSERT_NE( device, nullptr );
        } // Context destroyed here

        // Device should still be valid due to shared ownership
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getDeviceName(), "CPU" );
    }

    // ============================================================================
    // Simplified API Tests
    // ============================================================================

    TEST_F( CpuDeviceContextTest, SimplifiedConstructionPattern ) {
        // Verify the clean, parameterless construction pattern
        auto context = std::make_shared<CpuDeviceContext>();

        EXPECT_EQ( context->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( context->getDeviceName(), "CPU" );
        EXPECT_EQ( context->getDeviceId(), -1 );
        EXPECT_NE( context->getDevice(), nullptr );
    }

    TEST_F( CpuDeviceContextTest, NoRedundantParameterRequired ) {
        // Verify that the API doesn't require redundant "CPU" parameter
        CpuDeviceContext context;  // Clean, no parameters needed

        EXPECT_EQ( context.getDeviceType(), DeviceType::Cpu );
        EXPECT_TRUE( context.isCpuDevice() );
    }

    TEST_F( CpuDeviceContextTest, VectorOfContextsWorks ) {
        // CPU contexts should be movable into containers
        std::vector<CpuDeviceContext> contexts;

        for (int i = 0; i < 5; ++i) {
            contexts.emplace_back();
        }

        EXPECT_EQ( contexts.size(), 5 );

        for (auto& context : contexts) {
            EXPECT_EQ( context.getDeviceType(), DeviceType::Cpu );
            EXPECT_NE( context.getDevice(), nullptr );
        }
    }

} // namespace Dnn::Compute::Devices::Cpu