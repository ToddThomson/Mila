/**
 * @file CudaDeviceContext.cpp
 * @brief Unit tests for CudaDeviceContext class.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>

import Compute.CudaDeviceContext;
import Compute.DeviceType;
import Compute.DeviceRegistry;
import Compute.CudaDevice;

using namespace Mila::Dnn::Compute;

namespace Dnn::Compute::Devices::Cuda::Tests {

    /**
     * @brief Test fixture for CudaDeviceContext tests.
     *
     * Sets up device registry with CUDA device registration before each test
     * and ensures clean state for testing. Skips tests if no CUDA devices are available.
     */
    class CudaDeviceContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Check if CUDA devices are available
            int device_count;
            cudaError_t error = cudaGetDeviceCount( &device_count );

            if (error != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "No CUDA devices available. Skipping CUDA tests.";
            }

            // Register CUDA devices if not already registered
            /*for (int i = 0; i < device_count; ++i) {
                std::string device_name = "CUDA:" + std::to_string( i );
                if (!DeviceRegistry::instance().hasDevice( device_name )) {
                    DeviceRegistry::instance().registerDevice( device_name, [i]() {
                        return std::make_shared<CudaDevice>( i );
                        } );
                }
            }*/

            // Store the first available device name for testing
            if (device_count > 0) {
                valid_device_name_ = "CUDA:0";
            }
        }

        void TearDown() override {
            // Reset CUDA device to ensure clean state
            cudaDeviceReset();
        }

        std::string valid_device_name_;
    };

    // ============================================================================
    // Constructor Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, ConstructorWithValidDeviceName ) {
        EXPECT_NO_THROW( {
            CudaDeviceContext context( valid_device_name_ );
            } );
    }

    TEST_F( CudaDeviceContextTest, ConstructorWithInvalidDeviceName ) {
        EXPECT_THROW( {
            CudaDeviceContext context( "INVALID_DEVICE" );
            }, std::runtime_error );
    }

    TEST_F( CudaDeviceContextTest, ConstructorThrowsOnEmptyDeviceName ) {
        EXPECT_THROW( {
            CudaDeviceContext context( "" );
            }, std::runtime_error );
    }

    TEST_F( CudaDeviceContextTest, ConstructorThrowsOnWrongDeviceType ) {
        EXPECT_THROW( {
            CudaDeviceContext context( "CPU" );
            }, std::runtime_error );
    }

    TEST_F( CudaDeviceContextTest, ConstructorThrowsOnNonExistentCudaDevice ) {
        EXPECT_THROW( {
            CudaDeviceContext context( "CUDA:999" );
            }, std::runtime_error );
    }

    TEST_F( CudaDeviceContextTest, ConstructorInitializesDeviceIdCorrectly ) {
        CudaDeviceContext context( "CUDA:0" );
        EXPECT_EQ( context.getDeviceId(), 0 );
    }

    // ============================================================================
    // DeviceContext Interface Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, GetDeviceTypeReturnsCuda ) {
        CudaDeviceContext context( valid_device_name_ );
        EXPECT_EQ( context.getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( CudaDeviceContextTest, GetDeviceNameReturnsCorrectName ) {
        CudaDeviceContext context( valid_device_name_ );
        EXPECT_EQ( context.getDeviceName(), valid_device_name_ );
    }

    TEST_F( CudaDeviceContextTest, GetDeviceIdReturnsCorrectId ) {
        CudaDeviceContext context( "CUDA:0" );
        EXPECT_EQ( context.getDeviceId(), 0 );

        // Test with another device if available
        int device_count;
        if (cudaGetDeviceCount( &device_count ) == cudaSuccess && device_count > 1) {
            CudaDeviceContext context1( "CUDA:1" );
            EXPECT_EQ( context1.getDeviceId(), 1 );
        }
    }

    TEST_F( CudaDeviceContextTest, GetDeviceReturnsValidCudaDevice ) {
        CudaDeviceContext context( valid_device_name_ );
        auto device = context.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( device->getName(), valid_device_name_ );

        // Verify it's actually a CudaDevice
        auto cuda_device = std::dynamic_pointer_cast<CudaDevice>(device);
        ASSERT_NE( cuda_device, nullptr );
        EXPECT_EQ( cuda_device->getDeviceId(), context.getDeviceId() );
    }

    // ============================================================================
    // CUDA-Specific Method Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, GetComputeCapabilityReturnsValidValues ) {
        CudaDeviceContext context( valid_device_name_ );
        auto [major, minor] = context.getComputeCapability();

        // Compute capability should be valid (at least 3.0 for modern CUDA)
        EXPECT_GE( major, 3 );
        EXPECT_GE( minor, 0 );
        EXPECT_LE( major, 10 ); // Reasonable upper bound for testing
    }

    TEST_F( CudaDeviceContextTest, GetComputeCapabilityMatchesDevice ) {
        CudaDeviceContext context( valid_device_name_ );
        auto [major, minor] = context.getComputeCapability();

        // Verify matches actual device properties
        cudaDeviceProp props;
        cudaGetDeviceProperties( &props, context.getDeviceId() );

        EXPECT_EQ( major, props.major );
        EXPECT_EQ( minor, props.minor );
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, ConstructorErrorMessageContainsDeviceName ) {
        const std::string invalid_name = "CUDA:999";

        try {
            CudaDeviceContext context( invalid_name );
            FAIL() << "Expected std::runtime_error to be thrown";
        }
        catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_NE( error_msg.find( "Invalid CUDA device name" ), std::string::npos );
        }
    }

    // ============================================================================
    // Resource Management Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, DestructorHandlesCleanupGracefully ) {
        std::shared_ptr<ComputeDevice> device;

        {
            CudaDeviceContext context( valid_device_name_ );
            device = context.getDevice();
            ASSERT_NE( device, nullptr );
        } // Context destroyed here

        // Device should still be valid due to shared ownership
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( device->getName(), valid_device_name_ );
    }

    // ============================================================================
    // Inherited Helper Method Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, IsDeviceTypeWorksCorrectly ) {
        CudaDeviceContext context( valid_device_name_ );

        EXPECT_TRUE( context.isDeviceType( DeviceType::Cuda ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Cpu ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Metal ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::OpenCL ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Vulkan ) );
    }

    TEST_F( CudaDeviceContextTest, IsCudaDeviceReturnsTrue ) {
        CudaDeviceContext context( valid_device_name_ );
        EXPECT_TRUE( context.isCudaDevice() );
    }

    TEST_F( CudaDeviceContextTest, IsOtherDeviceTypesReturnFalse ) {
        CudaDeviceContext context( valid_device_name_ );

        EXPECT_FALSE( context.isCpuDevice() );
        EXPECT_FALSE( context.isMetalDevice() );
        EXPECT_FALSE( context.isOpenCLDevice() );
        EXPECT_FALSE( context.isVulkanDevice() );
    }

    // ============================================================================
    // Polymorphic Usage Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, PolymorphicUsageThroughBaseClass ) {
        std::unique_ptr<DeviceContext> context =
            std::make_unique<CudaDeviceContext>( valid_device_name_ );

        EXPECT_EQ( context->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( context->getDeviceName(), valid_device_name_ );
        EXPECT_EQ( context->getDeviceId(), 0 );
        EXPECT_TRUE( context->isCudaDevice() );
        EXPECT_FALSE( context->isCpuDevice() );

        auto device = context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( CudaDeviceContextTest, PolymorphicMoveSemantics ) {
        auto original = std::make_unique<CudaDeviceContext>( valid_device_name_ );
        auto original_device = original->getDevice();
        int original_device_id = original->getDeviceId();

        std::unique_ptr<DeviceContext> moved = std::move( original );

        EXPECT_EQ( moved->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( moved->getDeviceId(), original_device_id );
        EXPECT_EQ( moved->getDevice(), original_device );
    }

    // ============================================================================
    // Integration Tests with DeviceRegistry
    // ============================================================================

    TEST_F( CudaDeviceContextTest, UsesDeviceRegistryCorrectly ) {
        EXPECT_TRUE( DeviceRegistry::instance().hasDevice( valid_device_name_ ) );

        auto device_from_registry = DeviceRegistry::instance().createDevice( valid_device_name_ );
        ASSERT_NE( device_from_registry, nullptr );
        EXPECT_EQ( device_from_registry->getDeviceType(), DeviceType::Cuda );

        CudaDeviceContext context( valid_device_name_ );
        auto device_from_context = context.getDevice();

        // Both should be CUDA devices (separate instances from registry)
        EXPECT_EQ( device_from_registry->getDeviceType(), device_from_context->getDeviceType() );
        EXPECT_EQ( device_from_registry->getName(), device_from_context->getName() );
    }

    // ============================================================================
    // Multi-GPU Tests (if multiple devices available)
    // ============================================================================

    TEST_F( CudaDeviceContextTest, MultiGpuContextsWorkIndependently ) {
        int device_count;
        cudaGetDeviceCount( &device_count );

        if (device_count < 2) {
            SUCCEED() << "Multi-GPU tests require at least 2 CUDA devices";
            return;
        }

        CudaDeviceContext context0( "CUDA:0" );
        CudaDeviceContext context1( "CUDA:1" );

        EXPECT_EQ( context0.getDeviceId(), 0 );
        EXPECT_EQ( context1.getDeviceId(), 1 );

        auto device0 = context0.getDevice();
        auto device1 = context1.getDevice();

        ASSERT_NE( device0, nullptr );
        ASSERT_NE( device1, nullptr );

        EXPECT_EQ( device0->getName(), "CUDA:0" );
        EXPECT_EQ( device1->getName(), "CUDA:1" );
    }

    TEST_F( CudaDeviceContextTest, MultiGpuComputeCapabilitiesDifferent ) {
        int device_count;
        cudaGetDeviceCount( &device_count );

        if (device_count < 2) {
            SUCCEED() << "Multi-GPU tests require at least 2 CUDA devices";
            return;
        }

        CudaDeviceContext context0( "CUDA:0" );
        CudaDeviceContext context1( "CUDA:1" );

        auto [major0, minor0] = context0.getComputeCapability();
        auto [major1, minor1] = context1.getComputeCapability();

        // Both should have valid compute capabilities
        EXPECT_GE( major0, 3 );
        EXPECT_GE( major1, 3 );

        // Note: They may or may not be the same depending on hardware
    }

    // ============================================================================
    // Edge Cases and Boundary Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, RepeatedPropertyQueriesStable ) {
        CudaDeviceContext context( valid_device_name_ );

        // Perform many repeated property queries to test stability
        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ( context.getDeviceType(), DeviceType::Cuda );
            EXPECT_EQ( context.getDeviceName(), valid_device_name_ );
            EXPECT_EQ( context.getDeviceId(), 0 );

            auto device = context.getDevice();
            ASSERT_NE( device, nullptr );

            auto [major, minor] = context.getComputeCapability();
            EXPECT_GE( major, 3 );
        }
    }

    TEST_F( CudaDeviceContextTest, MultipleInstancesSameDevice ) {
        CudaDeviceContext context1( valid_device_name_ );
        CudaDeviceContext context2( valid_device_name_ );

        // Both should have identical device characteristics
        EXPECT_EQ( context1.getDeviceType(), context2.getDeviceType() );
        EXPECT_EQ( context1.getDeviceName(), context2.getDeviceName() );
        EXPECT_EQ( context1.getDeviceId(), context2.getDeviceId() );

        auto cap1 = context1.getComputeCapability();
        auto cap2 = context2.getComputeCapability();
        EXPECT_EQ( cap1, cap2 );
    }

    // FIXME:
    /*TEST_F( CudaDeviceContextTest, MoveConstructorPreservesState ) {
        CudaDeviceContext original( valid_device_name_ );
        auto original_device = original.getDevice();
        int original_device_id = original.getDeviceId();
        auto original_cap = original.getComputeCapability();

        CudaDeviceContext moved( std::move( original ) );

        EXPECT_EQ( moved.getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( moved.getDeviceName(), valid_device_name_ );
        EXPECT_EQ( moved.getDeviceId(), original_device_id );
        EXPECT_EQ( moved.getDevice(), original_device );
        EXPECT_EQ( moved.getComputeCapability(), original_cap );
    }*/
    //FIXME:
    /*TEST_F( CudaDeviceContextTest, MoveAssignmentPreservesState ) {
        CudaDeviceContext original( valid_device_name_ );
        auto original_device = original.getDevice();
        int original_device_id = original.getDeviceId();

        CudaDeviceContext target( "CUDA:0" );
        target = std::move( original );

        EXPECT_EQ( target.getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( target.getDeviceId(), original_device_id );
        EXPECT_EQ( target.getDevice(), original_device );
    }*/

    // ============================================================================
    // Device Context Role Tests
    // ============================================================================

    TEST_F( CudaDeviceContextTest, ContextProvideDeviceIdentificationOnly ) {
        // CudaDeviceContext is now focused only on device identification
        // Device activation is handled by memory resources and execution contexts

        CudaDeviceContext context( valid_device_name_ );

        // Verify it provides necessary identification information
        EXPECT_EQ( context.getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( context.getDeviceId(), 0 );
        EXPECT_NE( context.getDevice(), nullptr );

        // Verify it provides CUDA-specific properties
        auto [major, minor] = context.getComputeCapability();
        EXPECT_GE( major, 3 );
        EXPECT_GE( minor, 0 );
    }

    TEST_F( CudaDeviceContextTest, LightweightContextCreation ) {
        // Context creation should be lightweight (no stream/handle creation)
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 10; ++i) {
            CudaDeviceContext context( valid_device_name_ );
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );

        // Context creation should be fast (no heavy CUDA resource allocation)
        EXPECT_LT( duration.count(), 1000 ); // Should take less than 1 second for 10 contexts
    }
}