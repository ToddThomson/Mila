#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Dnn::Compute::Tests
{
    using namespace Mila::Dnn::Compute;

    class DeviceContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Ensure the singleton instance is reset before each test
            DeviceContext::instance().setDevice( "CPU" );
        }
    };

    TEST_F( DeviceContextTest, SingletonInstance ) {
        DeviceContext& instance1 = DeviceContext::instance();
        DeviceContext& instance2 = DeviceContext::instance();
        EXPECT_EQ( &instance1, &instance2 );
    }

    TEST_F( DeviceContextTest, DefaultDeviceIsCPU ) {
        auto device = DeviceContext::instance().getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( DeviceContextTest, SetValidDevice ) {
        DeviceContext::instance().setDevice( "CPU" );
        auto device = DeviceContext::instance().getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( DeviceContextTest, SetInvalidDeviceThrows ) {
        EXPECT_THROW( DeviceContext::instance().setDevice( "InvalidDevice" ), std::runtime_error );
    }
}