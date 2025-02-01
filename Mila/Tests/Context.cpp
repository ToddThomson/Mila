#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Mila::Context2::Tests
{
    using namespace Mila::Dnn;

    class ContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Ensure the singleton instance is reset before each test
            Compute::DeviceContext::instance().setDevice( "CPU" );
        }
    };

    /*TEST_F( ContextTest, SingletonInstance ) {
        Dnn::ModelContext& instance1 = Dnn::ModelContext::instance();
        Context& instance2 = Context::instance();
        EXPECT_EQ( &instance1, &instance2 );
    }

    TEST_F( ContextTest, DefaultDeviceIsCPU ) {
        auto device = Context::instance().device();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->name(), "CPU" );
    }

    TEST_F( ContextTest, SetValidDevice ) {
        Context::instance().setDevice( "CPU" );
        auto device = Context::instance().device();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->name(), "CPU" );
    }

    TEST_F( ContextTest, SetInvalidDeviceThrows ) {
        EXPECT_THROW( Context::instance().setDevice( "InvalidDevice" ), std::runtime_error );
    }*/
}