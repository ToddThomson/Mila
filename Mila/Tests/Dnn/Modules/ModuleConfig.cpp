#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestComponentConfig : public ComponentConfig<TestComponentConfig> {
    public:
    };

    class ComponentConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_device_name_ = "CPU";
            cuda_device_name_ = "CUDA:0";
            test_module_name_ = "test_module";
        }

        void TearDown() override {}

        std::string cpu_device_name_;
        std::string cuda_device_name_;
        std::string test_module_name_;
    };

    TEST_F( ComponentConfigTests, DefaultConstructor_ShouldSetDefaultValues ) {
        TestComponentConfig config;

        EXPECT_EQ( config.getName(), "" );
        EXPECT_EQ( config.getDeviceName(), "" );
        EXPECT_EQ( config.getContext(), nullptr );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Auto );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ComponentConfigTests, WithName_ShouldSetName ) {
        TestComponentConfig config;

        config.withName( test_module_name_ );

        EXPECT_EQ( config.getName(), test_module_name_ );
    }

    TEST_F( ComponentConfigTests, WithDeviceName_ShouldSetDeviceNameAndClearContext ) {
        TestComponentConfig config;
        auto context = std::make_shared<DeviceContext>( cpu_device_name_ );
        config.withContext( context );
        EXPECT_NE( config.getContext(), nullptr );

        config.withDeviceName( cuda_device_name_ );

        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getContext(), nullptr );
    }

    TEST_F( ComponentConfigTests, WithContext_ShouldSetContextAndClearDeviceName ) {
        TestComponentConfig config;
        config.withDeviceName( cpu_device_name_ );
        EXPECT_EQ( config.getDeviceName(), cpu_device_name_ );

        auto context = std::make_shared<DeviceContext>( cuda_device_name_ );
        config.withContext( context );

        EXPECT_EQ( config.getContext(), context );
        EXPECT_EQ( config.getDeviceName(), "" );
    }

    TEST_F( ComponentConfigTests, WithPrecision_ShouldSetComputePrecision ) {
        TestComponentConfig config;

        config.withPrecision( ComputePrecision::Policy::Disabled );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Disabled );

        config.withPrecision( ComputePrecision::Policy::Auto );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Auto );

        config.withPrecision( ComputePrecision::Policy::Performance );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Performance );

        config.withPrecision( ComputePrecision::Policy::Accuracy );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ComponentConfigTests, Training_ShouldSetTrainingMode ) {
        TestComponentConfig config;

        config.withTraining( true );

        EXPECT_TRUE( config.isTraining() );

        config.withTraining( false );

        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ComponentConfigTests, MethodChaining_ShouldReturnCorrectValues ) {
        TestComponentConfig config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withPrecision( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( ComponentConfigTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestComponentConfig config;
        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ComponentConfigTests, Validate_WithEmptyName_ShouldThrow ) {
        TestComponentConfig config;
        config.withDeviceName( cuda_device_name_ );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( ComponentConfigTests, Validate_WithoutDeviceOrContext_ShouldThrow ) {
        TestComponentConfig config;
        config.withName( test_module_name_ );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( ComponentConfigTests, Validate_WithContext_ShouldNotThrow ) {
        TestComponentConfig config;
        auto context = std::make_shared<DeviceContext>( cpu_device_name_ );
        config.withName( test_module_name_ )
            .withContext( context );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ComponentConfigTests, CRTP_ShouldAllowComplexMethodChaining ) {
        TestComponentConfig config;

        auto& ref1 = config.withName( test_module_name_ );
        auto& ref2 = ref1.withDeviceName( cuda_device_name_ );
        auto& ref3 = ref2.withPrecision( ComputePrecision::Policy::Accuracy );
        auto& ref4 = ref3.withTraining( true );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );
        EXPECT_EQ( &ref3, &config );
        EXPECT_EQ( &ref4, &config );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Accuracy );
        EXPECT_TRUE( config.isTraining() );
    }

    class DerivedComponentConfig : public ComponentConfig<DerivedComponentConfig> {
    public:
        DerivedComponentConfig& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        int getCustomOption() const { return custom_option_; }

        void validate() const {
            ComponentConfig<DerivedComponentConfig>::validate();
            if ( custom_option_ < 0 ) {
                throw std::invalid_argument( "Custom option must be non-negative" );
            }
        }

    private:
        int custom_option_ = 0;
    };

    TEST_F( ComponentConfigTests, DerivedConfig_ShouldInheritAndExtendBaseConfig ) {
        DerivedComponentConfig config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( 42 )
            .withPrecision( ComputePrecision::Policy::Disabled )
            .withTraining( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Disabled );
        EXPECT_TRUE( config.isTraining() );
        EXPECT_EQ( config.getCustomOption(), 42 );
    }

    TEST_F( ComponentConfigTests, DerivedConfig_Validate_ShouldEnforceCustomRules ) {
        DerivedComponentConfig config;
        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( -1 );

        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withCustomOption( 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ComponentConfigTests, DerivedConfig_Validate_ShouldEnforceBaseRules ) {
        DerivedComponentConfig config;
        config.withCustomOption( 42 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }
}