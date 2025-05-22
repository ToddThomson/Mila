#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestModuleConfig : public ModuleConfig<TestModuleConfig> {
    public:
    };

    class ModuleConfigTests : public ::testing::Test {
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

    TEST_F( ModuleConfigTests, DefaultConstructor_ShouldSetDefaultValues ) {
        TestModuleConfig config;

        EXPECT_EQ( config.getName(), "" );
        EXPECT_EQ( config.getDeviceName(), "" );
        EXPECT_EQ( config.getContext(), nullptr );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Auto );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ModuleConfigTests, WithName_ShouldSetName ) {
        TestModuleConfig config;

        config.withName( test_module_name_ );

        EXPECT_EQ( config.getName(), test_module_name_ );
    }

    TEST_F( ModuleConfigTests, WithDeviceName_ShouldSetDeviceNameAndClearContext ) {
        TestModuleConfig config;
        auto context = std::make_shared<DeviceContext>( cpu_device_name_ );
        config.withContext( context );
        EXPECT_NE( config.getContext(), nullptr );

        config.withDeviceName( cuda_device_name_ );

        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getContext(), nullptr );
    }

    TEST_F( ModuleConfigTests, WithContext_ShouldSetContextAndClearDeviceName ) {
        TestModuleConfig config;
        config.withDeviceName( cpu_device_name_ );
        EXPECT_EQ( config.getDeviceName(), cpu_device_name_ );

        auto context = std::make_shared<DeviceContext>( cuda_device_name_ );
        config.withContext( context );

        EXPECT_EQ( config.getContext(), context );
        EXPECT_EQ( config.getDeviceName(), "" );
    }

    TEST_F( ModuleConfigTests, WithPrecision_ShouldSetComputePrecision ) {
        TestModuleConfig config;

        config.withPrecision( ComputePrecision::Policy::Disabled );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Disabled );

        config.withPrecision( ComputePrecision::Policy::Auto );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Auto );

        config.withPrecision( ComputePrecision::Policy::Performance );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Performance );

        config.withPrecision( ComputePrecision::Policy::Accuracy );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ModuleConfigTests, Training_ShouldSetTrainingMode ) {
        TestModuleConfig config;

        config.training( true );

        EXPECT_TRUE( config.isTraining() );

        config.training( false );

        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ModuleConfigTests, MethodChaining_ShouldReturnCorrectValues ) {
        TestModuleConfig config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withPrecision( ComputePrecision::Policy::Performance )
            .training( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( ModuleConfigTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestModuleConfig config;
        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ModuleConfigTests, Validate_WithEmptyName_ShouldThrow ) {
        TestModuleConfig config;
        config.withDeviceName( cuda_device_name_ );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( ModuleConfigTests, Validate_WithoutDeviceOrContext_ShouldThrow ) {
        TestModuleConfig config;
        config.withName( test_module_name_ );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( ModuleConfigTests, Validate_WithContext_ShouldNotThrow ) {
        TestModuleConfig config;
        auto context = std::make_shared<DeviceContext>( cpu_device_name_ );
        config.withName( test_module_name_ )
            .withContext( context );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ModuleConfigTests, CRTP_ShouldAllowComplexMethodChaining ) {
        TestModuleConfig config;

        auto& ref1 = config.withName( test_module_name_ );
        auto& ref2 = ref1.withDeviceName( cuda_device_name_ );
        auto& ref3 = ref2.withPrecision( ComputePrecision::Policy::Accuracy );
        auto& ref4 = ref3.training( true );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );
        EXPECT_EQ( &ref3, &config );
        EXPECT_EQ( &ref4, &config );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Accuracy );
        EXPECT_TRUE( config.isTraining() );
    }

    class DerivedModuleConfig : public ModuleConfig<DerivedModuleConfig> {
    public:
        DerivedModuleConfig& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        int getCustomOption() const { return custom_option_; }

        void validate() const {
            ModuleConfig<DerivedModuleConfig>::validate();
            if ( custom_option_ < 0 ) {
                throw std::invalid_argument( "Custom option must be non-negative" );
            }
        }

    private:
        int custom_option_ = 0;
    };

    TEST_F( ModuleConfigTests, DerivedConfig_ShouldInheritAndExtendBaseConfig ) {
        DerivedModuleConfig config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( 42 )
            .withPrecision( ComputePrecision::Policy::Disabled )
            .training( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Disabled );
        EXPECT_TRUE( config.isTraining() );
        EXPECT_EQ( config.getCustomOption(), 42 );
    }

    TEST_F( ModuleConfigTests, DerivedConfig_Validate_ShouldEnforceCustomRules ) {
        DerivedModuleConfig config;
        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( -1 );

        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withCustomOption( 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ModuleConfigTests, DerivedConfig_Validate_ShouldEnforceBaseRules ) {
        DerivedModuleConfig config;
        config.withCustomOption( 42 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }
}