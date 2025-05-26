#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestComponentConfig : public ComponentConfig {
    public:
        // Add any test-specific methods here if needed
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

        EXPECT_EQ( config.getName(), "unnamed" );  // Default name is now "unnamed" instead of empty string
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Auto );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ComponentConfigTests, WithName_ShouldSetName ) {
        TestComponentConfig config;

        config.withName( test_module_name_ );

        EXPECT_EQ( config.getName(), test_module_name_ );
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
            .withPrecision( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( ComponentConfigTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestComponentConfig config;
        config.withName( test_module_name_ );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ComponentConfigTests, Validate_WithEmptyName_ShouldThrow ) {
        TestComponentConfig config;
        // Name is "unnamed" by default, so we need to set it to empty explicitly
        config.withName( "" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // The following tests need to be updated since we no longer have device or context methods in the base class
    // These might need to be implemented in a derived class for testing

    class DerivedComponentConfig : public ComponentConfig {
    public:
        DerivedComponentConfig& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        DerivedComponentConfig& withDeviceName( const std::string& device_name ) {
            device_name_ = device_name;
            return *this;
        }

        DerivedComponentConfig& withContext( std::shared_ptr<DeviceContext> context ) {
            context_ = context;
            return *this;
        }

        const std::string& getDeviceName() const { return device_name_; }
        std::shared_ptr<DeviceContext> getContext() const { return context_; }
        int getCustomOption() const { return custom_option_; }

        void validate() const override {
            ComponentConfig::validate();

            if ( custom_option_ < 0 ) {
                throw std::invalid_argument( "Custom option must be non-negative" );
            }

            // Add validation for device_name_ and context_ if needed
        }

    private:
        int custom_option_ = 0;
        std::string device_name_;
        std::shared_ptr<DeviceContext> context_;
    };

    TEST_F( ComponentConfigTests, DeducedThis_ShouldAllowMethodChaining ) {
        TestComponentConfig config;

        auto& ref1 = config.withName( test_module_name_ );
        auto& ref2 = ref1.withPrecision( ComputePrecision::Policy::Accuracy );
        auto& ref3 = ref2.withTraining( true );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );
        EXPECT_EQ( &ref3, &config );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecision(), ComputePrecision::Policy::Accuracy );
        EXPECT_TRUE( config.isTraining() );
    }

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
        config.withCustomOption( 42 )
            .withName( "" ); // Set name to empty to trigger base validation error

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }
}