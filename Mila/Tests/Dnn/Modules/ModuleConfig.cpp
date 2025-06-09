#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Modules::Base::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestConfigurationBase : public ConfigurationBase {
    public:
        // Add any test-specific methods here if needed
    };

    class ConfigurationBaseTests : public ::testing::Test {
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

    TEST_F( ConfigurationBaseTests, DefaultConstructor_ShouldSetDefaultValues ) {
        TestConfigurationBase config;

        EXPECT_EQ( config.getName(), "unnamed" );  // Default name is now "unnamed" instead of empty string
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ConfigurationBaseTests, WithName_ShouldSetName ) {
        TestConfigurationBase config;

        config.withName( test_module_name_ );

        EXPECT_EQ( config.getName(), test_module_name_ );
    }

    TEST_F( ConfigurationBaseTests, WithPrecision_ShouldSetComputePrecision ) {
        TestConfigurationBase config;

        config.withPrecisionPolicy( ComputePrecision::Policy::Native );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );

        config.withPrecisionPolicy( ComputePrecision::Policy::Auto );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );

        config.withPrecisionPolicy( ComputePrecision::Policy::Performance );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );

        config.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ConfigurationBaseTests, Training_ShouldSetTrainingMode ) {
        TestConfigurationBase config;

        config.withTraining( true );
        EXPECT_TRUE( config.isTraining() );

        config.withTraining( false );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ConfigurationBaseTests, MethodChaining_ShouldReturnCorrectValues ) {
        TestConfigurationBase config;

        config.withName( test_module_name_ )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( ConfigurationBaseTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestConfigurationBase config;
        config.withName( test_module_name_ );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ConfigurationBaseTests, Validate_WithEmptyName_ShouldThrow ) {
        TestConfigurationBase config;
        // Name is "unnamed" by default, so we need to set it to empty explicitly
        config.withName( "" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // The following tests need to be updated since we no longer have device or context methods in the base class
    // These might need to be implemented in a derived class for testing

    class DerivedConfigurationBase : public ConfigurationBase {
    public:
        DerivedConfigurationBase& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        DerivedConfigurationBase& withDeviceName( const std::string& device_name ) {
            device_name_ = device_name;
            return *this;
        }

        DerivedConfigurationBase& withContext( std::shared_ptr<DeviceContext> context ) {
            context_ = context;
            return *this;
        }

        const std::string& getDeviceName() const { return device_name_; }
        std::shared_ptr<DeviceContext> getContext() const { return context_; }
        int getCustomOption() const { return custom_option_; }

        void validate() const override {
            ConfigurationBase::validate();

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

    TEST_F( ConfigurationBaseTests, DeducedThis_ShouldAllowMethodChaining ) {
        TestConfigurationBase config;

        auto& ref1 = config.withName( test_module_name_ );
        auto& ref2 = ref1.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );
        auto& ref3 = ref2.withTraining( true );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );
        EXPECT_EQ( &ref3, &config );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( ConfigurationBaseTests, DerivedConfig_ShouldInheritAndExtendBaseConfig ) {
        DerivedConfigurationBase config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( 42 )
            .withPrecisionPolicy( ComputePrecision::Policy::Native )
            .withTraining( true );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );
        EXPECT_TRUE( config.isTraining() );
        EXPECT_EQ( config.getCustomOption(), 42 );
    }

    TEST_F( ConfigurationBaseTests, DerivedConfig_Validate_ShouldEnforceCustomRules ) {
        DerivedConfigurationBase config;
        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( -1 );

        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withCustomOption( 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ConfigurationBaseTests, DerivedConfig_Validate_ShouldEnforceBaseRules ) {
        DerivedConfigurationBase config;
        config.withCustomOption( 42 )
            .withName( "" ); // Set name to empty to trigger base validation error

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }
}