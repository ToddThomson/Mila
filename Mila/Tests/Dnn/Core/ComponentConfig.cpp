#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <sstream>

import Mila;

namespace Dnn::Components::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestComponentConfig : public ComponentConfig {
    public:
        // Test-only derived class; no additional behavior required.
        TestComponentConfig() = default;

        void validate() const override {
            // No additional validation for test config
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << "TestComponentConfig( precision_policy=" << static_cast<int>(getPrecisionPolicy())
                << ")";
            return oss.str();
        }
    };

    class ComponentConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_device_name_ = "CPU";
            cuda_device_name_ = "CUDA:0";
        }

        void TearDown() override {}

        std::string cpu_device_name_;
        std::string cuda_device_name_;
    };

    TEST_F( ComponentConfigTests, DefaultConstructor_ShouldSetDefaultValues ) {
        TestComponentConfig config;

        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    TEST_F( ComponentConfigTests, WithPrecision_ShouldSetComputePrecision ) {
        TestComponentConfig config;

        config.withPrecisionPolicy( ComputePrecision::Policy::Native );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );

        config.withPrecisionPolicy( ComputePrecision::Policy::Auto );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );

        config.withPrecisionPolicy( ComputePrecision::Policy::Performance );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );

        config.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ComponentConfigTests, MethodChaining_ShouldReturnCorrectValues ) {
        TestComponentConfig config;

        // Chain precision setters to verify fluent interface returns same object
        auto& ref1 = config.withPrecisionPolicy( ComputePrecision::Policy::Performance );
        auto& ref2 = ref1.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );

        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ComponentConfigTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestComponentConfig config;

        EXPECT_NO_THROW( config.validate() );
    }

    // Derived test configuration to exercise extension and custom validation
    class DerivedModuleConfig : public ComponentConfig {
    public:

        DerivedModuleConfig() = default;

        DerivedModuleConfig& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        DerivedModuleConfig& withDeviceName( const std::string& device_name ) {
            device_name_ = device_name;
            return *this;
        }

        const std::string& getDeviceName() const {
            return device_name_;
        }
        int getCustomOption() const {
            return custom_option_;
        }

        void validate() const override {
            // Enforce base validation first (currently no-op) then custom rules
            ComponentConfig::validate();

            if ( custom_option_ < 0 )
            {
                throw std::invalid_argument( "Custom option must be non-negative" );
            }
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "DerivedModuleConfig( precision_policy=" << static_cast<int>( getPrecisionPolicy() )
                << ", device_name=" << device_name_
                << ", custom_option=" << custom_option_
                << ")";
            return oss.str();
        }

    private:
        int custom_option_ = 0;
        std::string device_name_;
    };

    TEST_F( ComponentConfigTests, DeducedThis_ShouldAllowMethodChaining ) {
        TestComponentConfig config;

        auto& ref1 = config.withPrecisionPolicy( ComputePrecision::Policy::Performance );
        auto& ref2 = ref1.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );

        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ComponentConfigTests, DerivedConfig_ShouldInheritAndExtendBaseConfig ) {
        DerivedModuleConfig config;

        config
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( 42 )
            .withPrecisionPolicy( ComputePrecision::Policy::Native );

        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );
        EXPECT_EQ( config.getCustomOption(), 42 );
    }

    TEST_F( ComponentConfigTests, DerivedConfig_Validate_ShouldEnforceCustomRules ) {
        DerivedModuleConfig config;
        config.withDeviceName( cuda_device_name_ )
            .withCustomOption( -1 );

        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withCustomOption( 1 );
        EXPECT_NO_THROW( config.validate() );
    }
}