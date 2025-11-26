#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <sstream>

import Mila;

namespace Dnn::Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TestModuleConfig : public ComponentConfig {
    public:
        // Test-only derived class; no additional behavior required.

        TestModuleConfig() 
            : ComponentConfig( "test_component" ) {
		}

        void validate() const override {
            // Basic validation: name must not be empty
            if (getName().empty()) {
                throw std::invalid_argument("ModuleConfig: name must not be empty");
            }
		}

        std::string toString() const {
            std::ostringstream oss;
            oss << "ModuleConfig(name=" << getName()
                << ", precision_policy=" << static_cast<int>(getPrecisionPolicy())
                << ")";
            return oss.str();
		}
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

        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    TEST_F( ModuleConfigTests, WithName_ShouldSetName ) {
        TestModuleConfig config;

        config.withName( test_module_name_ );

        EXPECT_EQ( config.getName(), test_module_name_ );
    }

    TEST_F( ModuleConfigTests, WithPrecision_ShouldSetComputePrecision ) {
        TestModuleConfig config;

        config.withPrecisionPolicy( ComputePrecision::Policy::Native );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );

        config.withPrecisionPolicy( ComputePrecision::Policy::Auto );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );

        config.withPrecisionPolicy( ComputePrecision::Policy::Performance );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );

        config.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ModuleConfigTests, MethodChaining_ShouldReturnCorrectValues ) {
        TestModuleConfig config;

        config.withName( test_module_name_ )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( ModuleConfigTests, Validate_WithValidConfig_ShouldNotThrow ) {
        TestModuleConfig config;
        config.withName( test_module_name_ );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ModuleConfigTests, Validate_WithEmptyName_ShouldThrow ) {
        TestModuleConfig config;
        // Default name is "unnamed"; explicitly set empty to test validation failure
        config.withName( "" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // Derived test configuration to exercise extension and custom validation
    class DerivedModuleConfig : public ComponentConfig {
    public:

        DerivedModuleConfig()
            : ComponentConfig( "derived_component" )
        {
        }

        DerivedModuleConfig& withCustomOption( int value ) {
            custom_option_ = value;
            return *this;
        }

        DerivedModuleConfig& withDeviceName( const std::string& device_name ) {
            device_name_ = device_name;
            return *this;
        }

        const std::string& getDeviceName() const { return device_name_; }
        int getCustomOption() const { return custom_option_; }

        void validate() const override {
            // Enforce base validation first
            ComponentConfig::validate();

            if (custom_option_ < 0)
            {
                throw std::invalid_argument( "Custom option must be non-negative" );
            }
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "DerivedModuleConfig(name=" << getName()
                << ", precision_policy=" << static_cast<int>(getPrecisionPolicy())
                << ", device_name=" << device_name_
                << ", custom_option=" << custom_option_
                << ")";
            return oss.str();
		}

    private:
        int custom_option_ = 0;
        std::string device_name_;
    };

    TEST_F( ModuleConfigTests, DeducedThis_ShouldAllowMethodChaining ) {
        TestModuleConfig config;

        auto& ref1 = config.withName( test_module_name_ );
        auto& ref2 = ref1.withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_EQ( &ref1, &config );
        EXPECT_EQ( &ref2, &config );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( ModuleConfigTests, DerivedConfig_ShouldInheritAndExtendBaseConfig ) {
        DerivedModuleConfig config;

        config.withName( test_module_name_ )
            .withDeviceName( cuda_device_name_ )
            .withCustomOption( 42 )
            .withPrecisionPolicy( ComputePrecision::Policy::Native );

        EXPECT_EQ( config.getName(), test_module_name_ );
        EXPECT_EQ( config.getDeviceName(), cuda_device_name_ );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Native );
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
        config.withCustomOption( 42 )
            .withName( "" ); // Set name empty to trigger base validation

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }
}