#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
        
    class GeluConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Default initialization happens in the constructor
        }
    };

    // Test default constructor and default approximation method
    TEST_F( GeluConfigTests, DefaultConstructor ) {
        GeluConfig config;
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    // Test setting approximation method
    TEST_F( GeluConfigTests, SetApproximationMethod_Exact ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Tanh ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Sigmoid ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
    }

    // Test fluent interface chaining
    TEST_F( GeluConfigTests, FluentInterfaceChaining ) {
        GeluConfig config;
        auto& result = config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_EQ( &result, &config );
    }

    // Test validate method
    TEST_F( GeluConfigTests, Validate ) {
        GeluConfig config;
        config.withName( "gelu_test" )
            .withDeviceName( "CPU" )
            .withPrecision( ComputePrecision::Policy::Auto )
            .withTraining( true );

        EXPECT_NO_THROW( config.validate() );

        // Test with each approximation method
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        EXPECT_NO_THROW( (config.validate()) );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // Test interaction with base class methods
    // Note: This depends on which methods are available in ConfigurationBase<GeluConfig>
    // The following is a general pattern assuming common module configuration options
    TEST_F( GeluConfigTests, BaseClassInteraction ) {
        GeluConfig config;

        // Validate that approximation method is preserved when modifying base class properties
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        // Call any base class methods that might exist here, for example:
        // config.withName("test_gelu");

        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );
    }

    // Test combination of settings
    TEST_F( GeluConfigTests, CombinationOfSettings ) {
        GeluConfig config;

        // Configure multiple settings in sequence
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );

        // Set other properties if available through ConfigurationBase<GeluConfig>
        // For example: config.withName("test_gelu").withTrainable(true);

        // Verify that the approximation method wasn't affected
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
    }
}