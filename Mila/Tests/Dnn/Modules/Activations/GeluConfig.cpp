#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class GeluConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // GeluConfig uses default initialization
        }
    };

    TEST_F( GeluConfigTests, DefaultConstructor ) {
        GeluConfig config;
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Tanh ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );

        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
        EXPECT_EQ( &result, &config );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Exact ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Sigmoid ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, FluentInterfaceChaining ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, ValidationSuccess_TanhMethod ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, ValidationFailure_ExactMethod ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, ValidationFailure_SigmoidMethod ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, ValidationErrorMessage ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Only the Tanh approximation method is currently supported" ) != std::string::npos );
        }
    }

    TEST_F( GeluConfigTests, ConfigurationPersistence ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );

        GeluConfig copied_config = config;

        EXPECT_EQ( copied_config.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, EnumValues ) {
        EXPECT_EQ( static_cast<int>(GeluConfig::ApproximationMethod::Exact), 0 );
        EXPECT_EQ( static_cast<int>(GeluConfig::ApproximationMethod::Tanh), 1 );
        EXPECT_EQ( static_cast<int>(GeluConfig::ApproximationMethod::Sigmoid), 2 );
    }

    TEST_F( GeluConfigTests, DefaultStateValidation ) {
        GeluConfig config;

        EXPECT_NO_THROW( config.validate() );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, MethodChaining ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, MultipleMethodCalls ) {
        GeluConfig config;

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_EQ( config.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, ValidationAfterMethodChange ) {
        GeluConfig config;

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, RvalueFluentChainingAndName ) {
        // Ensure fluent API works on temporaries and preserves chaining
        auto cfg = GeluConfig()
            .withApproximationMethod( GeluConfig::ApproximationMethod::Exact )
            .withName( "gelu_temp" );

        EXPECT_EQ( cfg.getApproximationMethod(), GeluConfig::ApproximationMethod::Exact );
        EXPECT_EQ( cfg.getName(), "gelu_temp" );
    }

    TEST_F( GeluConfigTests, JsonSerializationRoundTrip ) {
        GeluConfig config;
        config.withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid )
              .withName( "gelu_json" );

        auto j = config.toJson();

        GeluConfig loaded;
        loaded.fromJson( j );

        EXPECT_EQ( loaded.getApproximationMethod(), GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_EQ( loaded.getName(), "gelu_json" );
    }
}