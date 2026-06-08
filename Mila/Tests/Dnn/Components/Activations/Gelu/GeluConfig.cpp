#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Dnn::Components::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class GeluConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // GeluConfig uses default initialization
        }
    };

    TEST_F( GeluConfigTests, DefaultConstructor ) {
        GeluConfig config;
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Tanh ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( ApproximationMethod::Tanh );

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
        EXPECT_EQ( &result, &config );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Exact ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, SetApproximationMethod_Sigmoid ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Sigmoid );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, FluentInterfaceChaining ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, ValidationSuccess_TanhMethod ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Tanh );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, ValidationFailure_ExactMethod ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, ValidationFailure_SigmoidMethod ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Sigmoid );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, ValidationErrorMessage ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "only the Tanh approximation method is currently supported" ) != std::string::npos );
        }
    }

    TEST_F( GeluConfigTests, ConfigurationPersistence ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Sigmoid );

        GeluConfig copied_config = config;

        EXPECT_EQ( copied_config.getApproximationMethod(), ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, EnumValues ) {
        EXPECT_EQ( static_cast<int>(ApproximationMethod::Exact), 0 );
        EXPECT_EQ( static_cast<int>(ApproximationMethod::Tanh), 1 );
        EXPECT_EQ( static_cast<int>(ApproximationMethod::Sigmoid), 2 );
    }

    TEST_F( GeluConfigTests, DefaultStateValidation ) {
        GeluConfig config;

        EXPECT_NO_THROW( config.validate() );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, MethodChaining ) {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( ApproximationMethod::Tanh );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, MultipleMethodCalls ) {
        GeluConfig config;

        config.withApproximationMethod( ApproximationMethod::Exact );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );

        config.withApproximationMethod( ApproximationMethod::Tanh );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );

        config.withApproximationMethod( ApproximationMethod::Sigmoid );
        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, ValidationAfterMethodChange ) {
        GeluConfig config;

        config.withApproximationMethod( ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );

        config.withApproximationMethod( ApproximationMethod::Exact );
        EXPECT_THROW( config.validate(), std::invalid_argument );

        config.withApproximationMethod( ApproximationMethod::Tanh );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, RvalueFluentChaining ) {
        // Ensure fluent API works on temporaries and preserves chaining for approximation method
        auto cfg = GeluConfig()
            .withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_EQ( cfg.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, MetadataSerializationRoundTrip ) {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Sigmoid );

        SerializationMetadata meta = config.toMetadata();

        GeluConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getApproximationMethod(), ApproximationMethod::Sigmoid );
    }
}