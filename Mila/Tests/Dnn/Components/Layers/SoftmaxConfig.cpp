#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class SoftmaxConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( SoftmaxConfigTests, DefaultConstructor )
    {
        SoftmaxConfig cfg;
        EXPECT_EQ( cfg.getAxis(), -1 );
    }

    TEST_F( SoftmaxConfigTests, WithAxis_FluentInterface )
    {
        SoftmaxConfig cfg;
        auto&& ref = cfg.withAxis( 1 );

        EXPECT_EQ( &ref, &cfg );
        EXPECT_EQ( cfg.getAxis(), 1 );
    }

    TEST_F( SoftmaxConfigTests, RvalueFluentChaining_WithName )
    {
        // Fluent API must work on temporaries and preserve chaining
        auto cfg = SoftmaxConfig()
            .withAxis( 2 )
            .withName( "softmax_temp" );

        EXPECT_EQ( cfg.getAxis(), 2 );
        EXPECT_EQ( cfg.getName(), "softmax_temp" );
    }

    TEST_F( SoftmaxConfigTests, Validate_NoThrow )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 0 ).withName( "valid_softmax" );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( SoftmaxConfigTests, ToString_ContainsNameAndAxis )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 3 ).withName( "print_softmax" ).withPrecisionPolicy( ComputePrecision::Policy::Performance );

        std::string s = cfg.toString();
        EXPECT_NE( s.find( "print_softmax" ), std::string::npos );
        EXPECT_NE( s.find( "axis=" ), std::string::npos );
        EXPECT_NE( s.find( "3" ), std::string::npos ); // axis value present in string
    }

    TEST_F( SoftmaxConfigTests, ConfigurationPersistence_Copy )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 5 ).withName( "persist_softmax" );

        SoftmaxConfig copy = cfg;

        EXPECT_EQ( copy.getAxis(), 5 );
        EXPECT_EQ( copy.getName(), "persist_softmax" );
    }

    TEST_F( SoftmaxConfigTests, AllowNegativeAxis )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( -1 ).withName( "neg_axis" );

        EXPECT_EQ( cfg.getAxis(), -1 );
        EXPECT_EQ( cfg.getName(), "neg_axis" );
        EXPECT_NO_THROW( cfg.validate() );
    }

    // JSON serialization tests
    TEST_F( SoftmaxConfigTests, ToJson_IncludesNameOnly )
    {
        SoftmaxConfig cfg;
        cfg.withName( "softmax_json" ).withAxis( 7 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        auto j = cfg.toJson();

        // Current implementation only serializes "name"
        EXPECT_TRUE( j.contains( "name" ) );
        EXPECT_EQ( j.at( "name" ).get<std::string>(), "softmax_json" );

        // Axis and precision are not serialized in the current implementation
        EXPECT_FALSE( j.contains( "axis" ) );
        EXPECT_FALSE( j.contains( "precision" ) );
    }

    TEST_F( SoftmaxConfigTests, ToJson_ReflectsNameChange )
    {
        SoftmaxConfig cfg;
        cfg.withName( "first_name" );

        auto j1 = cfg.toJson();
        EXPECT_TRUE( j1.contains( "name" ) );
        EXPECT_EQ( j1.at( "name" ).get<std::string>(), "first_name" );

        cfg.withName( "second_name" );
        auto j2 = cfg.toJson();
        EXPECT_TRUE( j2.contains( "name" ) );
        EXPECT_EQ( j2.at( "name" ).get<std::string>(), "second_name" );
    }
}