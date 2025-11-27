#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>

import Mila;

namespace Modules::Connections::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class ResidualConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( ResidualConfigTests, DefaultConstructor_ShouldSetDefaults )
    {
        ResidualConfig cfg;

        // Defaults from ResidualConfig.ixx
        EXPECT_FLOAT_EQ( cfg.getScalingFactor(), 1.0f );
        EXPECT_EQ( cfg.getConnectionType(), ConnectionType::Addition );

        EXPECT_EQ( cfg.getName(), "residual" );
    }

    TEST_F( ResidualConfigTests, WithConnectionTypeAndScaling_Setters )
    {
        ResidualConfig cfg;

        auto& ref = cfg.withConnectionType( ConnectionType::Addition )
            .withScalingFactor( 0.75f )
            .withName( "residual_test" );

        EXPECT_EQ( &ref, &cfg );
        EXPECT_EQ( cfg.getConnectionType(), ConnectionType::Addition );
        EXPECT_FLOAT_EQ( cfg.getScalingFactor(), 0.75f );
        EXPECT_EQ( cfg.getName(), "residual_test" );
    }

    TEST_F( ResidualConfigTests, FluentInterfaceChaining )
    {
        ResidualConfig cfg;

        auto& result = cfg
            .withScalingFactor( 0.5f )
            .withConnectionType( ConnectionType::Addition )
            .withName( "chained_residual" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( &result, &cfg );
        EXPECT_FLOAT_EQ( cfg.getScalingFactor(), 0.5f );
        EXPECT_EQ( cfg.getConnectionType(), ConnectionType::Addition );
        EXPECT_EQ( cfg.getName(), "chained_residual" );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( ResidualConfigTests, ValidationSuccess )
    {
        ResidualConfig cfg;
        cfg.withName( "valid_residual" )
            .withScalingFactor( 1.25f );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( ResidualConfigTests, ValidationFailure_NonPositiveScaling_Zero )
    {
        ResidualConfig cfg;
        cfg.withName( "bad_scaling_zero" )
            .withScalingFactor( 0.0f );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( ResidualConfigTests, ValidationFailure_NonPositiveScaling_Negative )
    {
        ResidualConfig cfg;
        cfg.withName( "bad_scaling_neg" )
            .withScalingFactor( -0.5f );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( ResidualConfigTests, ValidationErrorMessage_Scaling )
    {
        ResidualConfig cfg;
        cfg.withName( "err_msg" )
            .withScalingFactor( 0.0f );

        try
        {
            cfg.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch (const std::invalid_argument& e)
        {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "scaling_factor" ), std::string::npos );
        }
    }

    TEST_F( ResidualConfigTests, ConfigurationPersistence_Copy )
    {
        ResidualConfig cfg;
        cfg.withName( "persistent_residual" )
            .withScalingFactor( 0.9f )
            .withConnectionType( ConnectionType::Addition );

        ResidualConfig copy = cfg;

        EXPECT_FLOAT_EQ( copy.getScalingFactor(), 0.9f );
        EXPECT_EQ( copy.getConnectionType(), ConnectionType::Addition );
        EXPECT_EQ( copy.getName(), "persistent_residual" );
    }

    TEST_F( ResidualConfigTests, ToStringContainsFields )
    {
        ResidualConfig cfg;
        cfg.withName( "print_residual" )
            .withScalingFactor( 0.125f )
            .withConnectionType( ConnectionType::Addition );

        std::string s = cfg.toString();

        EXPECT_NE( s.find( "print_residual" ), std::string::npos );
        EXPECT_NE( s.find( "scaling factor" ), std::string::npos );
        EXPECT_NE( s.find( "Addition" ), std::string::npos );
    }

    TEST_F( ResidualConfigTests, MethodChainPreservation )
    {
        ResidualConfig cfg;
        cfg.withScalingFactor( 0.6f )
            .withName( "chain_preserve" );

        // Update scaling and ensure name and other values persist
        cfg.withScalingFactor( 1.2f );

        EXPECT_FLOAT_EQ( cfg.getScalingFactor(), 1.2f );
        EXPECT_EQ( cfg.getName(), "chain_preserve" );
        EXPECT_EQ( cfg.getConnectionType(), ConnectionType::Addition );
    }
}