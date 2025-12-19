#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Dnn::Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

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
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    TEST_F( SoftmaxConfigTests, WithAxis_FluentInterface )
    {
        SoftmaxConfig cfg;
        auto&& ref = cfg.withAxis( 1 );

        EXPECT_EQ( &ref, &cfg );
        EXPECT_EQ( cfg.getAxis(), 1 );
    }

    TEST_F( SoftmaxConfigTests, WithPrecisionPolicy_FluentInterface )
    {
        SoftmaxConfig cfg;
        auto&& ref = cfg.withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( &ref, &cfg );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( SoftmaxConfigTests, RvalueFluentChaining )
    {
        auto cfg = SoftmaxConfig()
            .withAxis( 2 )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_EQ( cfg.getAxis(), 2 );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( SoftmaxConfigTests, Validate_NoThrow )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 0 );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( SoftmaxConfigTests, ToString_ContainsAxisAndPrecision )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 3 ).withPrecisionPolicy( ComputePrecision::Policy::Performance );

        std::string s = cfg.toString();

        EXPECT_NE( s.find( "axis=" ), std::string::npos );
        EXPECT_NE( s.find( "3" ), std::string::npos );
        EXPECT_NE( s.find( "precision=" ), std::string::npos );
    }

    TEST_F( SoftmaxConfigTests, ConfigurationPersistence_Copy )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 5 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        SoftmaxConfig copy = cfg;

        EXPECT_EQ( copy.getAxis(), 5 );
        EXPECT_EQ( copy.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( SoftmaxConfigTests, AllowNegativeAxis )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( -1 );

        EXPECT_EQ( cfg.getAxis(), -1 );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( SoftmaxConfigTests, AllowPositiveAxis )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 0 );

        EXPECT_EQ( cfg.getAxis(), 0 );
        EXPECT_NO_THROW( cfg.validate() );

        cfg.withAxis( 3 );
        EXPECT_EQ( cfg.getAxis(), 3 );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( SoftmaxConfigTests, ToSerializationMetadata_IncludesAllFields )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 7 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        auto meta = cfg.toSerializationMetadata();

        EXPECT_TRUE( meta.has( "axis" ) );
        EXPECT_EQ( meta.getInt( "axis" ), 7 );

        EXPECT_TRUE( meta.has( "precision" ) );
        EXPECT_EQ( meta.getInt( "precision" ), static_cast<int64_t>( ComputePrecision::Policy::Accuracy ) );
    }

    TEST_F( SoftmaxConfigTests, ToSerializationMetadata_DefaultValues )
    {
        SoftmaxConfig cfg;

        auto meta = cfg.toSerializationMetadata();

        EXPECT_TRUE( meta.has( "axis" ) );
        EXPECT_EQ( meta.getInt( "axis" ), -1 );

        EXPECT_TRUE( meta.has( "precision" ) );
        EXPECT_EQ( meta.getInt( "precision" ), static_cast<int64_t>( ComputePrecision::Policy::Auto ) );
    }

    TEST_F( SoftmaxConfigTests, FromSerializationMetadata_AllFields )
    {
        SerializationMetadata meta;
        meta.set( "axis", int64_t( 5 ) )
            .set( "precision", static_cast<int64_t>( ComputePrecision::Policy::Performance ) );

        SoftmaxConfig cfg;
        cfg.fromSerializationMetadata( meta );

        EXPECT_EQ( cfg.getAxis(), 5 );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( SoftmaxConfigTests, FromSerializationMetadata_PartialFields )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 10 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        SerializationMetadata meta;
        meta.set( "axis", int64_t( 3 ) );

        cfg.fromSerializationMetadata( meta );

        EXPECT_EQ( cfg.getAxis(), 3 );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( SoftmaxConfigTests, FromSerializationMetadata_EmptyMetadata )
    {
        SoftmaxConfig cfg;
        cfg.withAxis( 7 ).withPrecisionPolicy( ComputePrecision::Policy::Performance );

        SerializationMetadata meta;

        cfg.fromSerializationMetadata( meta );

        EXPECT_EQ( cfg.getAxis(), 7 );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( SoftmaxConfigTests, Serialization_RoundTrip )
    {
        SoftmaxConfig original;
        original.withAxis( 2 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        auto meta = original.toSerializationMetadata();

        SoftmaxConfig restored;
        restored.fromSerializationMetadata( meta );

        EXPECT_EQ( restored.getAxis(), original.getAxis() );
        EXPECT_EQ( restored.getPrecisionPolicy(), original.getPrecisionPolicy() );
    }

    TEST_F( SoftmaxConfigTests, Serialization_NegativeAxis )
    {
        SoftmaxConfig original;
        original.withAxis( -1 );

        auto meta = original.toSerializationMetadata();

        SoftmaxConfig restored;
        restored.fromSerializationMetadata( meta );

        EXPECT_EQ( restored.getAxis(), -1 );
    }

    TEST_F( SoftmaxConfigTests, MultipleConfigurations )
    {
        SoftmaxConfig cfg1;
        cfg1.withAxis( 0 ).withPrecisionPolicy( ComputePrecision::Policy::Auto );

        SoftmaxConfig cfg2;
        cfg2.withAxis( 1 ).withPrecisionPolicy( ComputePrecision::Policy::Performance );

        SoftmaxConfig cfg3;
        cfg3.withAxis( 2 ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_EQ( cfg1.getAxis(), 0 );
        EXPECT_EQ( cfg1.getPrecisionPolicy(), ComputePrecision::Policy::Auto );

        EXPECT_EQ( cfg2.getAxis(), 1 );
        EXPECT_EQ( cfg2.getPrecisionPolicy(), ComputePrecision::Policy::Performance );

        EXPECT_EQ( cfg3.getAxis(), 2 );
        EXPECT_EQ( cfg3.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }
}