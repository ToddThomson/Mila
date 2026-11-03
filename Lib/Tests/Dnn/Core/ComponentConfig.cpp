/**
 * @file ComponentConfig.cpp
 * @brief Base-contract tests for the ComponentConfig interface.
 *
 * ComponentConfig is a pure interface: toMetadata / fromMetadata / validate /
 * toString. There is no concrete base behavior (precision policy moved to
 * BuildContext/ModelConfig and is no longer a config concern), so its "100%"
 * is a pattern-conformance test: a derived mock round-trips its metadata,
 * validates, summarizes, and destructs polymorphically through a base pointer.
 *
 * The mock carries its own field to exercise the metadata round-trip mechanism
 * independently of any removed base concept.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    namespace
    {
        // A concrete ComponentConfig with a single self-owned field, used to
        // exercise the interface contract. Internal linkage avoids collisions
        // with other test translation units.
        class MockConfig : public ComponentConfig
        {
        public:
            int value_ = 0;
            bool valid_ = true;

            SerializationMetadata toMetadata() const override
            {
                SerializationMetadata meta;
                meta.set( "value", static_cast<int64_t>( value_ ) );

                return meta;
            }

            void fromMetadata( const SerializationMetadata& meta ) override
            {
                if ( auto v = meta.tryGetInt( "value" ) )
                {
                    value_ = static_cast<int>( *v );
                }
            }

            void validate() const override
            {
                if ( !valid_ )
                {
                    throw std::invalid_argument( "MockConfig: invalid" );
                }
            }

            std::string toString() const override
            {
                return "MockConfig{value=" + std::to_string( value_ ) + "}";
            }
        };
    }

    class ComponentConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( ComponentConfigTests, Validate_PassesWhenValid )
    {
        MockConfig config;

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( ComponentConfigTests, Validate_ThrowsWhenInvalid )
    {
        MockConfig config;
        config.valid_ = false;

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( ComponentConfigTests, Metadata_RoundTripPreservesValue )
    {
        MockConfig source;
        source.value_ = 42;

        SerializationMetadata meta = source.toMetadata();

        MockConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.value_, 42 );
    }

    TEST_F( ComponentConfigTests, FromMetadata_IgnoresMissingKeys )
    {
        MockConfig config;
        config.value_ = 7;

        // Empty metadata must leave existing values untouched (the documented
        // forward/backward-compatibility contract).
        SerializationMetadata empty;
        config.fromMetadata( empty );

        EXPECT_EQ( config.value_, 7 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( ComponentConfigTests, ToString_NonEmpty )
    {
        MockConfig config;

        EXPECT_FALSE( config.toString().empty() );
    }

    // ====================================================================
    // Polymorphic contract
    // ====================================================================

    TEST_F( ComponentConfigTests, DestroysPolymorphicallyThroughBasePointer )
    {
        std::unique_ptr<ComponentConfig> config = std::make_unique<MockConfig>();

        EXPECT_NO_THROW( config->validate() );
        EXPECT_NO_THROW( config.reset() );
    }
}
