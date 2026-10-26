/**
 * @file ArgumentSchema.cpp
 * @brief A call's arguments are checked against the tool's parameters before anything runs, and a schema with a
 *        keyword the check does not make is refused (ToolCalling.md 3.2).
 */

#include <gtest/gtest.h>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::AI
{
    using Mila::AI::ArgumentSchema;

    namespace
    {
        constexpr const char* kWeatherParameters =
            R"({"type":"object","properties":{"location":{"type":"string","description":"City and country"},)"
            R"("units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]})";

        std::string mismatch( const ArgumentSchema& schema, const char* arguments )
        {
            return schema.check( arguments ).value_or( "" );
        }
    }

    TEST( ArgumentSchema, ArgumentsThatMatchPass )
    {
        const ArgumentSchema schema( kWeatherParameters );

        EXPECT_FALSE( schema.check( R"({"location":"Paris, France"})" ) );
        EXPECT_FALSE( schema.check( R"({"location":"Paris, France","units":"celsius"})" ) );
    }

    TEST( ArgumentSchema, AMissingRequiredArgumentIsNamed )
    {
        // Llama 3.1 8B called get_weather with {} (2026-10-25), and Chat ran it.
        const ArgumentSchema schema( kWeatherParameters );

        EXPECT_EQ( mismatch( schema, "{}" ), "'location' is required" );
    }

    TEST( ArgumentSchema, ArgumentsThatAreNotAnObjectFail )
    {
        const ArgumentSchema schema( kWeatherParameters );

        EXPECT_EQ( mismatch( schema, R"({"location":)" ), "the arguments are not JSON" );
        EXPECT_EQ( mismatch( schema, R"(["Paris"])" ), "the arguments are not a JSON object" );
    }

    TEST( ArgumentSchema, ATypeIsChecked )
    {
        const ArgumentSchema schema(
            R"({"type":"object","properties":{"days":{"type":"integer"},"note":{"type":["string","null"]}}})" );

        EXPECT_EQ( mismatch( schema, R"({"days":"three"})" ), "'days' must be an integer, not a string" );
        EXPECT_EQ( mismatch( schema, R"({"days":2.5})" ), "'days' must be an integer, not a number" );
        EXPECT_FALSE( schema.check( R"({"days":3.0})" ) ) << "3.0 is an integer in JSON Schema";
        EXPECT_FALSE( schema.check( R"({"note":null})" ) );
        EXPECT_EQ( mismatch( schema, R"({"note":7})" ), "'note' must be a string or null, not a number" );
    }

    TEST( ArgumentSchema, EnumAndConstAreChecked )
    {
        const ArgumentSchema schema( kWeatherParameters );
        const ArgumentSchema fixed( R"({"type":"object","properties":{"mode":{"const":"read"}}})" );

        EXPECT_EQ( mismatch( schema, R"({"location":"Paris","units":"kelvin"})" ),
            R"('units' must be one of "celsius", "fahrenheit")" );
        EXPECT_EQ( mismatch( fixed, R"({"mode":"write"})" ), R"('mode' must be "read")" );
    }

    TEST( ArgumentSchema, LengthsAndRangesAreChecked )
    {
        const ArgumentSchema schema( R"({"type":"object","properties":{)"
            R"("query":{"type":"string","minLength":2,"maxLength":4},)"
            R"("count":{"type":"integer","minimum":1,"maximum":10},)"
            R"("ratio":{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}}})" );

        EXPECT_EQ( mismatch( schema, R"({"query":"a"})" ), "'query' must be at least 2 characters" );
        EXPECT_EQ( mismatch( schema, R"({"query":"abcde"})" ), "'query' must be at most 4 characters" );
        EXPECT_FALSE( schema.check( "{\"query\":\"\xC3\xA9t\xC3\xA9\"}" ) ) << "a length counts characters, not bytes";
        EXPECT_EQ( mismatch( schema, R"({"count":0})" ), "'count' must be at least 1" );
        EXPECT_EQ( mismatch( schema, R"({"count":11})" ), "'count' must be at most 10" );
        EXPECT_EQ( mismatch( schema, R"({"ratio":0})" ), "'ratio' must be greater than 0" );
        EXPECT_EQ( mismatch( schema, R"({"ratio":1})" ), "'ratio' must be less than 1" );
        EXPECT_FALSE( schema.check( R"({"query":"ab","count":10,"ratio":0.5})" ) );
    }

    TEST( ArgumentSchema, ArraysAndNestedObjectsAreCheckedAtEveryDepth )
    {
        const ArgumentSchema schema( R"({"type":"object","properties":{"stops":{"type":"array","minItems":1,)"
            R"("maxItems":2,"items":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}})" );

        EXPECT_EQ( mismatch( schema, R"({"stops":[]})" ), "'stops' must have at least 1 item" );
        EXPECT_EQ( mismatch( schema, R"({"stops":[{"city":"A"},{"city":"B"},{"city":"C"}]})" ),
            "'stops' must have at most 2 items" );
        EXPECT_EQ( mismatch( schema, R"({"stops":[{"city":"A"},{}]})" ), "'stops[1].city' is required" );
        EXPECT_EQ( mismatch( schema, R"({"stops":[{"city":3}]})" ), "'stops[0].city' must be a string, not a number" );
    }

    TEST( ArgumentSchema, AdditionalPropertiesAreChecked )
    {
        const ArgumentSchema closed( R"({"type":"object","properties":{"path":{"type":"string"}},"additionalProperties":false})" );
        const ArgumentSchema typed( R"({"type":"object","additionalProperties":{"type":"string"}})" );
        const ArgumentSchema open( R"({"type":"object","properties":{"path":{"type":"string"}}})" );

        EXPECT_EQ( mismatch( closed, R"({"path":"a","mode":"w"})" ), "'mode' is not a parameter of this tool" );
        EXPECT_EQ( mismatch( typed, R"({"a":"x","b":2})" ), "'b' must be a string, not a number" );
        EXPECT_FALSE( open.check( R"({"path":"a","mode":"w"})" ) );
    }

    TEST( ArgumentSchema, AnnotationsCheckNothing )
    {
        const ArgumentSchema schema( R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",)"
            R"("title":"t","$comment":"c","properties":{"url":{"type":"string","format":"uri","default":"x",)"
            R"("examples":["https://example.com"]}}})" );

        EXPECT_FALSE( schema.check( R"({"url":"not a uri"})" ) ) << "format is an annotation, so it is not checked";
    }

    TEST( ArgumentSchema, AKeywordTheCheckDoesNotMakeIsRefused )
    {
        for ( const char* schema : {
            R"({"type":"object","properties":{"city":{"type":"string","pattern":"^[A-Z]"}}})",
            R"({"type":"object","anyOf":[{"required":["a"]},{"required":["b"]}]})",
            R"({"type":"object","properties":{"a":{"$ref":"#/$defs/a"}},"$defs":{"a":{"type":"string"}}})",
            R"({"type":"object","properties":{"tags":{"type":"array","uniqueItems":true}}})",
            R"({"type":"object","properties":{"tuple":{"type":"array","items":[{"type":"string"}]}}})" } )
        {
            EXPECT_THROW( ArgumentSchema{ schema }, std::invalid_argument ) << schema;
        }

        try
        {
            ArgumentSchema{ R"({"type":"object","properties":{"city":{"type":"string","pattern":"^[A-Z]"}}})" };
        }
        catch ( const std::invalid_argument& error )
        {
            EXPECT_STREQ( error.what(), "the schema uses 'pattern' at '/properties/city', which Mila does not check" );
        }
    }

    TEST( ArgumentSchema, ASchemaThatIsNotAnObjectOrNamesNoTypeIsRefused )
    {
        EXPECT_THROW( ArgumentSchema{ "{not json" }, std::invalid_argument );
        EXPECT_THROW( ArgumentSchema{ "[]" }, std::invalid_argument );
        EXPECT_THROW( ArgumentSchema{ R"({"type":"text"})" }, std::invalid_argument );
        EXPECT_THROW( ArgumentSchema{ R"({"type":"object","required":"city"})" }, std::invalid_argument );
        EXPECT_THROW( ArgumentSchema{ R"({"type":"object","properties":{"n":{"minimum":"1"}}})" }, std::invalid_argument );
    }
}
