/**
 * @file ToolRules.cpp
 * @brief The person's rules give a call allow, ask or deny, matching on its tool and its arguments, with a deny
 *        never overridden (ToolCalling.md 3.3).
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        ArgumentCondition pathMatches( const char* pattern )
        {
            return { "/path", ArgumentCondition::Test::Matches, { pattern } };
        }
    }

    TEST( ToolRules, ACallNoRuleAppliesToIsAskedAbout )
    {
        const ToolRules rules = ToolRules{}.add( { "read_file", ToolPermission::Allow } );

        EXPECT_EQ( ToolRules{}.decide( "read_file", "{}" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "write_file", "{}" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "read_file", "{}" ), ToolPermission::Allow );
    }

    TEST( ToolRules, RulesMatchOnArguments )
    {
        // A read inside the project folder allowed, and one outside it asked about.
        const ToolRules rules = ToolRules{}.add( { "read_file", ToolPermission::Allow, { pathMatches( "D:/Project/*" ) } } );

        EXPECT_EQ( rules.decide( "read_file", R"({"path":"D:/Project/src/main.cpp"})" ), ToolPermission::Allow );
        EXPECT_EQ( rules.decide( "read_file", R"({"path":"D:/Other/secret.txt"})" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "read_file", R"({"file":"D:/Project/a"})" ), ToolPermission::Ask )
            << "a condition on an argument the call does not have does not hold";
    }

    TEST( ToolRules, EveryConditionMustHold )
    {
        const ToolRules rules = ToolRules{}.add( { "fetch", ToolPermission::Allow, {
            { "/method", ArgumentCondition::Test::IsOneOf, { "GET", "HEAD" } },
            { "/url", ArgumentCondition::Test::Matches, { "https://docs.example.com/*" } } } } );

        EXPECT_EQ( rules.decide( "fetch", R"({"method":"GET","url":"https://docs.example.com/a"})" ), ToolPermission::Allow );
        EXPECT_EQ( rules.decide( "fetch", R"({"method":"POST","url":"https://docs.example.com/a"})" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "fetch", R"({"method":"GET","url":"https://evil.example.com/a"})" ), ToolPermission::Ask );
    }

    TEST( ToolRules, AValueThatIsNotAStringIsComparedAsItsJsonText )
    {
        const ToolRules rules = ToolRules{}.add(
            { "list", ToolPermission::Deny, { { "/options/recursive", ArgumentCondition::Test::IsOneOf, { "true" } } } } );

        EXPECT_EQ( rules.decide( "list", R"({"options":{"recursive":true}})" ), ToolPermission::Deny );
        EXPECT_EQ( rules.decide( "list", R"({"options":{"recursive":false}})" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "list", R"({"options":"recursive"})" ), ToolPermission::Ask )
            << "a pointer through a value that is not an object names nothing";
    }

    TEST( ToolRules, ADenyWinsOverAnAskAndAnAskOverAnAllow )
    {
        const ToolRules rules = ToolRules{}
            .add( { "*", ToolPermission::Allow } )
            .add( { "write_*", ToolPermission::Ask } )
            .add( { "write_file", ToolPermission::Deny, { pathMatches( "C:/Windows/*" ) } } );

        EXPECT_EQ( rules.decide( "read_file", R"({"path":"C:/Windows/a"})" ), ToolPermission::Allow );
        EXPECT_EQ( rules.decide( "write_file", R"({"path":"D:/Project/a"})" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "write_file", R"({"path":"C:/Windows/a"})" ), ToolPermission::Deny );

        // The order rules were added in decides nothing.
        const ToolRules reversed = ToolRules{}
            .add( { "write_file", ToolPermission::Deny, { pathMatches( "C:/Windows/*" ) } } )
            .add( { "write_*", ToolPermission::Ask } )
            .add( { "*", ToolPermission::Allow } );

        EXPECT_EQ( reversed.decide( "write_file", R"({"path":"C:/Windows/a"})" ), ToolPermission::Deny );
        EXPECT_EQ( reversed.decide( "write_file", R"({"path":"D:/Project/a"})" ), ToolPermission::Ask );
    }

    TEST( ToolRules, APatternIsMatchedWhole )
    {
        const ToolRules rules = ToolRules{}
            .add( { "get_weather", ToolPermission::Allow } )
            .add( { "search", ToolPermission::Allow, { pathMatches( "*.md" ) } } );

        EXPECT_EQ( rules.decide( "get_weather_forecast", "{}" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "search", R"({"path":"notes.md"})" ), ToolPermission::Allow );
        EXPECT_EQ( rules.decide( "search", R"({"path":"notes.md.exe"})" ), ToolPermission::Ask );
        EXPECT_EQ( rules.decide( "search", R"({"path":"a*b.md"})" ), ToolPermission::Allow );
    }

    TEST( ToolRules, ArgumentsThatAreNotJsonMatchOnlyRulesWithNoConditions )
    {
        const ToolRules rules = ToolRules{}
            .add( { "read_file", ToolPermission::Allow } )
            .add( { "read_file", ToolPermission::Deny, { pathMatches( "*" ) } } );

        EXPECT_EQ( rules.decide( "read_file", "{not json" ), ToolPermission::Allow );
    }

    TEST( ToolRules, ARuleItCannotApplyIsRefused )
    {
        ToolRules rules;

        EXPECT_THROW( rules.add( { "", ToolPermission::Allow } ), std::invalid_argument );
        EXPECT_THROW( rules.add( { "read_file", ToolPermission::Allow, { { "path", ArgumentCondition::Test::Matches, { "*" } } } } ),
            std::invalid_argument );
        EXPECT_THROW( rules.add( { "read_file", ToolPermission::Allow, { { "/a~2", ArgumentCondition::Test::Matches, { "*" } } } } ),
            std::invalid_argument );
        EXPECT_THROW( rules.add( { "read_file", ToolPermission::Allow, { { "/path", ArgumentCondition::Test::IsOneOf, {} } } } ),
            std::invalid_argument );
        EXPECT_TRUE( rules.rules().empty() );
    }
}
