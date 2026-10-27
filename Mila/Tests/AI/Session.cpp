/**
 * @file Session.cpp
 * @brief A session holds what a program gives an agent before any model exists, forgets the tools allowed for a
 *        conversation with it, and is copied as a fork (Direction.md 4.3).
 */

#include <gtest/gtest.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;
    using namespace Mila::Dnn;

    namespace
    {
        constexpr const char* kParameters =
            R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})";

        constexpr const char* kArguments = R"({"city":"Paris"})";

        Tool weatherTool()
        {
            return Tool{
                "get_weather",
                "Get the current weather in a city.",
                kParameters,
                []( std::string_view, const ToolGrant&, std::stop_token ) { return std::string( "Sunny." ); } };
        }

        /// One call of the weather tool through the session's own gate and asker.
        ToolOutcome call( Session& session )
        {
            return session.gate().answer( "get_weather", kArguments, session.asker(), {}, session.limits() );
        }
    }

    TEST( Session, HoldsWhatItIsGivenBeforeAnyModelExists )
    {
        Session session;

        session.withTool( weatherTool() )
            .withSystemPrompt( "You are terse." )
            .withRules( ToolRules{}.add( { "get_weather", ToolPermission::Allow } ) )
            .withLimits( { .rounds = 3, .tool_time = std::chrono::milliseconds( 500 ) } )
            .withTemplateVariable( "enable_thinking", "false" )
            .withSampling( SamplingOverride{ .temperature = 0.5f } );

        ASSERT_EQ( session.tools().size(), 1u );
        EXPECT_EQ( session.tools().front().name, "get_weather" );
        EXPECT_EQ( session.systemPrompt(), "You are terse." );
        EXPECT_EQ( session.gate().rules().decide( "get_weather", kArguments ), ToolPermission::Allow );
        EXPECT_EQ( session.limits().rounds, 3 );
        EXPECT_EQ( session.limits().tool_time, std::chrono::milliseconds( 500 ) );
        EXPECT_EQ( session.templateVariables().at( "enable_thinking" ), "false" );
        EXPECT_EQ( session.sampling().temperature, 0.5f );
        EXPECT_FALSE( session.sampling().top_k.has_value() );
        EXPECT_TRUE( session.conversation().empty() );
    }

    TEST( Session, RefusesALimitOfZeroOrLessAndKeepsTheLimitsItHad )
    {
        Session session;
        session.withLimits( { .rounds = 2 } );

        EXPECT_THROW( session.withLimits( { .rounds = 0 } ), std::invalid_argument );
        EXPECT_THROW( session.withLimits( { .time = std::chrono::milliseconds( 0 ) } ), std::invalid_argument );
        EXPECT_THROW( session.withLimits( { .reply_tokens = 0 } ), std::invalid_argument );
        EXPECT_THROW( session.withLimits( { .tool_time = std::chrono::milliseconds( -1 ) } ), std::invalid_argument );
        EXPECT_THROW( session.withLimits( { .result_tokens = 0 } ), std::invalid_argument );

        EXPECT_EQ( session.limits().rounds, 2 );
    }

    TEST( Session, ANewOrReopenedConversationForgetsTheToolsAllowedForTheLastOne )
    {
        int asked = 0;
        Session session;

        session.withTool( weatherTool() ).onAsk( [&asked]( const ToolQuestion& )
            {
                ++asked;

                return ToolAnswer::ForConversation;
            } );

        EXPECT_EQ( call( session ).decision, ToolDecision::AllowedByPersonForConversation );
        EXPECT_EQ( call( session ).decision, ToolDecision::AllowedForConversation );
        EXPECT_EQ( asked, 1 );

        session.turns().push_back( { Conversation::Role::User, "Hello" } );
        session.newConversation();

        EXPECT_TRUE( session.conversation().empty() );
        EXPECT_EQ( call( session ).decision, ToolDecision::AllowedByPersonForConversation );
        EXPECT_EQ( asked, 2 );

        session.openConversation( { { Conversation::Role::User, "Hello" }, { Conversation::Role::Assistant, "Hi." } } );

        EXPECT_EQ( session.conversation().size(), 2u );
        EXPECT_EQ( call( session ).decision, ToolDecision::AllowedByPersonForConversation );
        EXPECT_EQ( asked, 3 );
    }

    TEST( Session, WithNoOneToAskACallTheRulesAskAboutIsDeclined )
    {
        Session session;
        session.withTool( weatherTool() );

        EXPECT_EQ( call( session ).decision, ToolDecision::NoOneToAsk );
    }

    TEST( Session, OfferingNoToolsKeepsTheRules )
    {
        Session session;

        session.withTool( weatherTool() ).withRules( ToolRules{}.add( { "*", ToolPermission::Deny } ) );
        session.withoutTools();

        EXPECT_TRUE( session.tools().empty() );
        EXPECT_EQ( session.gate().rules().decide( "get_weather", kArguments ), ToolPermission::Deny );

        // A tool once offered can be offered again.
        EXPECT_NO_THROW( session.withTool( weatherTool() ) );
    }

    TEST( Session, ACopyIsAForkChangedApartFromTheOriginal )
    {
        Session original;
        original.withTool( weatherTool() ).withSystemPrompt( "You are terse." );
        original.openConversation( { { Conversation::Role::User, "Hello" } } );

        Session fork = original;
        fork.turns().push_back( { Conversation::Role::Assistant, "Hi." } );
        fork.withSystemPrompt( "You are thorough." ).withoutTools();

        EXPECT_EQ( original.conversation().size(), 1u );
        EXPECT_EQ( original.systemPrompt(), "You are terse." );
        EXPECT_EQ( original.tools().size(), 1u );

        EXPECT_EQ( fork.conversation().size(), 2u );
        EXPECT_TRUE( fork.tools().empty() );
    }
}
