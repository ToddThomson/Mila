/**
 * @file Places.cpp
 * @brief open_place matches a person's words to the workspace's views, projects and chats, and says what a place shows
 *        (AssistantInterface.md 5.2).
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

import nlohmann.json;
import Desktop.Places;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        std::vector<nlohmann::json> projects()
        {
            return {
                { { "id", "project-1" }, { "name", "Calculator app" }, { "purpose", "A calculator for the desk" }, { "updated", "2026-11-10T10:00:00.000Z" } },
                { { "id", "project-2" }, { "name", "Holiday in Lisbon" }, { "purpose", "" }, { "updated", "2026-11-12T10:00:00.000Z" } },
                { { "id", "project-3" }, { "name", "Calculator tests" }, { "purpose", "" }, { "updated", "2026-11-01T10:00:00.000Z" } } };
        }

        nlohmann::json chats()
        {
            return nlohmann::json::array( {
                { { "id", "chat-1" }, { "title", "Groceries for the week" }, { "updated", "2026-11-15T10:00:00.000Z" },
                  { "project", nullptr }, { "progress", nullptr }, { "finish", "stop" } },
                { { "id", "chat-2" }, { "title", "Keys of the calculator" }, { "updated", "2026-11-14T10:00:00.000Z" },
                  { "project", "project-1" }, { "progress", { { "steps", 3 }, { "finished", 1 }, { "current", 2 } } }, { "finish", "stop" } },
                { { "id", "chat-3" }, { "title", "Flights" }, { "updated", "2026-11-13T10:00:00.000Z" },
                  { "project", "project-2" }, { "progress", nullptr }, { "finish", "stop" } } } );
        }
    }

    TEST( PlacesTest, AViewsOwnNameOpensThatViewWithOrWithoutAllOrMy )
    {
        EXPECT_EQ( findPlaces( "projects", projects(), chats() )[ 0 ][ "view" ], "projects" );
        EXPECT_EQ( findPlaces( "All chats", projects(), chats() )[ 0 ][ "view" ], "chats" );
        EXPECT_EQ( findPlaces( "my workspace", projects(), chats() )[ 0 ][ "view" ], "workspace" );
        EXPECT_EQ( findPlaces( "activity", projects(), chats() )[ 0 ][ "view" ], "activity" );
        EXPECT_EQ( findPlaces( "reach", projects(), chats() )[ 0 ][ "view" ], "reach" );
        EXPECT_EQ( findPlaces( "workspace", projects(), chats() ).size(), 1u );
    }

    TEST( PlacesTest, AnExactNameIsTheOnePlaceEvenWhenOthersHoldItsWords )
    {
        const nlohmann::json places = findPlaces( "calculator app", projects(), chats() );

        ASSERT_EQ( places.size(), 1u );
        EXPECT_EQ( places[ 0 ][ "view" ], "project" );
        EXPECT_EQ( places[ 0 ][ "id" ], "project-1" );
        EXPECT_EQ( places[ 0 ][ "name" ], "Calculator app" );
    }

    TEST( PlacesTest, WordsAroundANameAreLeftOutAndAWordMayBeTheStartOfOne )
    {
        const nlohmann::json places = findPlaces( "the lisbon project", projects(), chats() );

        ASSERT_EQ( places.size(), 1u );
        EXPECT_EQ( places[ 0 ][ "id" ], "project-2" );

        const nlohmann::json grocery = findPlaces( "grocer", projects(), chats() );

        ASSERT_EQ( grocery.size(), 1u );
        EXPECT_EQ( grocery[ 0 ][ "view" ], "chat" );
        EXPECT_EQ( grocery[ 0 ][ "id" ], "chat-1" );
    }

    TEST( PlacesTest, SeveralThatFitAreProjectsFirstNewestFirstThenChats )
    {
        const nlohmann::json places = findPlaces( "calculator", projects(), chats() );

        ASSERT_EQ( places.size(), 3u );
        EXPECT_EQ( places[ 0 ][ "id" ], "project-1" );
        EXPECT_EQ( places[ 1 ][ "id" ], "project-3" );
        EXPECT_EQ( places[ 2 ][ "id" ], "chat-2" );

        const std::string said = severalPlaces( places );

        EXPECT_NE( said.find( "Ask the user which" ), std::string::npos );
        EXPECT_NE( said.find( "Calculator tests" ), std::string::npos );
    }

    TEST( PlacesTest, TheChatAskingIsNeverAPlaceThoughItsTitleIsTheRequest )
    {
        nlohmann::json with_asking = chats();
        with_asking.push_back( { { "id", "chat-4" }, { "title", "Open my chat about flights" }, { "updated", "2026-11-20T10:00:00.000Z" },
            { "project", nullptr } } );

        EXPECT_EQ( findPlaces( "flight", projects(), with_asking ).size(), 2u );

        const nlohmann::json places = findPlaces( "flight", projects(), with_asking, "chat-4" );

        ASSERT_EQ( places.size(), 1u );
        EXPECT_EQ( places[ 0 ][ "id" ], "chat-3" );
        EXPECT_TRUE( findPlaces( "open my chat about flights", projects(), with_asking, "chat-4" ).empty() );
    }

    TEST( PlacesTest, NothingFitsAndNothingIsReturned )
    {
        EXPECT_TRUE( findPlaces( "garden", projects(), chats() ).empty() );
        EXPECT_TRUE( findPlaces( "  ", projects(), chats() ).empty() );
        EXPECT_TRUE( findPlaces( "the project", projects(), chats() ).empty() ) << "filler alone named a place";
    }

    TEST( PlacesTest, AtMostFivePlacesAreOffered )
    {
        std::vector<nlohmann::json> many;

        for ( int index = 0; index < 8; ++index )
        {
            many.push_back( { { "id", "project-" + std::to_string( index ) }, { "name", "Notes " + std::to_string( index ) },
                { "updated", "2026-11-10T10:00:00.000Z" } } );
        }

        EXPECT_EQ( findPlaces( "notes", many, nlohmann::json::array() ).size(), kMostPlacesOffered );
    }

    TEST( PlacesTest, AllProjectsSaysWhichWaitOnThePersonButNotForTheAskingChat )
    {
        const nlohmann::json all = findPlaces( "projects", projects(), chats() )[ 0 ];
        const std::string described = describePlace( all, projects(), chats(), "chat-3", true );

        EXPECT_NE( described.find( "3 projects" ), std::string::npos );
        EXPECT_NE( described.find( "Calculator app: waiting on the user" ), std::string::npos ) << described;
        EXPECT_NE( described.find( "Holiday in Lisbon: nothing waiting" ), std::string::npos ) << described;

        // The chat asking is the one whose plan is under way.
        const std::string asking = describePlace( all, projects(), chats(), "chat-2", true );

        EXPECT_NE( asking.find( "Calculator app: nothing waiting" ), std::string::npos ) << asking;
    }

    TEST( PlacesTest, InAChatThatReadFromOutsideAListGivesNoNames )
    {
        const nlohmann::json all = findPlaces( "chats", projects(), chats() )[ 0 ];
        const std::string described = describePlace( all, projects(), chats(), "", false );

        EXPECT_NE( described.find( "3 chats" ), std::string::npos );
        EXPECT_EQ( described.find( "Groceries" ), std::string::npos ) << described;

        const std::string named = describePlace( all, projects(), chats(), "", true );

        EXPECT_NE( named.find( "Keys of the calculator, in Calculator app" ), std::string::npos ) << named;
    }

    TEST( PlacesTest, AProjectSaysItsPurposeAndChatsAndAChatOpensWhenTheReplyEnds )
    {
        const nlohmann::json project = findPlaces( "calculator app", projects(), chats() )[ 0 ];
        const std::string described = describePlace( project, projects(), chats(), "", true );

        EXPECT_NE( described.find( "Opened the project Calculator app (A calculator for the desk): 1 chat, waiting on the user" ),
            std::string::npos ) << described;
        EXPECT_NE( described.find( "Keys of the calculator" ), std::string::npos );

        const nlohmann::json chat = findPlaces( "flights", projects(), chats() )[ 0 ];

        EXPECT_NE( describePlace( chat, projects(), chats(), "", true ).find( "opens for the user when this reply ends" ),
            std::string::npos );
        EXPECT_NE( describePlace( chat, projects(), chats(), "chat-3", true ).find( "is this one" ), std::string::npos );
    }

    TEST( PlacesTest, TheLineKeptSaysWhereMilaWent )
    {
        EXPECT_EQ( wentLine( findPlaces( "calculator app", projects(), chats() )[ 0 ] ), "Opened Calculator app" );
        EXPECT_EQ( wentLine( findPlaces( "projects", projects(), chats() )[ 0 ] ), "Opened All projects" );
        EXPECT_EQ( wentLine( findPlaces( "workspace", projects(), chats() )[ 0 ] ), "Opened your workspace" );
    }

    TEST( PlacesTest, AChatWaitsOnThePersonForAStoppedPlanOrAnUnfinishedReply )
    {
        EXPECT_TRUE( waitsOnPerson( chats()[ 1 ] ) );
        EXPECT_FALSE( waitsOnPerson( chats()[ 0 ] ) );
        EXPECT_TRUE( waitsOnPerson( { { "finish", "length" } } ) );
        EXPECT_FALSE( waitsOnPerson( { { "progress", { { "current", nullptr } } }, { "finish", "stop" } } ) );
    }
}
