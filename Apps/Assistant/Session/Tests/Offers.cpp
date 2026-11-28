/**
 * @file Offers.cpp
 * @brief The offers Mila makes, a project gathering chats or a chat filed into a project, and how they are found and
 *        undone (AssistantInterface.md 5.3, 5.4 and 6).
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

import nlohmann.json;
import Assistant.Offers;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        std::vector<nlohmann::json> projects()
        {
            return {
                { { "id", "project-1" }, { "name", "Calculator app" }, { "created", "2026-11-01T10:00:00.000Z" }, { "updated", "2026-11-10T10:00:00.000Z" } },
                { { "id", "project-2" }, { "name", "Calculator tests" }, { "created", "2026-11-02T10:00:00.000Z" }, { "updated", "2026-11-01T10:00:00.000Z" } },
                { { "id", "project-3" }, { "name", "Holiday in Lisbon" }, { "created", "2026-11-03T10:00:00.000Z" }, { "updated", "2026-11-12T10:00:00.000Z" } } };
        }

        nlohmann::json chat( std::string id, std::string title, nlohmann::json project = nullptr,
            nlohmann::json offers = nlohmann::json::array() )
        {
            return { { "id", std::move( id ) }, { "title", std::move( title ) }, { "updated", "2026-11-15T10:00:00.000Z" },
                { "project", std::move( project ) }, { "offers", std::move( offers ) } };
        }

        nlohmann::json chats()
        {
            return nlohmann::json::array( {
                chat( "chat-asking", "Which tomatoes grow in shade" ),
                chat( "chat-beds", "Raised beds for the garden" ),
                chat( "chat-soil", "Soil for the vegetable patch" ),
                chat( "chat-keys", "Keys of the calculator", "project-1" ),
                chat( "chat-taxes", "2026 taxes" ),
                chat( "chat-pots-1", "Pots for herbs" ),
                chat( "chat-pots-2", "Pots for the balcony" ) } );
        }

        std::string proposal( const std::string& name, const std::string& named_chats, const std::string& purpose = "Growing vegetables" )
        {
            return nlohmann::json{ { "name", name }, { "purpose", purpose }, { "chats", named_chats } }.dump();
        }

        std::vector<std::string> gatheredIds( const nlohmann::json& offered )
        {
            std::vector<std::string> ids;

            for ( const nlohmann::json& gathered : offered[ "offer" ][ "chats" ] )
            {
                ids.push_back( gathered[ "id" ].get<std::string>() );
            }

            return ids;
        }

        nlohmann::json assistant( nlohmann::json calls )
        {
            return { { "role", "assistant" }, { "content", "" }, { "tool_calls", std::move( calls ) } };
        }

        nlohmann::json withOffer( std::string call_id, nlohmann::json offer )
        {
            return { { "call_id", std::move( call_id ) }, { "name", "propose_project" }, { "offer", std::move( offer ) } };
        }
    }

    TEST( OffersTest, AProjectGathersTheAskingChatFirstThenEachChatNamedByItsTitle )
    {
        const nlohmann::json offered = offerProject( proposal( "Garden", "Raised beds for the garden\n- soil\n3. Raised beds" ),
            chats(), projects(), "chat-asking", true );

        ASSERT_TRUE( offered[ "offer" ].is_object() );
        EXPECT_EQ( gatheredIds( offered ), ( std::vector<std::string>{ "chat-asking", "chat-beds", "chat-soil" } ) );
        EXPECT_EQ( offered[ "offer" ][ "kind" ], "project" );
        EXPECT_EQ( offered[ "offer" ][ "state" ], "open" );
        EXPECT_EQ( offered[ "offer" ][ "name" ], "Garden" );
        EXPECT_EQ( offered[ "offer" ][ "purpose" ], "Growing vegetables" );
        EXPECT_EQ( offered[ "offer" ][ "in_window" ], false );

        const std::string result = offered[ "result" ].get<std::string>();

        EXPECT_NE( result.find( "this chat and 2 others: Raised beds for the garden; Soil for the vegetable patch" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "answers in their next message" ), std::string::npos ) << result;
        // Gemma 4 26B-A4B planned in the same reply as the offer, and the offer was never answered (Todd's walk, 2026-11-27).
        EXPECT_NE( result.find( "do not plan or start the work it is for until they answer" ), std::string::npos ) << result;
    }

    TEST( OffersTest, TheChatsToGatherMayBeAListOfTitles )
    {
        // As the schema asks, and as Gemma 4 26B-A4B wrote them (Todd's walk, 2026-11-27); an empty list is this chat alone.
        const std::string listed = nlohmann::json{ { "name", "Garden" }, { "purpose", "Growing vegetables" },
            { "chats", nlohmann::json::array( { "Raised beds for the garden", "soil" } ) } }.dump();

        EXPECT_EQ( gatheredIds( offerProject( listed, chats(), projects(), "chat-asking", true ) ),
            ( std::vector<std::string>{ "chat-asking", "chat-beds", "chat-soil" } ) );

        const std::string alone = nlohmann::json{ { "name", "Garden" }, { "chats", nlohmann::json::array() } }.dump();

        EXPECT_EQ( gatheredIds( offerProject( alone, chats(), projects(), "chat-asking", true ) ),
            ( std::vector<std::string>{ "chat-asking" } ) );
    }

    TEST( OffersTest, ATitleThatBeginsWithANumberIsNotAListsNumber )
    {
        const nlohmann::json offered = offerProject( proposal( "Paperwork", "2026 taxes" ), chats(), projects(), "chat-asking", true );

        EXPECT_EQ( gatheredIds( offered ), ( std::vector<std::string>{ "chat-asking", "chat-taxes" } ) );
    }

    TEST( OffersTest, AChatInAProjectSeveralThatFitAndNoneAreToldBackAndNotGathered )
    {
        const nlohmann::json offered = offerProject( proposal( "Garden", "Keys of the calculator\npots\nweather" ),
            chats(), projects(), "chat-asking", true );

        EXPECT_EQ( gatheredIds( offered ), ( std::vector<std::string>{ "chat-asking" } ) );

        const std::string result = offered[ "result" ].get<std::string>();

        EXPECT_NE( result.find( "\"Keys of the calculator\" is in a project, so it stays there" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "\"pots\" fits several chats: Pots for herbs; Pots for the balcony" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "\"weather\" names no chat" ), std::string::npos ) << result;
    }

    TEST( OffersTest, InAChatThatHasReadFromOutsideTheModelIsToldNoTitleAndThePersonAnswersInTheWindow )
    {
        const nlohmann::json offered = offerProject( proposal( "Garden", "raised beds\npots" ), chats(), projects(), "chat-asking", false );

        ASSERT_TRUE( offered[ "offer" ].is_object() );
        EXPECT_EQ( offered[ "offer" ][ "in_window" ], true );

        const std::string result = offered[ "result" ].get<std::string>();

        EXPECT_NE( result.find( "this chat and 1 other." ), std::string::npos ) << result;
        EXPECT_EQ( result.find( "Raised beds for the garden" ), std::string::npos ) << result;
        EXPECT_EQ( result.find( "Pots for" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "\"pots\" fits 2 chats" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "answers in the window" ), std::string::npos ) << result;
    }

    TEST( OffersTest, NothingIsOfferedWithNoNameOrANameAProjectHas )
    {
        EXPECT_TRUE( offerProject( proposal( "  ", "" ), chats(), projects(), "chat-asking", true )[ "offer" ].is_null() );

        const nlohmann::json taken = offerProject( proposal( "calculator APP", "" ), chats(), projects(), "chat-asking", true );

        EXPECT_TRUE( taken[ "offer" ].is_null() );
        EXPECT_NE( taken[ "result" ].get<std::string>().find( "already a project called Calculator app" ), std::string::npos );
    }

    TEST( OffersTest, NothingIsOfferedFromAChatInAProject )
    {
        EXPECT_TRUE( offerProject( proposal( "Garden", "" ), chats(), projects(), "chat-keys", true )[ "offer" ].is_null() );
    }

    TEST( OffersTest, ChatsThePersonDeclinedAProjectForAreNotOfferedOneAgain )
    {
        nlohmann::json held = chats();
        held[ 0 ][ "offers" ] = nlohmann::json::array( { { { "kind", "project" }, { "state", "declined" }, { "name", "Garden" },
            { "chats", nlohmann::json::array( { { { "id", "chat-asking" } }, { { "id", "chat-beds" } }, { { "id", "chat-soil" } } } ) } } } );

        const nlohmann::json again = offerProject( proposal( "Vegetables", "raised beds" ), held, projects(), "chat-asking", true );

        EXPECT_TRUE( again[ "offer" ].is_null() );
        EXPECT_NE( again[ "result" ].get<std::string>().find( "declined" ), std::string::npos );

        // A chat the person never declined one for makes it a new offer.
        EXPECT_TRUE( offerProject( proposal( "Vegetables", "taxes" ), held, projects(), "chat-asking", true )[ "offer" ].is_object() );
    }

    TEST( OffersTest, AnOfferWaitingOnTheSameChatsIsNotMadeTwice )
    {
        nlohmann::json held = chats();
        held[ 1 ][ "offers" ] = nlohmann::json::array( { { { "kind", "project" }, { "state", "undecided" }, { "name", "Garden" },
            { "chats", nlohmann::json::array( { { { "id", "chat-asking" } }, { { "id", "chat-beds" } } } ) } } } );

        EXPECT_TRUE( offerProject( proposal( "Garden beds", "raised beds" ), held, projects(), "chat-asking", true )[ "offer" ].is_null() );
    }

    TEST( OffersTest, AMoveNamesOneProjectAsOpenPlaceMatchesOne )
    {
        const nlohmann::json one = projectForMove( R"({"project":"the lisbon project"})", projects(), true );

        ASSERT_TRUE( one.contains( "project" ) );
        EXPECT_EQ( one[ "project" ][ "id" ], "project-3" );
        EXPECT_EQ( one[ "project" ][ "name" ], "Holiday in Lisbon" );

        const nlohmann::json several = projectForMove( R"({"project":"calculator"})", projects(), true );

        EXPECT_FALSE( several.contains( "project" ) );
        EXPECT_NE( several[ "result" ].get<std::string>().find( "Calculator app; Calculator tests" ), std::string::npos );
        EXPECT_EQ( projectForMove( R"({"project":"calculator"})", projects(), false )[ "result" ], "2 projects fit, and nothing was moved." );
    }

    TEST( OffersTest, AMoveToNoProjectListsThemOnlyWhereNamesMayReachTheModel )
    {
        const nlohmann::json allowed = projectForMove( R"({"project":"garden"})", projects(), true );

        EXPECT_NE( allowed[ "result" ].get<std::string>().find( "The projects are: Calculator app; Calculator tests; Holiday in Lisbon." ), std::string::npos );
        EXPECT_EQ( projectForMove( R"({"project":"garden"})", projects(), false )[ "result" ], "No project is called that, and nothing was moved." );
    }

    TEST( OffersTest, FilingIsAllowedOnceThePersonTookAMoveIntoTheProject )
    {
        const std::vector<nlohmann::json> made = { { { "kind", "project.made" }, { "suggested", true } },
            { { "kind", "chat.moved" }, { "suggested", "project" } } };

        EXPECT_FALSE( filingAllowed( made ) );

        std::vector<nlohmann::json> taken = made;
        taken.push_back( { { "kind", "chat.moved" }, { "suggested", "move" } } );

        EXPECT_TRUE( filingAllowed( taken ) );
    }

    TEST( OffersTest, ThePendingOfferIsOneOpenInMilasLastReplyOrAMoveMadeAndSaidThere )
    {
        nlohmann::json messages = nlohmann::json::array( {
            { { "role", "user" }, { "content", "garden" } },
            assistant( nlohmann::json::array( { withOffer( "call-1", { { "kind", "project" }, { "state", "open" } } ) } ) ) } );

        EXPECT_EQ( pendingOffer( messages ), ( nlohmann::json{ { "message", 1 }, { "call", 0 } } ) );

        // A later reply leaves it behind: only the next message answers an offer.
        messages.push_back( { { "role", "user" }, { "content", "and the soil?" } } );
        messages.push_back( { { "role", "assistant" }, { "content", "Loam." }, { "tool_calls", nullptr } } );

        EXPECT_TRUE( pendingOffer( messages ).is_null() );

        messages.push_back( assistant( nlohmann::json::array( {
            withOffer( "call-2", { { "kind", "move" }, { "state", "accepted" }, { "said", true } } ) } ) ) );

        EXPECT_EQ( pendingOffer( messages ), ( nlohmann::json{ { "message", 4 }, { "call", 0 } } ) );

        messages.back()[ "tool_calls" ][ 0 ][ "offer" ][ "said" ] = false;

        EXPECT_TRUE( pendingOffer( messages ).is_null() );
    }

    TEST( OffersTest, EveryOfferIsFoundWithWhereItIsKeptAndCanBeChangedInPlace )
    {
        nlohmann::json messages = nlohmann::json::array( {
            { { "role", "user" }, { "content", "garden" } },
            assistant( nlohmann::json::array( { { { "call_id", "call-1" }, { "name", "find_records" } },
                withOffer( "call-2", { { "kind", "project" }, { "state", "open" }, { "name", "Garden" } } ) } ) ) } );

        const nlohmann::json offers = offersIn( messages );

        ASSERT_EQ( offers.size(), 1u );
        EXPECT_EQ( offers[ 0 ][ "message" ], 1 );
        EXPECT_EQ( offers[ 0 ][ "call_id" ], "call-2" );
        EXPECT_EQ( offers[ 0 ][ "name" ], "Garden" );

        nlohmann::json* offer = offerOnCall( messages, "call-2" );

        ASSERT_NE( offer, nullptr );
        ( *offer )[ "state" ] = "declined";

        EXPECT_EQ( messages[ 1 ][ "tool_calls" ][ 1 ][ "offer" ][ "state" ], "declined" );
        EXPECT_EQ( offerOnCall( messages, "call-1" ), nullptr );
    }

    TEST( OffersTest, AProjectMadeFromASuggestionIsUndoneOnlyUntilItChangesOrAnotherIsMade )
    {
        const nlohmann::json made = { { "id", "project-4" }, { "name", "Garden" }, { "created", "2026-11-20T10:00:00.000Z" },
            { "suggested", { { "conversation_id", "chat-asking" }, { "call_id", "call-2" } } } };

        std::vector<nlohmann::json> all = projects();
        all.push_back( made );

        std::vector<nlohmann::json> history = { { { "kind", "project.made" }, { "suggested", true } },
            { { "kind", "chat.moved" }, { "suggested", "project" } }, { { "kind", "suggestion.made" } } };

        EXPECT_TRUE( projectUndoable( made, history, all ) );

        std::vector<nlohmann::json> renamed = history;
        renamed.push_back( { { "kind", "project.renamed" } } );

        EXPECT_FALSE( projectUndoable( made, renamed, all ) );

        std::vector<nlohmann::json> moved_by_person = history;
        moved_by_person.push_back( { { "kind", "chat.moved" } } );

        EXPECT_FALSE( projectUndoable( made, moved_by_person, all ) );

        // The folder the gathered chat held became the project's as it was made: still the making (Todd, 2026-11-27).
        std::vector<nlohmann::json> with_folder = history;
        with_folder.push_back( { { "kind", "project.folder_added" }, { "suggested", "project" } } );

        EXPECT_TRUE( projectUndoable( made, with_folder, all ) );

        // One the person gave later is a change.
        std::vector<nlohmann::json> folder_given = history;
        folder_given.push_back( { { "kind", "project.folder_added" } } );

        EXPECT_FALSE( projectUndoable( made, folder_given, all ) );

        std::vector<nlohmann::json> later = all;
        later.push_back( { { "id", "project-5" }, { "created", "2026-11-21T10:00:00.000Z" } } );

        EXPECT_FALSE( projectUndoable( made, history, later ) );

        nlohmann::json by_person = made;
        by_person.erase( "suggested" );

        EXPECT_FALSE( projectUndoable( by_person, history, all ) );
    }
}
