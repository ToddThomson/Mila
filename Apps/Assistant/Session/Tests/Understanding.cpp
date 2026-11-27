/**
 * @file Understanding.cpp
 * @brief A project's note: written within its limits, waiting when its chat read from outside, accepted or declined,
 *        edited keeping its mark, cleared with a chat it rests on, and given to a new chat as data
 *        (AssistantInterface.md 4.5, 5.6, 6 and 8).
 */

#include <gtest/gtest.h>
#include <format>
#include <string>
#include <variant>
#include <vector>

import nlohmann.json;
import Assistant.Understanding;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        constexpr const char* kNow = "2026-11-21T10:00:00.000Z";

        std::string arguments( std::string summary, std::string questions = "", std::string next = "" )
        {
            return nlohmann::json{ { "summary", std::move( summary ) }, { "questions", std::move( questions ) },
                { "next", std::move( next ) } }.dump();
        }

        nlohmann::json written( const std::string& text, bool exposed = false, nlohmann::json derived_from = nlohmann::json::array(),
            const std::string& id = "note-1", const std::string& chat = "chat-tomatoes" )
        {
            return noteFromArguments( text, id, chat, "call-3", derived_from, exposed, "gemma-4-12b-it-q4_0", kNow );
        }

        nlohmann::json sourcesOf( const nlohmann::json& note )
        {
            nlohmann::json chats = nlohmann::json::array();

            for ( const nlohmann::json& source : note[ "derived_from" ] )
            {
                chats.push_back( source[ "conversation" ] );
            }

            return chats;
        }
    }

    TEST( Understanding, ANoteIsARecordOfWhatMilaWroteAndWhereFrom )
    {
        const nlohmann::json made = written( arguments( "Growing tomatoes on the balcony.",
            "- Which variety?\n2. When to sow?\n\n", "Order the seeds." ) );

        ASSERT_TRUE( made.contains( "note" ) ) << made.dump();
        const nlohmann::json& note = made[ "note" ];

        EXPECT_EQ( note[ "id" ], "note-1" );
        EXPECT_EQ( note[ "kind" ], "understanding" );
        EXPECT_EQ( note[ "summary" ], "Growing tomatoes on the balcony." );
        EXPECT_EQ( note[ "questions" ], nlohmann::json::array( { "Which variety?", "When to sow?" } ) );
        EXPECT_EQ( note[ "next" ], "Order the seeds." );
        EXPECT_EQ( note[ "source" ][ "kind" ], "model" );
        EXPECT_EQ( note[ "source" ][ "ref" ][ "conversation" ], "chat-tomatoes" );
        EXPECT_EQ( note[ "source" ][ "ref" ][ "call" ], "call-3" );
        EXPECT_EQ( note[ "source" ][ "model" ], "gemma-4-12b-it-q4_0" );
        EXPECT_FALSE( note[ "may_direct" ].get<bool>() );
        EXPECT_FALSE( note[ "exposed" ].get<bool>() );
        EXPECT_EQ( sourcesOf( note ), nlohmann::json::array( { "chat-tomatoes" } ) );
    }

    TEST( Understanding, ALineWithNoWordsIsNoQuestion )
    {
        // "``" was Gemma 4 26B-A4B's way of writing none (Todd's walk, 2026-11-27).
        for ( const char* none : { "``", "-", "N/A", "\"\"", "...", "?" } )
        {
            const nlohmann::json made = written( arguments( "Hello World in C++.", none ) );

            ASSERT_TRUE( made.contains( "note" ) ) << none;

            if ( std::string_view( none ) == "N/A" )
            {
                // Letters are kept: what reads as none to a person is a word to the note.
                EXPECT_EQ( made[ "note" ][ "questions" ].size(), 1u );
            }
            else
            {
                EXPECT_TRUE( made[ "note" ][ "questions" ].empty() ) << none;
            }
        }

        EXPECT_EQ( written( arguments( "Hello.", "``\nWhich compiler?" ) )[ "note" ][ "questions" ],
            nlohmann::json::array( { "Which compiler?" } ) );
    }

    TEST( Understanding, ANoteRestsOnItsChatAndOnWhatTheNoteItWasGivenRestedOn )
    {
        const nlohmann::json given = { { { "conversation", "chat-soil" } }, { { "conversation", "chat-tomatoes" } },
            { { "conversation", "chat-beds" } } };

        EXPECT_EQ( sourcesOf( written( arguments( "Summary." ), false, given )[ "note" ] ),
            nlohmann::json::array( { "chat-tomatoes", "chat-soil", "chat-beds" } ) );
    }

    TEST( Understanding, ANoteOverALimitIsRefusedBackToTheModelAndNoneIsMade )
    {
        const std::string long_summary( kSummaryCharacters + 1, 'a' );
        std::string many;

        for ( std::size_t question = 0; question <= kMostQuestions; ++question )
        {
            many += std::format( "Question {}?\n", question );
        }

        for ( const std::string& refused : { arguments( "" ), arguments( long_summary ), arguments( "Summary.", many ),
            arguments( "Summary.", std::string( kQuestionCharacters + 1, 'q' ) ),
            arguments( "Summary.", "", std::string( kNextCharacters + 1, 'n' ) ), std::string( "not json" ) } )
        {
            const nlohmann::json made = written( refused );

            EXPECT_FALSE( made.contains( "note" ) ) << refused;
            ASSERT_TRUE( made.contains( "result" ) );
            EXPECT_TRUE( made[ "result" ].get<std::string>().starts_with( "Not noted:" ) ) << made[ "result" ];
        }

        EXPECT_TRUE( written( arguments( std::string( kSummaryCharacters, 'a' ) ) ).contains( "note" ) );

        // A limit counts characters as a person reads them, not bytes.
        std::string accented;

        for ( std::size_t character = 0; character < kSummaryCharacters; ++character )
        {
            accented += "\xC3\xA9";
        }

        EXPECT_TRUE( written( arguments( accented ) ).contains( "note" ) );
    }

    TEST( Understanding, ANoteFromAChatNotExposedBecomesCurrentAndOneFromAnExposedChatWaits )
    {
        nlohmann::json understanding = emptyUnderstanding();

        EXPECT_EQ( admitNote( understanding, written( arguments( "First." ) )[ "note" ] ), "current" );
        EXPECT_EQ( admitNote( understanding, written( arguments( "From a page." ), true, {}, "note-2" )[ "note" ] ), "waiting" );

        EXPECT_EQ( understanding[ "current" ][ "summary" ], "First." ) << "a note from an exposed chat replaced the current one";
        EXPECT_EQ( understanding[ "waiting" ][ "summary" ], "From a page." );

        // At most one waits, the newest from an exposed chat; one from a chat not exposed leaves it waiting.
        EXPECT_EQ( admitNote( understanding, written( arguments( "From another page." ), true, {}, "note-3" )[ "note" ] ), "waiting" );
        EXPECT_EQ( admitNote( understanding, written( arguments( "Second." ), false, {}, "note-4" )[ "note" ] ), "current" );

        EXPECT_EQ( understanding[ "waiting" ][ "id" ], "note-3" );
        EXPECT_EQ( understanding[ "current" ][ "id" ], "note-4" );
    }

    TEST( Understanding, AWaitingNoteAcceptedBecomesCurrentKeepingItsMarkAndOneDeclinedGoes )
    {
        nlohmann::json understanding = emptyUnderstanding();
        admitNote( understanding, written( arguments( "Current." ) )[ "note" ] );
        admitNote( understanding, written( arguments( "From a page." ), true, {}, "note-2" )[ "note" ] );

        EXPECT_FALSE( decideWaiting( understanding, "note-1", true, kNow ) ) << "a note that is not waiting was decided";
        ASSERT_TRUE( decideWaiting( understanding, "note-2", true, kNow ) );

        EXPECT_EQ( understanding[ "current" ][ "id" ], "note-2" );
        EXPECT_TRUE( understanding[ "current" ][ "exposed" ].get<bool>() ) << "accepting removed the mark";
        EXPECT_EQ( understanding[ "current" ][ "accepted" ], kNow );
        EXPECT_TRUE( understanding[ "waiting" ].is_null() );

        admitNote( understanding, written( arguments( "Another page." ), true, {}, "note-3" )[ "note" ] );
        ASSERT_TRUE( decideWaiting( understanding, "note-3", false, kNow ) );

        EXPECT_EQ( understanding[ "current" ][ "id" ], "note-2" );
        EXPECT_TRUE( understanding[ "waiting" ].is_null() );
        EXPECT_FALSE( decideWaiting( understanding, "note-3", true, kNow ) );
    }

    TEST( Understanding, ThePersonsEditKeepsTheMarkAndWhatItRestsOn )
    {
        nlohmann::json note = written( arguments( "From a page.", "Which?", "Read more." ), true,
            { { { "conversation", "chat-soil" } } } )[ "note" ];

        const auto edited = editedNote( note, { { "summary", "In my words now." }, { "questions", nlohmann::json::array( { "One?", " ", "Two?" } ) } },
            "note-edited", kNow );

        ASSERT_TRUE( std::holds_alternative<nlohmann::json>( edited ) );
        const nlohmann::json& changed = std::get<nlohmann::json>( edited );

        EXPECT_EQ( changed[ "summary" ], "In my words now." );
        EXPECT_EQ( changed[ "questions" ], nlohmann::json::array( { "One?", "Two?" } ) );
        EXPECT_EQ( changed[ "next" ], "Read more." ) << "a field not edited changed";
        EXPECT_TRUE( changed[ "exposed" ].get<bool>() ) << "an edit removed the mark";
        EXPECT_EQ( changed[ "derived_from" ], note[ "derived_from" ] );
        EXPECT_EQ( changed[ "source" ][ "author" ], "person" );
        EXPECT_EQ( changed[ "edited" ], kNow );
        EXPECT_EQ( changed[ "id" ], "note-edited" ) << "a chat that read the note before the edit could rewrite it unread";

        EXPECT_TRUE( std::holds_alternative<std::string>( editedNote( note, { { "summary", "" } }, "note-x", kNow ) ) );
        EXPECT_TRUE( std::holds_alternative<std::string>( editedNote( nullptr, { { "summary", "x" } }, "note-x", kNow ) ) );
    }

    TEST( Understanding, DeletingAChatClearsEveryNoteThatRestsOnIt )
    {
        nlohmann::json understanding = emptyUnderstanding();
        admitNote( understanding, written( arguments( "Current." ), false, { { { "conversation", "chat-soil" } } } )[ "note" ] );
        admitNote( understanding, written( arguments( "Waiting." ), true, {}, "note-2", "chat-page" )[ "note" ] );

        EXPECT_TRUE( clearDerivedFrom( understanding, "chat-other" ).empty() );
        EXPECT_EQ( clearDerivedFrom( understanding, "chat-soil" ), std::vector<std::string>{ "note-1" } );
        EXPECT_TRUE( understanding[ "current" ].is_null() );
        EXPECT_EQ( understanding[ "waiting" ][ "id" ], "note-2" );

        EXPECT_EQ( clearDerivedFrom( understanding, "chat-page" ), std::vector<std::string>{ "note-2" } );
        EXPECT_TRUE( understanding[ "waiting" ].is_null() );
    }

    TEST( Understanding, TheTextGivenToAChatNamesItAsANoteWithWhereItCameFrom )
    {
        const nlohmann::json note = written( arguments( "Growing tomatoes.", "Which variety?", "Order seeds." ) )[ "note" ];
        const std::string given = notesText( note, "Garden", "Tomato seedlings" );

        EXPECT_NE( given.find( "Notes kept for the project \"Garden\"" ), std::string::npos ) << given;
        EXPECT_NE( given.find( "noted on 2026-11-21 in the chat \"Tomato seedlings\"" ), std::string::npos ) << given;
        EXPECT_NE( given.find( "not the user's words" ), std::string::npos ) << given;
        EXPECT_NE( given.find( "Summary: Growing tomatoes." ), std::string::npos ) << given;
        EXPECT_NE( given.find( "- Which variety?" ), std::string::npos ) << given;
        EXPECT_NE( given.find( "Next: Order seeds." ), std::string::npos ) << given;

        const auto edited = editedNote( note, { { "next", "Sow in March." } }, "note-2", "2026-11-22T09:00:00.000Z" );
        const std::string after = notesText( std::get<nlohmann::json>( edited ), "Garden", "" );

        EXPECT_NE( after.find( "noted on 2026-11-22, as the user edited it" ), std::string::npos ) << after;
    }

    TEST( Understanding, WhatANoteRestsOnIsReadFromTheChatsStoredCalls )
    {
        const nlohmann::json messages = nlohmann::json::array( {
            { { "role", "user" }, { "content", "Hello" }, { "tool_calls", nullptr } },
            { { "role", "assistant" }, { "tool_calls", nlohmann::json::array( {
                { { "name", "read_project_notes" }, { "given", { { "id", "note-1" },
                    { "derived_from", { { { "conversation", "chat-soil" } }, { { "conversation", "chat-beds" } } } } } } },
                { { "name", "open_place" } } } ) } } } );

        const nlohmann::json turn = nlohmann::json::array( { { { "name", "read_project_notes" }, { "given", { { "id", "note-2" },
            { "derived_from", { { { "conversation", "chat-beds" } }, { { "conversation", "chat-pots" } } } } } } } } );

        nlohmann::json chats = nlohmann::json::array();

        for ( const nlohmann::json& source : givenDerivedFrom( messages, turn ) )
        {
            chats.push_back( source[ "conversation" ] );
        }

        EXPECT_EQ( chats, nlohmann::json::array( { "chat-soil", "chat-beds", "chat-pots" } ) );
        EXPECT_TRUE( givenDerivedFrom( nlohmann::json::array(), nlohmann::json::array() ).empty() );

        // Which notes the chat read, so one that never read the current note does not write over it.
        EXPECT_TRUE( noteGivenIn( messages, turn, "note-1" ) ) << "a note given in an earlier reply";
        EXPECT_TRUE( noteGivenIn( messages, turn, "note-2" ) ) << "a note given in this reply";
        EXPECT_FALSE( noteGivenIn( messages, turn, "note-3" ) );
        EXPECT_FALSE( noteGivenIn( nlohmann::json::array(), nlohmann::json::array(), "note-1" ) );
    }
}
