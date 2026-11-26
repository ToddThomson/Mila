/**
 * @file SetupSteps.cpp
 * @brief What the setup has to accomplish is the app's: kept with its chat and never as a plan, read by the workspace for
 *        an unfinished setup, and the choice of model put only once the other two are answered or skipped
 *        (AssistantOnboarding.md 4.4).
 */

#include <gtest/gtest.h>
#include <string>

import nlohmann.json;
import Assistant.SetupSteps;
import Assistant.ConversationSummary;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;
    using Step = SetupSteps::Step;

    TEST( SetupStepsTest, AFreshSetupHasEveryStepPendingAndTheChoiceNotYetDue )
    {
        const SetupSteps steps = SetupSteps::fresh();

        EXPECT_TRUE( steps.pending( Step::Name ) );
        EXPECT_TRUE( steps.pending( Step::Uses ) );
        EXPECT_FALSE( steps.readyForChoice() );
        EXPECT_FALSE( steps.finished() );
    }

    TEST( SetupStepsTest, TheChoiceIsDueOnceTheTwoQuestionsAreAnsweredOrSkipped )
    {
        SetupSteps steps = SetupSteps::fresh();

        steps.done( Step::Name, "Sam" );
        EXPECT_FALSE( steps.readyForChoice() ) << "what Sam will use Mila for is still to ask";

        steps.skip( Step::Uses );
        EXPECT_TRUE( steps.readyForChoice() ) << "a skip is an answer";
        EXPECT_FALSE( steps.finished() ) << "what Mila needs is the one that ends the setup";

        steps.done( Step::Needs, "nothing more" );
        EXPECT_TRUE( steps.finished() );
    }

    TEST( SetupStepsTest, ObjectivesSurviveBeingStoredWithWhatWasSaid )
    {
        SetupSteps steps = SetupSteps::fresh();
        steps.done( Step::Name, "Sam" );
        steps.skip( Step::Uses );

        const nlohmann::json stored = steps.toJson();
        ASSERT_EQ( stored.size(), 3u );
        EXPECT_EQ( stored[ 0 ][ "objective" ], "What to call you" );
        EXPECT_EQ( stored[ 0 ][ "status" ], "done" );
        EXPECT_EQ( stored[ 0 ][ "said" ], "Sam" );
        EXPECT_EQ( stored[ 1 ][ "status" ], "skipped" );
        EXPECT_EQ( stored[ 2 ][ "status" ], "pending" );

        const SetupSteps read = SetupSteps::fromJson( stored );
        EXPECT_FALSE( read.pending( Step::Name ) );
        EXPECT_TRUE( read.readyForChoice() );
        EXPECT_EQ( read.toJson(), stored );
    }

    TEST( SetupStepsTest, AnythingElseStoredIsAFreshStart )
    {
        const nlohmann::json plan = nlohmann::json::array( { { { "text", "Write the parser" }, { "status", "done" } } } );

        EXPECT_EQ( SetupSteps::fromJson( plan ).toJson(), SetupSteps::fresh().toJson() );
        EXPECT_EQ( SetupSteps::fromJson( nullptr ).toJson(), SetupSteps::fresh().toJson() );
        EXPECT_TRUE( SetupSteps::outstandingIn( nullptr ).is_null() ) << "a chat that is not the setup's has none";

        nlohmann::json earlier = SetupSteps::fresh().toJson();
        earlier[ 2 ][ "objective" ] = "Which model Mila runs on";
        EXPECT_TRUE( SetupSteps::outstandingIn( earlier ).is_null() ) << "an earlier build's objectives are not flagged";
    }

    // What the workspace reads of the setup's chat: what is still to accomplish, for its Needs you card, and nothing of a
    // plan, since the setup is not one.
    TEST( SetupStepsTest, TheWorkspaceReadsWhatTheSetupHasStillToAccomplishAndNoPlan )
    {
        SetupSteps steps = SetupSteps::fresh();
        steps.done( Step::Name, "Sam" );

        const nlohmann::json unfinished = summarizeConversation( { { "setup", steps.toJson() }, { "messages", nlohmann::json::array() } } );
        EXPECT_EQ( unfinished[ "setup" ], nlohmann::json::array( { "What you will use Mila for", "What Mila needs for your work" } ) );
        EXPECT_TRUE( unfinished[ "progress" ].is_null() );

        steps.skip( Step::Uses );
        steps.done( Step::Needs, "nothing more" );

        const nlohmann::json finished = summarizeConversation( { { "setup", steps.toJson() }, { "messages", nlohmann::json::array() } } );
        EXPECT_EQ( finished[ "setup" ], nlohmann::json::array() );

        const nlohmann::json other = summarizeConversation( { { "messages", nlohmann::json::array() } } );
        EXPECT_TRUE( other[ "setup" ].is_null() );
    }
}
