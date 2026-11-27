/**
 * @file SetupSteps.cpp
 * @brief What the setup has to accomplish is the app's: kept with its chat and never as a plan, read by the workspace for
 *        an unfinished setup, and finished once its two questions are answered or skipped (AssistantOnboarding.md 4.4).
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

    TEST( SetupStepsTest, AFreshSetupHasEveryStepPending )
    {
        const SetupSteps steps = SetupSteps::fresh();

        EXPECT_TRUE( steps.pending( Step::Name ) );
        EXPECT_TRUE( steps.pending( Step::Uses ) );
        EXPECT_FALSE( steps.finished() );
    }

    TEST( SetupStepsTest, TheSetupIsFinishedOnceBothQuestionsAreAnsweredOrSkipped )
    {
        SetupSteps steps = SetupSteps::fresh();

        steps.done( Step::Name, "Sam" );
        EXPECT_FALSE( steps.finished() ) << "what Sam will use Mila for is still to ask";

        steps.skip( Step::Uses );
        EXPECT_TRUE( steps.finished() ) << "a skip is an answer";
    }

    TEST( SetupStepsTest, ObjectivesSurviveBeingStoredWithWhatWasSaid )
    {
        SetupSteps steps = SetupSteps::fresh();
        steps.done( Step::Name, "Sam" );

        const nlohmann::json stored = steps.toJson();
        ASSERT_EQ( stored.size(), 2u );
        EXPECT_EQ( stored[ 0 ][ "objective" ], "What to call you" );
        EXPECT_EQ( stored[ 0 ][ "status" ], "done" );
        EXPECT_EQ( stored[ 0 ][ "said" ], "Sam" );
        EXPECT_EQ( stored[ 1 ][ "status" ], "pending" );

        const SetupSteps read = SetupSteps::fromJson( stored );
        EXPECT_FALSE( read.pending( Step::Name ) );
        EXPECT_TRUE( read.pending( Step::Uses ) );
        EXPECT_EQ( read.toJson(), stored );
    }

    TEST( SetupStepsTest, AnythingElseStoredIsAFreshStart )
    {
        const nlohmann::json plan = nlohmann::json::array( { { { "text", "Write the parser" }, { "status", "done" } } } );

        EXPECT_EQ( SetupSteps::fromJson( plan ).toJson(), SetupSteps::fresh().toJson() );
        EXPECT_EQ( SetupSteps::fromJson( nullptr ).toJson(), SetupSteps::fresh().toJson() );
        EXPECT_TRUE( SetupSteps::outstandingIn( nullptr ).is_null() ) << "a chat that is not the setup's has none";

        // Round 12 kept a third, Mila's own assessment, which no longer exists.
        nlohmann::json earlier = SetupSteps::fresh().toJson();
        earlier.push_back( { { "objective", "What Mila needs for your work" }, { "status", "pending" }, { "said", "" } } );
        EXPECT_TRUE( SetupSteps::outstandingIn( earlier ).is_null() ) << "an earlier build's objectives are not flagged";
    }

    // What the workspace reads of the setup's chat: what is still to accomplish, for its Needs you card, and nothing of a
    // plan, since the setup is not one.
    TEST( SetupStepsTest, TheWorkspaceReadsWhatTheSetupHasStillToAccomplishAndNoPlan )
    {
        SetupSteps steps = SetupSteps::fresh();
        steps.done( Step::Name, "Sam" );

        const nlohmann::json unfinished = summarizeConversation( { { "setup", steps.toJson() }, { "messages", nlohmann::json::array() } } );
        EXPECT_EQ( unfinished[ "setup" ], nlohmann::json::array( { "What you will use Mila for" } ) );
        EXPECT_TRUE( unfinished[ "progress" ].is_null() );

        steps.skip( Step::Uses );

        const nlohmann::json finished = summarizeConversation( { { "setup", steps.toJson() }, { "messages", nlohmann::json::array() } } );
        EXPECT_EQ( finished[ "setup" ], nlohmann::json::array() );

        const nlohmann::json other = summarizeConversation( { { "messages", nlohmann::json::array() } } );
        EXPECT_TRUE( other[ "setup" ].is_null() );
    }
}
