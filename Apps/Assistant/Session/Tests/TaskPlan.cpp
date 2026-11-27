/**
 * @file TaskPlan.cpp
 * @brief A plan's steps as update_step marks them: an earlier step left in progress is named, so it is closed rather
 *        than left showing unfinished, and the plan says when every step is finished (MilaAssistant.md 3.1).
 */

#include <gtest/gtest.h>
#include <string>

import Assistant.TaskPlan;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        std::string marking( int step, const char* status )
        {
            return std::string( R"({"step":)" ) + std::to_string( step ) + R"(,"status":")" + status + R"("})";
        }
    }

    TEST( TaskPlan, AStepLeftInProgressIsNamedWhenALaterOneIsMarked )
    {
        TaskPlan plan;
        (void)plan.propose( R"({"steps":"Get a folder\nWrite main.cpp\nBuild it"})" );
        (void)plan.update( marking( 1, "in_progress" ) );

        // Gemma 4 26B-A4B went on to step 2 with step 1 never marked done (Todd's walk, 2026-11-27).
        const std::string result = plan.update( marking( 2, "in_progress" ) );

        EXPECT_NE( result.find( "Step 1 is still in progress" ), std::string::npos ) << result;

        (void)plan.update( marking( 1, "done" ) );
        EXPECT_EQ( plan.update( marking( 2, "done" ) ).find( "still in progress" ), std::string::npos );
    }

    TEST( TaskPlan, SeveralStepsLeftInProgressAreNamedTogether )
    {
        TaskPlan plan;
        (void)plan.propose( R"({"steps":"One\nTwo\nThree\nFour"})" );
        (void)plan.update( marking( 1, "in_progress" ) );
        (void)plan.update( marking( 2, "in_progress" ) );

        const std::string result = plan.update( marking( 4, "done" ) );

        EXPECT_NE( result.find( "Steps 1 and 2 are still in progress" ), std::string::npos ) << result;
    }

    TEST( TaskPlan, APlanIsFinishedWhenEveryStepIsDoneOrSkipped )
    {
        TaskPlan plan;
        EXPECT_FALSE( plan.finished() );

        (void)plan.propose( R"({"steps":"One\nTwo"})" );
        (void)plan.update( marking( 1, "done" ) );
        EXPECT_FALSE( plan.finished() );

        (void)plan.update( marking( 2, "skipped" ) );
        EXPECT_TRUE( plan.finished() );
    }
}
