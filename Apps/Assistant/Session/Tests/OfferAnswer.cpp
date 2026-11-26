/**
 * @file OfferAnswer.cpp
 * @brief A person's message answers Mila's offer only when it is nothing but an answer (AssistantInterface.md 5.4).
 */

#include <gtest/gtest.h>
#include <string_view>

import Assistant.OfferAnswer;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    TEST( OfferAnswerTest, ABareAcceptanceTakesTheOfferWhateverItsCaseAndPunctuation )
    {
        for ( const std::string_view said : { "yes", "Yes, please!", "OK", "sure.", "please do", "Go ahead", "  yep  ", "Sounds good." } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Accept ) << said;
        }
    }

    TEST( OfferAnswerTest, ABareRefusalDeclinesItWithEitherApostrophe )
    {
        for ( const std::string_view said : { "no", "Not now", "No thanks.", "don't", "don\xE2\x80\x99t", "nope", "Maybe later" } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Decline ) << said;
        }
    }

    // Found in round 12's run on Gemma 4 12B: "No thanks, not now." to the setup's download went to the model, which said
    // it would carry on without it, and nothing was declined.
    TEST( OfferAnswerTest, AnswersOfOneKindTogetherAreThatAnswer )
    {
        for ( const std::string_view said : { "No thanks, not now.", "no, not yet", "Nope. Maybe later." } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Decline ) << said;
        }

        for ( const std::string_view said : { "Yes please, go ahead!", "OK, sounds good", "sure, do it" } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Accept ) << said;
        }
    }

    TEST( OfferAnswerTest, AnswersOfTwoKindsTogetherAreNoAnswer )
    {
        for ( const std::string_view said : { "yes, not now", "no, go ahead", "ok, undo" } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Other ) << said;
        }
    }

    TEST( OfferAnswerTest, KeepItSeparateAndUndoAreTheirOwnAnswer )
    {
        EXPECT_EQ( readOfferAnswer( "Keep it separate" ), OfferAnswer::KeepSeparate );
        EXPECT_EQ( readOfferAnswer( "undo" ), OfferAnswer::KeepSeparate );
        EXPECT_EQ( readOfferAnswer( "Put it back." ), OfferAnswer::KeepSeparate );
    }

    TEST( OfferAnswerTest, AnythingMoreIsAnOrdinaryMessageForMila )
    {
        for ( const std::string_view said : { "yes, but call it Garden", "", "What about the tomatoes?", "no idea what you mean",
            "ok so how do I water them", "yesterday" } )
        {
            EXPECT_EQ( readOfferAnswer( said ), OfferAnswer::Other ) << said;
        }
    }
}
