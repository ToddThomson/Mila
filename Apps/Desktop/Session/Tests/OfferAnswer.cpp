/**
 * @file OfferAnswer.cpp
 * @brief A person's message answers Mila's offer only when it is nothing but an answer (AssistantInterface.md 5.4).
 */

#include <gtest/gtest.h>
#include <string_view>

import Desktop.OfferAnswer;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

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
