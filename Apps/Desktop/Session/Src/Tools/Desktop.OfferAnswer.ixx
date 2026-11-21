/**
 * @file Desktop.OfferAnswer.ixx
 * @brief What a person's message says to an offer Mila made in the conversation (AssistantInterface.md 5.4).
 */

export module Desktop.OfferAnswer;

import std;

namespace Mila::Desktop
{
    /**
     * @brief A message read as an answer to an offer, by app code and never by the model.
     *
     * Only a message that is nothing but an acceptance, a refusal or *keep it separate* answers; anything longer is
     * Other, and goes to Mila as an ordinary message.
     */
    export enum class OfferAnswer
    {
        Accept,
        Decline,
        KeepSeparate,
        Other,
    };

    /**
     * @brief The answer @p text gives: case folded, with the spaces and the punctuation around and between its words
     *        not counted, so *Yes, please!* accepts and *Yes, but call it Garden* is Other.
     */
    export OfferAnswer readOfferAnswer( std::string_view text );
}
