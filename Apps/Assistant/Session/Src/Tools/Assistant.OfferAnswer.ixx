/**
 * @file Assistant.OfferAnswer.ixx
 * @brief What a person's message says to an offer Mila made in the conversation (AssistantInterface.md 5.4).
 */

export module Assistant.OfferAnswer;

import std;

namespace Mila::Assistant
{
    /**
     * @brief A message read as an answer to an offer, by app code and never by the model.
     *
     * Only a message that is nothing but acceptances, nothing but refusals or nothing but *keep it separate* answers --
     * *No thanks, not now* is two refusals -- and anything else, a word of another kind among them included, is Other,
     * and goes to Mila as an ordinary message.
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
