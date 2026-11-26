/**
 * @file OfferAnswer.cpp
 * @brief Reading a person's message as an answer to an offer.
 */

module Assistant.OfferAnswer;

import std;

namespace Mila::Assistant
{
    namespace
    {
        constexpr std::array<std::string_view, 24> kAcceptances = {
            "yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "please", "please do", "do it", "go ahead",
            "yes please", "yes do it", "ok do it", "okay do it", "sounds good", "make it", "alright", "all right",
            "good idea", "great", "perfect", "fine" };

        constexpr std::array<std::string_view, 16> kRefusals = {
            "no", "n", "nope", "nah", "no thanks", "no thank you", "not now", "not yet", "maybe later", "later",
            "dont", "no dont", "leave it", "no leave it", "not this time", "no not now" };

        constexpr std::array<std::string_view, 10> kKeepSeparate = {
            "keep it separate", "keep separate", "keep this separate", "keep it apart", "keep this chat separate",
            "undo", "undo it", "undo that", "move it back", "put it back" };

        // A typographic apostrophe, U+2019, which a phone or a Mac's text system types for '.
        constexpr std::string_view kRightSingleQuote = "\xE2\x80\x99";

        /// The words alone, case folded and one space apart; an apostrophe joins its word, so *don't* is *dont*.
        std::string normalized( std::string_view text )
        {
            std::string words;
            bool space = false;

            for ( std::size_t index = 0; index < text.size(); ++index )
            {
                const unsigned char byte = static_cast<unsigned char>( text[ index ] );

                if ( byte == '\'' )
                {
                    continue;
                }

                if ( text.substr( index ).starts_with( kRightSingleQuote ) )
                {
                    index += kRightSingleQuote.size() - 1;

                    continue;
                }

                if ( std::isalnum( byte ) != 0 || byte >= 0x80 )
                {
                    if ( space && !words.empty() )
                    {
                        words.push_back( ' ' );
                    }

                    words.push_back( static_cast<char>( std::tolower( byte ) ) );
                    space = false;
                }
                else
                {
                    space = true;
                }
            }

            return words;
        }

        /// Whether the words are one or more of the phrases, end to end: *no thanks, not now* is two refusals, and
        /// *yes please, go ahead* two acceptances, where a phrase of another kind or any other word makes it Other.
        bool among( std::string_view words, std::span<const std::string_view> phrases )
        {
            if ( words.empty() )
            {
                return false;
            }

            // reaches[ i ]: the words from i to the end are phrases alone.
            std::vector<bool> reaches( words.size() + 1, false );
            reaches[ words.size() ] = true;

            for ( std::size_t start = words.size(); start-- > 0; )
            {
                if ( start > 0 && words[ start - 1 ] != ' ' )
                {
                    continue;
                }

                for ( const std::string_view phrase : phrases )
                {
                    const std::size_t end = start + phrase.size();

                    if ( end <= words.size() && words.substr( start, phrase.size() ) == phrase
                        && ( end == words.size() ? reaches[ end ] : words[ end ] == ' ' && reaches[ end + 1 ] ) )
                    {
                        reaches[ start ] = true;

                        break;
                    }
                }
            }

            return reaches[ 0 ];
        }
    }

    OfferAnswer readOfferAnswer( std::string_view text )
    {
        const std::string words = normalized( text );

        if ( among( words, kAcceptances ) )
        {
            return OfferAnswer::Accept;
        }

        if ( among( words, kRefusals ) )
        {
            return OfferAnswer::Decline;
        }

        if ( among( words, kKeepSeparate ) )
        {
            return OfferAnswer::KeepSeparate;
        }

        return OfferAnswer::Other;
    }
}
