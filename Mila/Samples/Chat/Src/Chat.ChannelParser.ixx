/**
 * @file Chat.ChannelParser.ixx
 * @brief Splits a Gemma 4 channel-structured response into reasoning and answer.
 *
 * Per the Gemma 4 prompt-formatting protocol
 * (https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4), a thinking
 * response has the form:
 * @code
 *   <|channel>thought
 *   [internal reasoning]
 *   <channel|>[final answer]<turn|>
 * @endcode
 * The reasoning sits between <|channel> (and the "thought" label) and <channel|>;
 * the final answer follows <channel|>. ChannelParser separates the two so the
 * chat harness can surface or suppress the reasoning independently.
 */

module;
#include <string>
#include <string_view>
#include <array>

export module Chat.ChannelParser;

namespace Mila::ChatApp
{
    /**
     * @brief Result of splitting a response into its reasoning and answer channels.
     *
     * thinking holds the reasoning-channel text (empty when thinking is inactive
     * or the channel is empty). answer holds the user-facing final text and is the
     * only part stored in the conversation history.
     */
    export struct ParsedResponse
    {
        std::string thinking;
        std::string answer;
    };

    /**
     * @brief Separates a Gemma channel-structured response into reasoning and answer.
     *
     * A response with no kChannelOpen marker is returned verbatim as the answer
     * (the model emitted no channel structure), so the parser is a no-op for
     * non-channel output. When kChannelOpen is present:
     *   - the reasoning is the text between the "thought" label and kChannelClose,
     *   - the answer is the text after kChannelClose.
     * A kChannelOpen with no kChannelClose means generation stopped inside the
     * reasoning channel: the remainder is reasoning and the answer is empty.
     *
     * The delimiter strings are the single point of coupling to the model's chat
     * template. They match the documented Gemma 4 control tokens.
     *
     * Thread safety: parse() is stateless and safe to call concurrently.
     */
    export class ChannelParser
    {
    public:

        static ParsedResponse parse( const std::string& response )
        {
            const auto open = response.find( kChannelOpen );

            if ( open == std::string::npos )
                return { std::string{}, trim( response ) };

            const size_t header_start = open + kChannelOpen.size();
            const auto close = response.find( kChannelClose, header_start );

            // Any text before the channel structure is rare but is answer text.
            const std::string prefix = (open > 0) ? response.substr( 0, open ) : std::string{};

            if ( close == std::string::npos )
            {
                const std::string reasoning = stripChannelLabel(
                    response.substr( header_start ) );

                return { trim( reasoning ), trim( prefix ) };
            }

            const std::string reasoning = stripChannelLabel(
                response.substr( header_start, close - header_start ) );

            const std::string answer = prefix + response.substr( close + kChannelClose.size() );

            return { trim( reasoning ), trim( answer ) };
        }

    private:

        static constexpr std::string_view kChannelOpen = "<|channel>";
        static constexpr std::string_view kChannelClose = "<channel|>";

        /**
         * @brief Drop the leading channel-name label (e.g. "thought") from a region.
         *
         * The reasoning region opens with the channel name on its own line:
         * "thought\n[reasoning]". When the first line is a known channel label it
         * is removed; otherwise the region is returned unchanged.
         */
        static std::string stripChannelLabel( std::string_view region )
        {
            const auto first = region.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
                return std::string{};

            const auto newline = region.find( '\n', first );
            const std::string_view first_line = (newline == std::string_view::npos)
                ? region.substr( first )
                : region.substr( first, newline - first );

            if ( isChannelLabel( trimView( first_line ) ) )
            {
                return (newline == std::string_view::npos)
                    ? std::string{}
                    : std::string( region.substr( newline + 1 ) );
            }

            return std::string( region );
        }

        static bool isChannelLabel( std::string_view word )
        {
            constexpr std::array<std::string_view, 4> kLabels = {
                "thought", "thinking", "analysis", "reasoning" };

            for ( const auto label : kLabels )
            {
                if ( word == label )
                    return true;
            }

            return false;
        }

        static std::string_view trimView( std::string_view text )
        {
            const auto first = text.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
                return {};

            const auto last = text.find_last_not_of( " \t\r\n" );

            return text.substr( first, last - first + 1 );
        }

        static std::string trim( std::string_view text )
        {
            return std::string( trimView( text ) );
        }
    };
}
