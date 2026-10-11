/**
 * @file Chat.ChannelParser.ixx
 * @brief Splits a channel-structured response into reasoning and answer.
 *
 * The delimiters are a parameter (ChannelDelimiters), so one parser serves every family that
 * separates the two: Gemma 4's labelled channel and Qwen 3.8's <think> span differ only in
 * their markers. The Gemma structure below is what the algorithm was written against.
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
     * @brief The markers one family wraps its reasoning in.
     *
     * Parameterized rather than hardcoded because the split is one concept across families and
     * only the delimiters differ: Gemma opens a labelled channel, Qwen opens a <think> span.
     * Two parsers would be two places for the same bug.
     */
    export struct ChannelDelimiters
    {
        std::string_view open;
        std::string_view close;

        /// Whether the region opens with a channel name on its own line. Gemma's does; Qwen's
        /// does not, and stripping one there would eat a reasoning line that happened to begin
        /// with the word "analysis".
        bool labelled{ true };

        /**
         * @brief Whether the channel can already be open when the response begins.
         *
         * Qwen's generation primer ends ON the opening marker, so the model's first token is
         * already inside the reasoning and the response carries a CLOSE with no OPEN. Without
         * this the parser finds no structure, hands the whole thing back as the answer, and the
         * reasoning is displayed as the reply -- the exact failure the split exists to prevent.
         *
         * False for Gemma, where a bare close marker is stray text rather than a boundary.
         */
        bool primer_opens_channel{ false };
    };

    /// Gemma 4's channel structure -- see the file header.
    export inline constexpr ChannelDelimiters kGemmaChannel{
        "<|channel>", "<channel|>", true, false };

    /// Qwen 3.8's reasoning span, registered in the checkpoint vocabulary.
    export inline constexpr ChannelDelimiters kQwenThinkSpan{
        "<think>", "</think>", false, true };

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
     * Multiple channel cycles can appear back-to-back with nothing but whitespace
     * between them (observed empirically with Gemma 4 tool calls: an empty warm-up
     * cycle followed by the real one once the tool exchange is stripped out) --
     * parse() consumes every adjacent cycle it finds, concatenating non-empty
     * reasoning pieces, rather than only the first. The chain stops at the first
     * cycle boundary followed by real (non-whitespace) text, which becomes the
     * answer.
     *
     * The delimiter strings are the single point of coupling to the model's chat
     * template. They match the documented Gemma 4 control tokens.
     *
     * Thread safety: parse() is stateless and safe to call concurrently.
     */
    export class ChannelParser
    {
    public:

        static ParsedResponse parse(
            const std::string& response, const ChannelDelimiters& channel = kGemmaChannel )
        {
            const auto first_open = response.find( channel.open );

            if ( first_open == std::string::npos )
            {
                // The channel was open before the first token, so the response opens inside the
                // reasoning: text up to the close is thinking, and an absent close means
                // generation stopped there -- the same reading an unterminated channel gets
                // below.
                if ( channel.primer_opens_channel )
                {
                    const auto close = response.find( channel.close );

                    return close == std::string::npos
                        ? ParsedResponse{ trim( response ), std::string{} }
                        : ParsedResponse{
                            trim( response.substr( 0, close ) ),
                            trim( response.substr( close + channel.close.size() ) ) };
                }

                return { std::string{}, trim( response ) };
            }

            // Any text before the first channel is rare but is answer text.
            const std::string prefix = (first_open > 0) ? response.substr( 0, first_open ) : std::string{};

            std::string thinking;
            size_t cursor = first_open;

            while ( true )
            {
                const size_t header_start = cursor + channel.open.size();
                const auto close = response.find( channel.close, header_start );

                if ( close == std::string::npos )
                {
                    appendThinking( thinking,
                        stripChannelLabel( response.substr( header_start ), channel.labelled ) );
                    return { trim( thinking ), trim( prefix ) };
                }

                appendThinking( thinking, stripChannelLabel(
                    response.substr( header_start, close - header_start ), channel.labelled ) );

                const size_t after_close = close + channel.close.size();
                const auto next_open = response.find( channel.open, after_close );

                const bool adjacent = next_open != std::string::npos
                    && isAllWhitespace( std::string_view( response ).substr( after_close, next_open - after_close ) );

                if ( !adjacent )
                    return { trim( thinking ), trim( prefix + response.substr( after_close ) ) };

                cursor = next_open;
            }
        }

        /**
         * @brief True when word is a recognized reasoning-channel label.
         *
         * Public because the streaming display's channel router must classify
         * label lines identically to this post-hoc parser (its validation oracle).
         */
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

    private:

        /**
         * @brief Drop the leading channel-name label (e.g. "thought") from a region.
         *
         * The reasoning region opens with the channel name on its own line:
         * "thought\n[reasoning]". When the first line is a known channel label it
         * is removed; otherwise the region is returned unchanged.
         */
        static std::string stripChannelLabel( std::string_view region, bool labelled )
        {
            if ( !labelled )
                return std::string( region );

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

        static std::string trim( std::string_view text )
        {
            return std::string( trimView( text ) );
        }

        /// Append a reasoning piece to the accumulator, joined by a blank line when both
        /// sides are non-empty. Empty pieces (a warm-up cycle with no content) are no-ops.
        static void appendThinking( std::string& accum, const std::string& piece )
        {
            if ( piece.empty() )
                return;

            if ( !accum.empty() )
                accum += "\n\n";

            accum += piece;
        }

        static bool isAllWhitespace( std::string_view text )
        {
            return text.find_first_not_of( " \t\r\n" ) == std::string_view::npos;
        }
    };
}
