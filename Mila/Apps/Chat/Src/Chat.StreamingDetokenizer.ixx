/**
 * @file Chat.StreamingDetokenizer.ixx
 * @brief Re-assembles per-token decode bytes into complete UTF-8 characters.
 */

export module Chat.StreamingDetokenizer;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief Holds an incomplete UTF-8 tail until the token that completes it arrives.
     *
     * Per-token decode() can split a multi-byte UTF-8 sequence across tokens;
     * printing the raw bytes would show replacement garbage. feed() returns only
     * the prefix ending on a complete character boundary and holds the
     * incomplete tail for the next token. flush() surrenders whatever is held
     * (round end), complete or not.
     */
    export class StreamingDetokenizer
    {
    public:

        std::string feed( std::string_view bytes )
        {
            held_.append( bytes );

            const std::size_t complete = completeUtf8PrefixLength( held_ );
            std::string out = held_.substr( 0, complete );
            held_.erase( 0, complete );

            return out;
        }

        std::string flush()
        {
            return std::exchange( held_, std::string{} );
        }

    private:

        static std::size_t completeUtf8PrefixLength( std::string_view text )
        {
            std::size_t i = text.size();
            int continuation_bytes = 0;

            while ( i > 0 && continuation_bytes < 3 )
            {
                const unsigned char c = static_cast<unsigned char>( text[ i - 1 ] );

                if ( (c & 0x80) == 0 )
                    return text.size();  // ASCII tail: everything is complete

                if ( (c & 0xC0) == 0xC0 )
                {
                    const std::size_t need =
                        (c & 0xE0) == 0xC0 ? 2 :
                        (c & 0xF0) == 0xE0 ? 3 :
                        (c & 0xF8) == 0xF0 ? 4 : 1;  // invalid lead byte: emit as-is
                    const std::size_t have = text.size() - (i - 1);

                    return have >= need ? text.size() : i - 1;
                }

                --i;  // continuation byte: keep scanning back for its lead
                ++continuation_bytes;
            }

            // Start of buffer or 3+ trailing continuation bytes without a lead:
            // not a valid in-progress sequence, emit as-is.
            return text.size();
        }

        std::string held_;
    };
}
