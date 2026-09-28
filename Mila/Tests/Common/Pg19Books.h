/**
 * @file Pg19Books.h
 * @brief The PG-19 test split's books as the long-context tests read them (ModelFamilyParity.md 8.2 G2, 8.4 L3).
 */

#pragma once

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>

namespace Mila::Tests::Common
{
    /// Data/Datasets/PG19/raw/test; Data/Datasets/PG19/README.md says how to fetch it.
    inline std::filesystem::path pg19TestPath()
    {
        return std::filesystem::path( TEST_DATA_DIR ) / "Datasets" / "PG19" / "raw" / "test";
    }

    /// The first `characters` bytes of a book, or fewer if it is shorter. C stdio: an input-stream header in a TU
    /// that does `import Mila;` leaves std::basic_istream::sentry incomplete.
    inline std::string readBook( const std::filesystem::path& book, std::size_t characters )
    {
        std::FILE* book_file = std::fopen( book.string().c_str(), "rb" );

        if ( book_file == nullptr )
        {
            return {};
        }

        std::string text( characters, '\0' );
        text.resize( std::fread( text.data(), 1, text.size(), book_file ) );
        std::fclose( book_file );

        return text;
    }

    /**
     * @brief Join Gutenberg's 70-column wraps: a lone newline becomes a space, and a blank line stays a paragraph.
     *
     * The books are stored with CRLF line endings, which become LF first. Before 2026-09-28 they did not, so every
     * line break became "\r " and no paragraph survived; G2's results before that date read that text.
     */
    inline std::string joinWraps( const std::string& crlf )
    {
        std::string stored;
        stored.reserve( crlf.size() );

        for ( std::size_t index = 0; index < crlf.size(); ++index )
        {
            if ( crlf[ index ] != '\r' || index + 1 == crlf.size() || crlf[ index + 1 ] != '\n' )
            {
                stored.push_back( crlf[ index ] );
            }
        }

        std::string joined = stored;

        for ( std::size_t index = 0; index < joined.size(); ++index )
        {
            const bool wrap = stored[ index ] == '\n'
                && ( index == 0 || stored[ index - 1 ] != '\n' )
                && ( index + 1 == stored.size() || stored[ index + 1 ] != '\n' );

            if ( wrap )
            {
                joined[ index ] = ' ';
            }
        }

        return joined;
    }
}
