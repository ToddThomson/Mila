/**
 * @file HttpReplyReader.cpp
 * @brief The bodies of HttpReplyReader's members; the interface is HttpReplyReader.ixx.
 */

module Distribution.HttpReplyReader;

import std;

namespace Mila::Distribution
{
    namespace
    {
        std::string lowered( std::string_view text )
        {
            std::string result( text );

            for ( char& character : result )
            {
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
            }

            return result;
        }

        std::string_view trimmed( std::string_view text )
        {
            while ( !text.empty() && ( text.front() == ' ' || text.front() == '\t' ) )
            {
                text.remove_prefix( 1 );
            }

            while ( !text.empty() && ( text.back() == ' ' || text.back() == '\t' ) )
            {
                text.remove_suffix( 1 );
            }

            return text;
        }

        /// A decimal Content-Length, refused when it is not one or does not fit.
        std::optional<std::uint64_t> parseLength( std::string_view text )
        {
            if ( text.empty() )
            {
                return std::nullopt;
            }

            std::uint64_t value = 0;
            const auto [ end, error ] = std::from_chars( text.data(), text.data() + text.size(), value );

            if ( error != std::errc() || end != text.data() + text.size() )
            {
                return std::nullopt;
            }

            return value;
        }
    }

    bool HttpReplyReader::read( std::string_view bytes, const BodyCallback& body )
    {
        while ( !bytes.empty() )
        {
            switch ( stage_ )
            {
                case Stage::Failed:
                    return false;

                case Stage::Complete:
                    return true;

                case Stage::Head:
                    if ( !readHead( bytes ) )
                        return false;

                    break;

                case Stage::BodyByLength:
                case Stage::ChunkData:
                {
                    const std::size_t taken = static_cast<std::size_t>( std::min<std::uint64_t>( remaining_, bytes.size() ) );
                    const std::string_view run = bytes.substr( 0, taken );
                    bytes.remove_prefix( taken );
                    remaining_ -= taken;

                    if ( remaining_ == 0 )
                    {
                        stage_ = stage_ == Stage::BodyByLength ? Stage::Complete : Stage::ChunkEnd;
                    }

                    if ( !deliver( run, body ) )
                        return false;

                    break;
                }

                case Stage::BodyToClose:
                {
                    const std::string_view run = bytes;
                    bytes = {};

                    if ( !deliver( run, body ) )
                        return false;

                    break;
                }

                case Stage::ChunkSize:
                    if ( !readChunkSize( bytes ) )
                        return false;

                    break;

                case Stage::ChunkEnd:
                    if ( !readChunkEnd( bytes ) )
                        return false;

                    break;

                case Stage::Trailers:
                    if ( !readTrailers( bytes ) )
                        return false;

                    break;
            }
        }

        return stage_ != Stage::Failed;
    }

    bool HttpReplyReader::finish()
    {
        if ( stage_ == Stage::Complete )
            return true;

        if ( stage_ == Stage::BodyToClose )
        {
            stage_ = Stage::Complete;

            return true;
        }

        if ( stage_ == Stage::Failed )
            return false;

        return fail( head_complete_
            ? "the connection closed before the reply's body ended"
            : "the connection closed before a reply was read" );
    }

    bool HttpReplyReader::headComplete() const noexcept
    {
        return head_complete_;
    }

    bool HttpReplyReader::complete() const noexcept
    {
        return stage_ == Stage::Complete;
    }

    bool HttpReplyReader::stopped() const noexcept
    {
        return stopped_;
    }

    long HttpReplyReader::status() const noexcept
    {
        return status_;
    }

    const std::string& HttpReplyReader::location() const noexcept
    {
        return location_;
    }

    std::uint64_t HttpReplyReader::contentLength() const noexcept
    {
        return content_length_;
    }

    const std::string& HttpReplyReader::error() const noexcept
    {
        return error_;
    }

    bool HttpReplyReader::fail( std::string reason )
    {
        stage_ = Stage::Failed;
        error_ = std::move( reason );
        pending_.clear();

        return false;
    }

    bool HttpReplyReader::readHead( std::string_view& bytes )
    {
        while ( !bytes.empty() )
        {
            const char character = bytes.front();
            bytes.remove_prefix( 1 );

            // Blank lines before a status line are passed over, as RFC 9112 section 2.2 allows.
            if ( pending_.empty() && ( character == '\r' || character == '\n' ) )
                continue;

            pending_.push_back( character );

            if ( pending_.size() > kMaximumHeadBytes )
                return fail( "the reply's head is longer than " + std::to_string( kMaximumHeadBytes ) + " bytes" );

            if ( pending_.ends_with( "\r\n\r\n" ) || pending_.ends_with( "\n\n" ) )
                return parseHead();
        }

        return true;
    }

    bool HttpReplyReader::parseHead()
    {
        std::vector<std::string_view> lines;
        std::string_view rest( pending_ );

        while ( !rest.empty() )
        {
            const std::size_t end = rest.find( '\n' );
            std::string_view line = rest.substr( 0, end );
            rest = end == std::string_view::npos ? std::string_view{} : rest.substr( end + 1 );

            if ( line.ends_with( '\r' ) )
            {
                line.remove_suffix( 1 );
            }

            if ( !line.empty() )
            {
                lines.push_back( line );
            }
        }

        if ( lines.empty() || !lines.front().starts_with( "HTTP/1." ) )
            return fail( "the reply does not begin with an HTTP/1 status line" );

        const std::string_view status_line = lines.front();
        const std::size_t space = status_line.find( ' ' );
        const std::string_view code = space == std::string_view::npos ? std::string_view{} : status_line.substr( space + 1, 3 );
        long status = 0;
        const auto [ end, error ] = std::from_chars( code.data(), code.data() + code.size(), status );

        const std::size_t after_code = space + 4;
        const bool code_ends = after_code >= status_line.size() || status_line[ after_code ] == ' ';

        if ( code.size() != 3 || error != std::errc() || end != code.data() + code.size() || !code_ends || status < 100 )
            return fail( "the reply's status line carries no status code" );

        std::optional<std::uint64_t> length;
        bool transfer_encoding = false;
        bool chunked = false;
        std::string location;

        for ( std::size_t index = 1; index < lines.size(); ++index )
        {
            const std::string_view line = lines[ index ];

            // A folded line (RFC 9112 section 5.2): refused rather than joined.
            if ( line.front() == ' ' || line.front() == '\t' )
                return fail( "the reply folds a header across lines" );

            const std::size_t colon = line.find( ':' );

            if ( colon == std::string_view::npos || colon == 0 )
                return fail( "the reply carries a header line with no name" );

            const std::string name = lowered( line.substr( 0, colon ) );
            const std::string_view value = trimmed( line.substr( colon + 1 ) );

            if ( name == "content-length" )
            {
                // A list of the same length is one length; two lengths are a reply that cannot be read safely.
                std::string_view values = value;

                while ( !values.empty() )
                {
                    const std::size_t comma = values.find( ',' );
                    const auto parsed = parseLength( trimmed( values.substr( 0, comma ) ) );
                    values = comma == std::string_view::npos ? std::string_view{} : values.substr( comma + 1 );

                    if ( !parsed || ( length && *length != *parsed ) )
                        return fail( "the reply's Content-Length is not one length" );

                    length = parsed;
                }
            }
            else if ( name == "transfer-encoding" )
            {
                transfer_encoding = true;
                const std::string codings = lowered( value );
                const std::size_t last = codings.rfind( ',' );
                chunked = trimmed( last == std::string::npos ? std::string_view( codings ) : std::string_view( codings ).substr( last + 1 ) ) == "chunked";
            }
            else if ( name == "location" )
            {
                location.assign( value );
            }
        }

        pending_.clear();

        if ( status < 200 )
        {
            // Switching protocols would hand the connection to another protocol, which nothing here speaks.
            if ( status == 101 )
                return fail( "the server switched protocols" );

            return true;
        }

        head_complete_ = true;
        status_ = status;
        location_ = std::move( location );

        if ( status == 204 || status == 304 )
        {
            stage_ = Stage::Complete;
        }
        else if ( transfer_encoding )
        {
            // A transfer coding overrides a length (RFC 9112 section 6.3); one not ending in chunked runs to the close.
            stage_ = chunked ? Stage::ChunkSize : Stage::BodyToClose;
        }
        else if ( length )
        {
            content_length_ = *length;
            remaining_ = *length;
            stage_ = remaining_ == 0 ? Stage::Complete : Stage::BodyByLength;
        }
        else
        {
            stage_ = Stage::BodyToClose;
        }

        return true;
    }

    bool HttpReplyReader::readLine( std::string_view& bytes, bool& ended )
    {
        ended = false;

        while ( !bytes.empty() )
        {
            const char character = bytes.front();
            bytes.remove_prefix( 1 );

            if ( character == '\n' )
            {
                if ( pending_.ends_with( '\r' ) )
                {
                    pending_.pop_back();
                }

                ended = true;

                return true;
            }

            pending_.push_back( character );

            if ( pending_.size() > kMaximumHeadBytes )
                return fail( "a line of the reply's chunked body is longer than " + std::to_string( kMaximumHeadBytes ) + " bytes" );
        }

        return true;
    }

    bool HttpReplyReader::readChunkSize( std::string_view& bytes )
    {
        bool ended = false;

        if ( !readLine( bytes, ended ) )
            return false;

        if ( !ended )
            return true;

        std::string_view line( pending_ );
        line = trimmed( line.substr( 0, line.find( ';' ) ) );
        std::uint64_t size = 0;
        const auto [ end, error ] = std::from_chars( line.data(), line.data() + line.size(), size, 16 );

        if ( line.empty() || error != std::errc() || end != line.data() + line.size() )
            return fail( "the reply's chunked body carries a chunk size that is not one" );

        pending_.clear();
        remaining_ = size;
        stage_ = size == 0 ? Stage::Trailers : Stage::ChunkData;

        return true;
    }

    bool HttpReplyReader::readChunkEnd( std::string_view& bytes )
    {
        bool ended = false;

        if ( !readLine( bytes, ended ) )
            return false;

        if ( !ended )
            return true;

        if ( !pending_.empty() )
            return fail( "a chunk of the reply's body runs past its size" );

        stage_ = Stage::ChunkSize;

        return true;
    }

    bool HttpReplyReader::readTrailers( std::string_view& bytes )
    {
        while ( !bytes.empty() )
        {
            const std::size_t before = bytes.size();
            bool ended = false;

            if ( !readLine( bytes, ended ) )
                return false;

            trailer_bytes_ += before - bytes.size();

            if ( trailer_bytes_ > kMaximumHeadBytes )
                return fail( "the reply's trailers are longer than " + std::to_string( kMaximumHeadBytes ) + " bytes" );

            if ( !ended )
                return true;

            const bool last = pending_.empty();
            pending_.clear();

            if ( last )
            {
                stage_ = Stage::Complete;

                return true;
            }
        }

        return true;
    }

    bool HttpReplyReader::deliver( std::string_view bytes, const BodyCallback& body )
    {
        if ( bytes.empty() || !body || body( bytes.data(), bytes.size() ) )
            return true;

        stopped_ = true;

        return false;
    }
}
