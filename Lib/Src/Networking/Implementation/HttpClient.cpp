/**
 * @file HttpClient.cpp
 * @brief The bodies of HttpClient and the URL helpers beside it.
 */

module Networking.HttpClient;

import std;

namespace Mila::Networking
{
    std::string hostOf( const std::string& url )
    {
        const auto scheme_end = url.find( "://" );
        const std::size_t start = ( scheme_end == std::string::npos ) ? 0 : scheme_end + 3;
        const auto host_end = url.find_first_of( "/?#", start );

        std::string authority = ( host_end == std::string::npos )
            ? url.substr( start )
            : url.substr( start, host_end - start );

        // Strip any userinfo and port so only the host compares.
        const auto at = authority.rfind( '@' );

        if ( at != std::string::npos )
        {
            authority = authority.substr( at + 1 );
        }

        const auto colon = authority.rfind( ':' );

        if ( colon != std::string::npos && authority.find( ']' ) == std::string::npos )
        {
            authority = authority.substr( 0, colon );
        }

        return authority;
    }

    std::string resolveRedirect( const std::string& base, const std::string& location )
    {
        if ( location.empty() )
        {
            return base;
        }

        // Absolute: has a scheme.
        if ( location.find( "://" ) != std::string::npos )
        {
            return location;
        }

        const auto scheme_end = base.find( "://" );
        const std::string scheme = ( scheme_end == std::string::npos )
            ? std::string( "https" )
            : base.substr( 0, scheme_end );

        // Protocol-relative: //host/path
        if ( location.starts_with( "//" ) )
        {
            return scheme + ":" + location;
        }

        const std::size_t authority_start = ( scheme_end == std::string::npos )
            ? 0
            : scheme_end + 3;
        const auto authority_end = base.find( '/', authority_start );

        const std::string origin = ( authority_end == std::string::npos )
            ? base
            : base.substr( 0, authority_end );

        // Root-relative: /path
        if ( location.front() == '/' )
        {
            return origin + location;
        }

        // Path-relative: resolve against the base's directory.
        std::string path = ( authority_end == std::string::npos )
            ? std::string( "/" )
            : base.substr( authority_end );

        const auto query = path.find_first_of( "?#" );

        if ( query != std::string::npos )
        {
            path = path.substr( 0, query );
        }

        const auto last_slash = path.rfind( '/' );

        path = ( last_slash == std::string::npos ) ? "/" : path.substr( 0, last_slash + 1 );

        return origin + path + location;
    }

    HttpClient::HttpClient( std::shared_ptr<const IHttpTransport> transport )
        : transport_( std::move( transport ) )
    {
        // Guarded here rather than at each consumer: makeDefaultHttpTransport never
        // returns null, so a null can only come from a caller passing one, and a crash
        // deep inside a redirect loop would say nothing about where it came from.
        if ( transport_ == nullptr )
        {
            throw std::runtime_error( "HttpClient requires a transport." );
        }
    }

    bool isWebAddress( std::string_view url )
    {
        const std::size_t scheme_end = url.find( "://" );

        if ( scheme_end == std::string_view::npos )
        {
            return false;
        }

        std::string scheme( url.substr( 0, scheme_end ) );

        for ( char& character : scheme )
        {
            character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
        }

        return scheme == "http" || scheme == "https";
    }

    HttpResult HttpClient::fetch(
        const HttpRequest& request,
        const SinkCallback& sink,
        const ProgressCallback& progress ) const
    {
        for ( const auto& header : request.headers )
        {
            std::string name = header.name;

            for ( char& character : name )
            {
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
            }

            // A request's own headers go to every host it reaches; only the token is held to the first.
            if ( name == "authorization" )
            {
                throw std::invalid_argument( "a request's own headers may not carry Authorization; give the token instead" );
            }
        }

        HttpResult result;

        std::string url = request.url;
        const std::string origin_host = hostOf( url );

        if ( !isWebAddress( url ) )
        {
            result.final_url = url;
            result.message = std::format( "'{}' is not an http or https address", url );

            return result;
        }

        std::uint64_t bytes_received = 0;
        std::uint64_t content_length = 0;
        bool sink_failed = false;
        bool aborted = false;
        bool range_overrun = false;
        bool cut = false;

        const auto on_headers = [&content_length]( long, std::uint64_t length )
            {
                content_length = length;
            };

        const auto counting_sink =
            [&]( const char* data, std::size_t length ) -> bool
            {
                // A server that ignores a part's Range streams the whole file; it is stopped at the first byte
                // past the part, not buffered to the end.
                if ( request.last_byte
                    && request.resume_from + bytes_received + length > *request.last_byte + 1 )
                {
                    range_overrun = true;

                    return false;
                }

                // Past the bound the sink is given the bytes up to it and the transfer stops, however much the
                // transport delivered in one run. A body of exactly the bound is read whole and not cut.
                if ( request.maximum_body_bytes > 0 && bytes_received + length > request.maximum_body_bytes )
                {
                    const auto kept = static_cast<std::size_t>( request.maximum_body_bytes - bytes_received );

                    if ( kept > 0 && !sink( data, kept ) )
                    {
                        sink_failed = true;

                        return false;
                    }

                    bytes_received += kept;
                    cut = true;

                    return false;
                }

                if ( !sink( data, length ) )
                {
                    sink_failed = true;

                    return false;
                }

                bytes_received += length;

                if ( progress )
                {
                    // The total is the whole file, not this response: a resumed transfer
                    // reports a content-length covering only the remaining bytes.
                    const std::uint64_t total = ( content_length == 0 )
                        ? 0
                        : request.resume_from + content_length;

                    if ( !progress( request.resume_from + bytes_received, total ) )
                    {
                        aborted = true;

                        return false;
                    }
                }

                return true;
            };

        for ( int hop = 0; hop <= request.maximum_redirects; ++hop )
        {
            HttpFetch fetch;
            fetch.url = url;
            fetch.headers = request.headers;
            fetch.body = request.body;
            fetch.low_speed_timeout_seconds = request.low_speed_timeout_seconds;

            // THE RULE: auth travels only to the host the request started on.
            if ( !request.token.empty() && hostOf( url ) == origin_host )
            {
                fetch.headers.push_back( { "Authorization", "Bearer " + request.token } );
            }

            if ( request.last_byte )
            {
                fetch.headers.push_back(
                    { "Range", std::format( "bytes={}-{}", request.resume_from, *request.last_byte ) } );
            }
            else if ( request.resume_from > 0 )
            {
                fetch.headers.push_back(
                    { "Range", std::format( "bytes={}-", request.resume_from ) } );
            }

            const HttpResponse response = transport_->fetch( fetch, counting_sink, on_headers );

            result.http_code = response.http_code;
            result.bytes_received = bytes_received;
            result.content_length = content_length;
            result.final_url = url;

            if ( aborted )
            {
                result.status = HttpStatus::TransportError;
                result.message = "transfer aborted by progress callback";

                return result;
            }

            if ( range_overrun )
            {
                result.status = HttpStatus::RangeIgnored;
                result.message = "server ignored Range and sent more than the part asked for";

                return result;
            }

            if ( sink_failed )
            {
                result.status = HttpStatus::TransportError;
                result.message = "sink rejected data (write failed)";

                return result;
            }

            // Only a 2xx body reaches the sink, so a cut is a successful reply read to the bound.
            if ( cut )
            {
                result = classify( request, response, std::move( result ) );
                result.cut = true;

                return result;
            }

            if ( response.transport_failed )
            {
                result.status = HttpStatus::TransportError;
                result.message = response.message;

                return result;
            }

            if ( isRedirect( response.http_code ) )
            {
                if ( response.location.empty() )
                {
                    result.status = HttpStatus::RedirectRefused;
                    result.message = std::format(
                        "{} redirect without a Location header", response.http_code );

                    return result;
                }

                const std::string next = resolveRedirect( url, response.location );

                if ( request.body )
                {
                    result.status = HttpStatus::RedirectRefused;
                    result.message = std::format(
                        "{} redirect to {}, and a request with a body is not redirected", response.http_code, next );

                    return result;
                }

                if ( !isWebAddress( next ) )
                {
                    result.status = HttpStatus::RedirectRefused;
                    result.message = std::format(
                        "{} redirect to {}, which is not an http or https address", response.http_code, next );

                    return result;
                }

                if ( request.redirects_stay_on_site
                    && ( hostOf( next ) != origin_host || ( url.starts_with( "https:" ) && !next.starts_with( "https:" ) ) ) )
                {
                    result.status = HttpStatus::RedirectLeftSite;
                    result.redirect_target = next;
                    result.message = std::format( "{} redirect to {}, which leaves the site", response.http_code, next );

                    return result;
                }

                url = next;

                continue;
            }

            return classify( request, response, std::move( result ) );
        }

        result.status = HttpStatus::TooManyRedirects;
        result.message = std::format( "exceeded {} redirects", request.maximum_redirects );

        return result;
    }

    HttpResult HttpClient::fetchString( const HttpRequest& request, std::string& body ) const
    {
        body.clear();

        return fetch( request, [&body]( const char* data, std::size_t length )
            {
                body.append( data, length );

                return true;
            } );
    }

    bool HttpClient::isRedirect( long http_code )
    {
        return http_code == 301 || http_code == 302 || http_code == 303
            || http_code == 307 || http_code == 308;
    }

    HttpResult HttpClient::classify(
        const HttpRequest& request,
        const HttpResponse& response,
        HttpResult result )
    {
        switch ( response.http_code )
        {
            case 200:
                // A range was requested and the server answered with the whole file.
                // Appending would concatenate, so the caller must discard and restart.
                result.status = ( request.resume_from > 0 || request.last_byte )
                    ? HttpStatus::RangeIgnored
                    : HttpStatus::Ok;

                if ( result.status == HttpStatus::RangeIgnored )
                {
                    result.message = "server ignored Range and sent the whole file";
                }

                return result;

            case 206:
                result.status = HttpStatus::Ok;

                return result;

            case 416:
                // Only meaningful for a resume, where it means the offset is at or past the
                // end and the partial already holds every byte the resource has -- a state
                // the caller can settle with a digest. Without a Range it is a plain error.
                result.status = ( request.resume_from > 0 )
                    ? HttpStatus::RangeNotSatisfiable
                    : HttpStatus::TransportError;
                result.message = ( request.resume_from > 0 )
                    ? "416: the resume offset is at or past the end of the file"
                    : "unexpected HTTP status 416";

                return result;

            case 401:
                result.status = HttpStatus::Unauthorized;
                // Not keyed on request.token: the cross-host rule may have dropped it on
                // the hop that answered, so the client cannot say which case this is.
                result.message = "401: unauthorized -- no token, or one the server refused";

                return result;

            case 403:
                result.status = HttpStatus::Forbidden;
                result.message = "403: token is valid but the repository terms are not accepted";

                return result;

            case 404:
                result.status = HttpStatus::NotFound;
                result.message = "404: no such repository, revision or file";

                return result;

            default:
                result.status = ( response.http_code >= 500 )
                    ? HttpStatus::ServerError
                    : HttpStatus::TransportError;
                result.message = response.message.empty()
                    ? std::format( "unexpected HTTP status {}", response.http_code )
                    : std::format( "unexpected HTTP status {}: {}",
                        response.http_code, response.message );

                return result;
        }
    }
}
