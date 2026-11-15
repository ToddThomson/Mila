/**
 * @file HttpStatus.ixx
 * @brief The outcome of a transfer, by the fix each failure needs.
 */

export module Networking.HttpStatus;

import std;

namespace Mila::Networking
{
    /**
     * @brief Outcome of a transfer attempt, distinguishing failures that need different fixes.
     */
    export enum class HttpStatus
    {
        Ok,               ///< Transfer completed.
        Unauthorized,     ///< 401 -- no token, or the token is not valid.
        Forbidden,        ///< 403 -- token valid, but the repository's terms are not accepted.
        NotFound,         ///< 404 -- wrong coordinate, revision or file path.
        RangeIgnored,     ///< A range request drew a 200; the partial must be discarded.
        RangeNotSatisfiable, ///< 416 on a resume; the offset is at or past the end of the file.
        TransportError,   ///< Connection, TLS or I/O failure.
        ServerError       ///< 5xx.
    };

    export inline std::string toString( HttpStatus status )
    {
        switch ( status )
        {
            case HttpStatus::Ok:             return "Ok";
            case HttpStatus::Unauthorized:   return "Unauthorized";
            case HttpStatus::Forbidden:      return "Forbidden";
            case HttpStatus::NotFound:       return "NotFound";
            case HttpStatus::RangeIgnored:   return "RangeIgnored";
            case HttpStatus::RangeNotSatisfiable: return "RangeNotSatisfiable";
            case HttpStatus::TransportError: return "TransportError";
            case HttpStatus::ServerError:    return "ServerError";
            default:                         return "Unknown";
        }
    }
}
