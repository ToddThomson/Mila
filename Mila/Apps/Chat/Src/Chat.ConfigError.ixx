/**
 * @file Chat.ConfigError.ixx
 * @brief A configuration that parsed but cannot be honoured.
 */

export module Chat.ConfigError;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief A configuration that parsed but cannot be honoured.
     *
     * Separate from a malformed command line so the two can exit with different codes -- a caller can tell "you
     * typed it wrong" from "the file says something impossible".
     */
    export struct ConfigError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
}
