/**
 * @file Environment.ixx
 * @brief Reading an environment variable, once, safely.
 *
 * The only place in the library that calls std::getenv, and therefore the only file needing an
 * exemption from the MSVC CRT's deprecation of it. Callers receive an owned string or nothing,
 * so the borrowed pointer whose lifetime the warning is about never escapes this file.
 */

module;
// std::getenv is deprecated by the MSVC CRT and not by the C++ standard, and there is no
// portable non-deprecated replacement: _dupenv_s is MSVC, secure_getenv is glibc with different
// semantics. The hazard is the returned pointer's lifetime, which readEnvironmentVariable ends
// by copying before it returns. Confined to this file so nothing else has to claim the same.
#define _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include <optional>
#include <string>

export module Distribution.Environment;

namespace Mila::Distribution
{
    /**
     * @brief The value of an environment variable, or nothing.
     *
     * An empty value reads as absent. Every caller here is choosing a path or a credential from
     * a first-match-wins list, and an exported-but-empty variable means "not configured" in that
     * setting rather than "configured to the empty string".
     */
    export std::optional<std::string> readEnvironmentVariable( const char* name )
    {
        const char* const value = std::getenv( name );

        if ( value == nullptr || *value == '\0' )
        {
            return std::nullopt;
        }

        return std::string( value );
    }
}
