/**
 * @file Chat.Ansi.ixx
 * @brief Terminal colour escapes, in one place.
 *
 * Extracted from ConsoleRenderer once a second module needed to colour its own output: a table
 * that tints its cells and a renderer that tints its lines must agree on how an escape is
 * written, and two copies of the same three functions would be one concept under two names.
 *
 * Truecolour is assumed, as it was before the extraction -- main.cpp turns on virtual-terminal
 * processing on Windows, and nothing here checks for a terminal, so redirected output carries
 * the escapes. Colour must therefore never be the only signal a line is carrying.
 */

module;
#include <format>
#include <string>

export module Chat.Ansi;

namespace Mila::ChatApp
{
    // inline, matching every other small free function in the Chat modules: an exported
    // non-inline definition in a module interface leaves the consumer with an external to
    // resolve rather than a body to emit.
    export inline std::string bg( int r, int g, int b )
    {
        return std::format( "\x1b[48;2;{};{};{}m", r, g, b );
    }

    export inline std::string fg( int r, int g, int b )
    {
        return std::format( "\x1b[38;2;{};{};{}m", r, g, b );
    }

    export inline const char* reset()
    {
        return "\x1b[0m";
    }
}
