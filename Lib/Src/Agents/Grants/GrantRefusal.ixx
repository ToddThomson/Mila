/**
 * @file GrantRefusal.ixx
 * @brief A call refused by its grant before anything was opened, sent or run.
 */

export module Agents.GrantRefusal;

import std;

namespace Mila::Agents
{
    /**
     * @brief The grant holds nothing the call asks for: no folder, folders only to read, no network, or no way to run
     *        programs (ToolCalling.md 3.3 and 3.4).
     *
     * Decided from the grant alone, so its words are the library's and the call read nothing from outside. A refusal
     * that looked at the disk, such as a file that is not there, is an ordinary `std::runtime_error`.
     */
    export class GrantRefusal : public std::runtime_error
    {
    public:

        using std::runtime_error::runtime_error;
    };
}
