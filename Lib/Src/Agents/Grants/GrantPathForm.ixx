/**
 * @file GrantPathForm.ixx
 * @brief Which platform's form of path a grant reads (ToolCalling.md 3.4).
 */

export module Agents.GrantPathForm;

import std;

namespace Mila::Agents
{
    /**
     * @brief The form a grant accepts a path in, which is its platform's.
     *
     * Windows: a drive-letter path or a relative one, / or \ between names, and the names Windows reads as something
     * other than a file refused. Posix: a path from the root or a relative one, / between names, and macOS's resource
     * fork name refused. The build's own form is the grant's; both are named here so each is tested everywhere.
     */
    export enum class GrantPathForm
    {
        Windows,
        Posix
    };
}
