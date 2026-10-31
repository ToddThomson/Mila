/**
 * @file ToolNeeds.ixx
 * @brief What a tool touches when it runs: files read or written, the network, and programs it runs.
 */

export module AI.ToolNeeds;

import std;

namespace Mila::AI
{
    /**
     * @brief A tool's declared needs (ToolCalling.md 3.1).
     *
     * Facts the application declares and the person is shown when asked, never hints to the model, which is not
     * shown them. A call's grant is narrowed to them: a tool that declares no files is given no folder.
     */
    export struct ToolNeeds
    {
        bool reads_files{ false };
        bool writes_files{ false };
        bool uses_network{ false };

        /// Runs programs through its grant's container (ContainedPrograms.md 4.1); what they reach is still the
        /// folders and network the other needs give.
        bool runs_programs{ false };
    };
}
