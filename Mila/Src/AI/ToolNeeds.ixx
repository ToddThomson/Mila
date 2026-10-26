/**
 * @file ToolNeeds.ixx
 * @brief What a tool touches when it runs: files read or written, the network, and the areas it works in.
 */

export module AI.ToolNeeds;

import std;

namespace Mila::AI
{
    /**
     * @brief A tool's declared needs (ToolCalling.md 3.1).
     *
     * Facts the person is shown when asked, as the application declared them; never hints to the model, which is
     * not shown them.
     */
    export struct ToolNeeds
    {
        bool reads_files{ false };
        bool writes_files{ false };
        bool uses_network{ false };

        /// The folders, sites or other areas it works in, as the person would name them.
        std::vector<std::string> areas;
    };
}
