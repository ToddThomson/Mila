/**
 * @file Chat.ModelListing.ixx
 * @brief A rendered model listing, split by how it should be shown.
 */

export module Chat.ModelListing;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief A rendered listing, split by how it should be shown.
     *
     * Two kinds of line, and the eye wants them apart: the table is the content the command was
     * run to produce, and the notes are commentary on it.
     */
    export struct ModelListing
    {
        std::vector<std::string> table;
        std::vector<std::string> notes;
    };
}
