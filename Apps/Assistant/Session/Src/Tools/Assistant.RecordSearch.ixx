/**
 * @file Assistant.RecordSearch.ixx
 * @brief `find_records`' work: the passages of the person's past chats that hold their words, each with where it is
 *        (Specifications/Apps/AssistantInterface.md 5.2).
 */

export module Assistant.RecordSearch;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Assistant
{
    /// The most exchanges one call returns, the most recently changed chats first.
    export constexpr std::size_t kMostRecordsFound = 8;

    /// About how many characters of a message the passage around a match keeps.
    export constexpr std::size_t kPassageCharacters = 300;

    /**
     * @brief Whether a stored chat has read from outside (AssistantInterface.md 5.6): its stored exposure holds a
     *        source, or, stored before exposure was, any tool result is in its turns.
     */
    export bool chatIsExposed( const nlohmann::json& conversation );

    /**
     * @brief The exchanges of other chats that hold every one of @p words, case folded: a message of the person's with
     *        Mila's reply to it, searched together, never a call's result.
     *
     * Each match says its chat by title, its project, when the chat last changed, and the passage of each message
     * around the first word it holds, or its start where it holds none, so an answer found by its question is read. A
     * result holding a match from a chat that has read from outside is from outside, so it exposes the chat that reads
     * it; one holding only the person's and Mila's words in chats that have not is the program's own. With no match it
     * says so, and is the program's own.
     *
     * @param conversations Every stored chat, whole.
     * @param projects      The stored projects' records, for each match's project name.
     * @param open_chat     The chat asking, which is not searched.
     */
    export Mila::Agents::ToolResult findRecords( std::string_view words, std::span<const nlohmann::json> conversations,
        std::span<const nlohmann::json> projects, std::string_view open_chat );
}
