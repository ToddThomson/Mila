/**
 * @file Assistant.Places.ixx
 * @brief `open_place`'s work: the places of the workspace a person's words name, and what the model is told of the one
 *        it opened (Specifications/Apps/AssistantInterface.md 5.2).
 */

export module Assistant.Places;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /// The most places returned when several fit, for the model or the person to choose among.
    export constexpr std::size_t kMostPlacesOffered = 5;

    /// The most names a list's result gives the model.
    export constexpr std::size_t kMostNamesListed = 20;

    /**
     * @brief The places @p words name, each `{view, id, name}`, at most five.
     *
     * A view's own name (`workspace`, `projects`, `chats`, `activity`, `reach`, with a leading *all*, *my* or *the*)
     * opens that view. Otherwise a project or chat whose name is the words, case folded, is the one place; failing
     * that, every project and then every chat whose name holds each of the words but *the*, *a*, *my*, *project* and
     * *chat*, the most recently changed first. Nothing fits, and the result is empty.
     *
     * @param projects  The stored projects' records.
     * @param chats     `conversations.list`'s items: `id`, `title`, `updated`, `project`.
     * @param open_chat The chat asking, never a place it is taken to: its title is often the very words of the request.
     */
    export nlohmann::json findPlaces( std::string_view words, std::span<const nlohmann::json> projects, const nlohmann::json& chats,
        std::string_view open_chat = {} );

    /**
     * @brief What the model reads of the place opened: what it is and, for a list or a project, what it shows.
     *
     * A list gives each project's name and whether it waits on the person, or each chat's title and project, up to
     * twenty; in a chat that has read from outside (@p names_allowed false) it gives only how many. The chat
     * @p open_chat is the one asking, which waits on no one while its reply is written.
     */
    export std::string describePlace( const nlohmann::json& place, std::span<const nlohmann::json> projects,
        const nlohmann::json& chats, std::string_view open_chat, bool names_allowed );

    /// What the model reads when several places fit: each by kind and name, so it can ask the person which.
    export std::string severalPlaces( const nlohmann::json& places );

    /// The line the conversation keeps of where Mila went: *Opened Calculator app*.
    export std::string wentLine( const nlohmann::json& place );

    /// Whether a chat in `conversations.list` waits on the person: a plan stopped short, or a reply that ended unfinished.
    export bool waitsOnPerson( const nlohmann::json& chat );
}
