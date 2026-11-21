/**
 * @file Desktop.Offers.ixx
 * @brief The offers Mila makes in a chat, a project from `propose_project` or a move from `move_to_project`, kept on
 *        the call that made them (AssistantInterface.md 5.3, 5.4 and 6).
 */

export module Desktop.Offers;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    /**
     * @brief propose_project's offer: the project named, gathering the asking chat and the chats the model named.
     *
     * Each line of `chats` names one chat by its title or words from it, matched as open_place matches; a chat in a
     * project is never gathered, nor one that several fit. Nothing is offered for a name a project already has, from a
     * chat in a project, for chats already waiting on an answer to the same offer, or for chats all of which the person
     * declined a project for.
     *
     * @param chats `conversations.list`'s items, each with its `offers`.
     * @param open_chat The asking chat, which is gathered first.
     * @param names_allowed False in a chat that has read from outside: the model is told no title it did not write,
     *                      and the person answers in the window.
     * @return `offer` (`kind`, `state`, `name`, `purpose`, `chats` [{`id`, `title`}], `in_window`), or null when nothing
     *         was offered; and `result`, what the model reads.
     */
    export nlohmann::json offerProject( std::string_view arguments, const nlohmann::json& chats,
        std::span<const nlohmann::json> projects, std::string_view open_chat, bool names_allowed );

    /**
     * @brief The project move_to_project's words name, exactly one, as open_place matches a project.
     *
     * @return `project` ({`id`, `name`}), or `result`, what the model reads when none or several fit.
     */
    export nlohmann::json projectForMove( std::string_view arguments, std::span<const nlohmann::json> projects, bool names_allowed );

    /// Whether the person has said yes to Mila filing a chat into this project, read from its history: a move Mila
    /// suggested and the person took, a `chat.moved` whose `suggested` is `move`. A chat gathered into a project made
    /// from a suggestion (`suggested` `project`) is not one.
    export bool filingAllowed( std::span<const nlohmann::json> history );

    /// Every offer in a stored chat's messages, each with the `message` and `call_id` it is kept on, oldest first.
    export nlohmann::json offersIn( const nlohmann::json& messages );

    /**
     * @brief The offer the person's next message answers: one still open on a call of Mila's last reply, or a move
     *        Mila made and said there, which *keep it separate* undoes.
     *
     * @return `{message, call}`, the indices in @p messages and in that message's calls, or null.
     */
    export nlohmann::json pendingOffer( const nlohmann::json& messages );

    /// The offer kept on the call @p call_id, in place, or null when no call there has one.
    export nlohmann::json* offerOnCall( nlohmann::json& messages, std::string_view call_id );

    /**
     * @brief Whether a project made from a suggestion can still be undone: nothing has happened in it since but the
     *        moves that made it and Mila's offers, and no project has been made after it.
     */
    export bool projectUndoable( const nlohmann::json& project, std::span<const nlohmann::json> history,
        std::span<const nlohmann::json> projects );
}
