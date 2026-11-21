/**
 * @file Desktop.Understanding.ixx
 * @brief A project's *What Mila understands*: the note `note_understanding` writes, the one waiting for the person, and
 *        what a new chat in the project is given (AssistantInterface.md 4.5, 5.6 and 6).
 */

export module Desktop.Understanding;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    /// The most a note may hold, set to keep what each new chat is given small; the measurement of phase 7 sets them.
    export inline constexpr std::size_t kSummaryCharacters = 1200;
    export inline constexpr std::size_t kMostQuestions = 5;
    export inline constexpr std::size_t kQuestionCharacters = 200;
    export inline constexpr std::size_t kNextCharacters = 200;

    /**
     * @brief note_understanding's arguments as a note, or why not.
     *
     * `summary` is required; `questions` is one a line; `next` is one sentence. A note over a limit is refused, so the
     * model writes it again shorter rather than having it cut.
     *
     * @param derived_from The chats the note rests on: the chat writing it, and those the note it was given rested on.
     * @param exposed Whether the writing chat has read from outside, so the note waits for the person (5.6).
     * @return `note`, the record of AssistantInterface.md 8; or `result`, what the model reads of a refusal.
     */
    export nlohmann::json noteFromArguments( std::string_view arguments, const std::string& id, const std::string& conversation_id,
        const std::string& call_id, const nlohmann::json& derived_from, bool exposed, const std::string& model,
        const std::string& now );

    /**
     * @brief Take a note into the project's understanding: current, or waiting when it was written from a chat that
     *        had read from outside, replacing any note already waiting.
     *
     * @return `current` or `waiting`, where it went.
     */
    export std::string admitNote( nlohmann::json& understanding, nlohmann::json note );

    /**
     * @brief The person's answer to the waiting note named @p id: accepted, it becomes current and keeps its mark.
     *
     * @return False when no note of that id waits.
     */
    export bool decideWaiting( nlohmann::json& understanding, std::string_view id, bool accept, const std::string& now );

    /**
     * @brief The current note as the person edited it: the fields they changed, the rest kept, and the mark and what
     *        it rests on unchanged, since their words may still carry what it was drawn from.
     *
     * @param id The edited note's own id, so a chat that read it before the edit must read it again to rewrite it.
     * @return The note, or a string saying why it was refused.
     */
    export std::variant<nlohmann::json, std::string> editedNote( const nlohmann::json& current, const nlohmann::json& fields,
        const std::string& id, const std::string& now );

    /**
     * @brief A deleted chat takes every note that rests on it, current and waiting, since a paragraph cannot be unmixed
     *        (AssistantInterface.md 8, *Deleting*).
     *
     * @return The ids of the notes cleared.
     */
    export std::vector<std::string> clearDerivedFrom( nlohmann::json& understanding, std::string_view conversation_id );

    /// An empty understanding, as a project with none has.
    export nlohmann::json emptyUnderstanding();

    /**
     * @brief The text a chat in the project is given of its current note: quoted, named as a note, and when it was
     *        written and from which chat, so the model reads it as data and not as the person's words.
     *
     * @param chat_title The title of the chat the note was written in, or empty when it is not stored.
     */
    export std::string notesText( const nlohmann::json& note, std::string_view project_name, std::string_view chat_title );

    /**
     * @brief The chats every note given in this chat rested on, from its stored calls, so a note written here rests on
     *        them too.
     *
     * @param calls Each a stored call; one that gave a note carries `given` ({`id`, `derived_from`}).
     */
    export nlohmann::json givenDerivedFrom( const nlohmann::json& messages, const nlohmann::json& calls );

    /**
     * @brief Whether this chat was given the note @p id, from its stored calls: a note written here replaces the
     *        project's whole, so a chat that never read the current one would drop what it holds.
     */
    export bool noteGivenIn( const nlohmann::json& messages, const nlohmann::json& calls, std::string_view id );
}
