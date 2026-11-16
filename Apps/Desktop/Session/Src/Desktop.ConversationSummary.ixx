/**
 * @file Desktop.ConversationSummary.ixx
 * @brief What the home screen reads of a stored conversation: how far its plan came, how its last reply ended, and the
 *        files its calls wrote (Specifications/Apps/AssistantInterface.md 4.2).
 */

export module Desktop.ConversationSummary;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    /**
     * @brief A stored conversation's summary, as `conversations.list` carries it beside the conversation's id.
     *
     * Returns an object of three fields, each read from what the store already holds:
     * - `progress`: `{steps, finished, current}` for a plan, `current` the step it was at (the step in progress, else
     *   the first pending) or null once every step is done or skipped; null with no plan.
     * - `finish`: how the last reply ended (`turn.end`'s values), or null before any reply.
     * - `wrote`: `[{path, tool}]`, each file a call wrote or edited and that call did not fail, once each, the most
     *   recently written last. The path is as the call named it, relative to the conversation's folder.
     *
     * Anything the conversation holds in another shape is read as absent, so one malformed call hides nothing else.
     */
    export nlohmann::json summarizeConversation( const nlohmann::json& conversation );
}
