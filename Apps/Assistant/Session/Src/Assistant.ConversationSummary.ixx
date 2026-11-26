/**
 * @file Assistant.ConversationSummary.ixx
 * @brief What the workspace reads of a stored conversation: how far its plan came, how its last reply began and ended,
 *        the files its calls wrote and what left the computer (Specifications/Apps/AssistantInterface.md 10).
 */

export module Assistant.ConversationSummary;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /**
     * @brief A stored conversation's summary, as `conversations.list` carries it beside the conversation's id.
     *
     * Returns an object of eight fields, each read from what the store already holds:
     * - `progress`: `{steps, finished, current}` for a plan, `current` the step it was at (the step in progress, else
     *   the first pending) or null once every step is done or skipped; null with no plan.
     * - `finish`: how the last reply ended (`turn.end`'s values), or null before any reply.
     * - `reply_start`: the last reply's answer as its first line on a list would show it, white space run together, at
     *   most 160 bytes and never cut inside a character; null before any reply, or for one with no answer.
     * - `wrote`: `[{path, tool, time, written}]`, each file a call wrote or edited and that call did not fail, once
     *   each, the most recently written last, with the last write's time and the file it left (Assistant.Creations'
     *   fileRecord; null for a write stored before). The path is as the call named it, relative to the conversation's
     *   folder. The session replaces `written` with `disk` before the page sees it.
     * - `sent`: `[{kind, what, where, time}]`, each thing that left the computer from this chat, oldest first, at most
     *   the latest 50: a page read, a search, a command given the network, or a link the person opened. `time` is null
     *   for a call stored before calls had one. A call the person declined sent nothing and is left out.
     * - `kept`: `[{message, time, start}]`, each reply the person kept, by its index among the messages, when it was
     *   kept, and how it begins as `reply_start` does.
     * - `offers`: each offer Mila made on a call (Assistant.Offers' offersIn), with the `message` and `call_id` it is
     *   kept on, oldest first.
     * - `setup`: for the setup's chat, the names of what the setup has still to accomplish, empty once it is over
     *   (Assistant.SetupSteps' outstandingIn); null for every other chat.
     *
     * Anything the conversation holds in another shape is read as absent, so one malformed call hides nothing else.
     */
    export nlohmann::json summarizeConversation( const nlohmann::json& conversation );
}
