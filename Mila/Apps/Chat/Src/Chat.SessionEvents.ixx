/**
 * @file Chat.SessionEvents.ixx
 * @brief What a chat session reports to the front end driving it (ChatDesktop.md sections 4.1 and 7).
 */

export module Chat.SessionEvents;

import std;
import Chat.ChannelParser;
import Chat.Message;
import Chat.RoundStats;
import Chat.TextChannel;
import Chat.ToolDecision;

namespace Mila::ChatApp
{
    /**
     * @brief The session's side of the conversation with a front end: everything it would say, and the one question it asks.
     *
     * Every call arrives on the thread running the request that caused it -- a turn's events on the thread inside
     * ChatSession::respond(). The console implements this over its renderer; the session process (ChatDesktop.md
     * section 6.1) over the message schema.
     */
    export class SessionEvents
    {
    public:

        virtual ~SessionEvents() = default;

        /// A sentence about the session outside any reply: what a command did, or why it did nothing.
        virtual void info( std::string_view text ) = 0;

        /// Something the person asked for failed.
        virtual void error( std::string_view text ) = 0;

        /// Detail written only at detail level All: a loaded model's description and its memory.
        virtual void log( std::string_view text ) = 0;

        /// A model load starts; @p label says which. Every loadBegan() is followed by loadEnded(), thrown or not.
        virtual void loadBegan( std::string_view label ) = 0;
        virtual void loadEnded() = 0;

        /**
         * @brief An install's transfer moved: @p received of @p total bytes of the file now downloading.
         *
         * Called as often as the transfer reports, so a front end throttles what it shows. One install fetches
         * several files, so a smaller total, or a count that goes backwards, is the next file starting.
         */
        virtual void installProgress( std::uint64_t received, std::uint64_t total ) = 0;

        /**
         * @brief A reply starts.
         *
         * @param streams True when the reply's text arrives token by token through text(); false when it arrives
         *        whole, once per round.
         */
        virtual void turnBegan( bool streams ) = 0;

        /// One generate call starts. A turn has one more round than it has rounds of tool calls.
        virtual void roundBegan() = 0;

        /// One token generated, for a live count.
        virtual void tokenGenerated() = 0;

        /// Text to append to a channel of the reply.
        virtual void text( TextChannel channel, std::string_view text ) = 0;

        /// A reasoning span starts, or resumes after a tool call: what follows on the thought channel is a new piece.
        virtual void thoughtBreak() = 0;

        /// The reasoning span closed; what follows is answer text.
        virtual void thoughtClosed() = 0;

        /// The model began writing a tool call. Its syntax is not shown; the call is reported once it has been read.
        virtual void toolCallOpened() = 0;

        /// One generate call ended, with its measurements and why it stopped.
        virtual void roundEnded( const RoundStats& round ) = 0;

        /**
         * @brief The model called a tool the person has not allowed for this conversation: run it?
         *
         * The reply waits on the answer (ChatDesktop.md section 4.5).
         */
        virtual ToolDecision approveToolCall( const ToolCall& call ) = 0;

        /// A tool call was answered, by its tool or by the person declining it. The model reads @p result.
        virtual void toolCalled( const ToolCall& call, std::string_view result, bool is_error ) = 0;

        /**
         * @brief The reply ended.
         *
         * @param display Every round's reasoning and text, as the turn would be shown whole.
         * @param raw     The model's output verbatim, control tokens included.
         */
        virtual void turnEnded( const ParsedResponse& display, std::string_view raw ) = 0;
    };
}
