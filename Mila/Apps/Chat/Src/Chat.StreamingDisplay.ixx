/**
 * @file Chat.StreamingDisplay.ixx
 * @brief The console's live display of a streamed reply.
 *
 * Takes the session's channel events and feeds them, through an incremental rich-text formatter per channel, to
 * ConsoleRenderer's streaming block writer. The session's buffered parse of the same reply is the oracle the
 * console checks what this displayed against.
 */

export module Chat.StreamingDisplay;

import std;
import Chat.Config;
import Chat.IncrementalRichFormatter;
import Chat.Renderer;
import Chat.RichText;
import Chat.TextChannel;

namespace Mila::ChatApp
{
    /**
     * @brief One turn's live display: answer text streams into the solid block, reasoning dim-streams at detail
     *        Thoughts and above and is hidden at Off.
     */
    export class StreamingResponseDisplay
    {
    public:

        StreamingResponseDisplay( ConsoleRenderer& renderer, DetailLevel detail )
            : renderer_( renderer ), detail_( detail )
        {
            renderer_.streamBegin();
        }

        bool showsThoughts() const
        {
            return detail_ >= DetailLevel::Thoughts;
        }

        void text( TextChannel channel, std::string_view text )
        {
            if ( channel == TextChannel::Answer )
            {
                renderer_.streamText( false, answer_formatter_.feed( text ) );
            }
            else if ( showsThoughts() )
            {
                renderer_.streamText( true, thought_formatter_.feed( text ) );
            }
        }

        /// A new reasoning piece starts.
        void thoughtBreak()
        {
            if ( showsThoughts() )
                renderer_.streamThinkingBreak();
        }

        /// The reasoning span closed: what its formatter still holds is reasoning, so it is written now.
        void thoughtClosed()
        {
            if ( !showsThoughts() )
                return;

            const StyledText tail = thought_formatter_.flush();

            if ( !tail.empty() )
                renderer_.streamText( true, tail );
        }

        /// Close the visual line before out-of-band lines (tool trace, info) print.
        void suspendForTrace()
        {
            renderer_.streamSuspend();
        }

        /// Turn end: flush every stage and close the block.
        void finish()
        {
            if ( showsThoughts() )
            {
                const StyledText thought_tail = thought_formatter_.flush();

                if ( !thought_tail.empty() )
                    renderer_.streamText( true, thought_tail );
            }

            const StyledText answer_tail = answer_formatter_.flush();

            if ( !answer_tail.empty() )
                renderer_.streamText( false, answer_tail );

            renderer_.streamEnd();
        }

    private:

        ConsoleRenderer& renderer_;
        DetailLevel detail_;

        IncrementalRichFormatter answer_formatter_;
        IncrementalRichFormatter thought_formatter_;
    };
}
