/**
 * @file Chat.ConsoleFrontEnd.ixx
 * @brief Chat at a terminal: reads commands and turns from standard input and draws the session's events.
 */

module;

// For SIGINT and SIG_DFL, which are macros and so are not part of the std module.
#include <csignal>

export module Chat.ConsoleFrontEnd;

import std;
import Chat.ParsedResponse;
import Chat.Footprint;
import Chat.ModelCatalog;
import Chat.Renderer;
import Chat.RichText;
import Chat.Session;
import Chat.Settings;
import Chat.StreamingDisplay;
import Chat.TextChannel;
import Chat.ToolDecision;
import Chat.Json;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    /**
     * @brief The console's front end over a ChatSession: the slash commands, the tables, and the live display.
     *
     * Everything the console prints is decided here, from the session's events and its state. Ctrl+C during a
     * reply stops that reply; anywhere else it ends the process, as it always has.
     */
    export class ConsoleFrontEnd final : public SessionEvents
    {
    public:

        explicit ConsoleFrontEnd( ChatConfig config )
            : session_( std::move( config ), *this )
        {
        }

        /// @copydoc ChatSession::registerTool
        void registerTool( std::string name, std::function<std::string( const std::string& )> handler )
        {
            session_.registerTool( std::move( name ), std::move( handler ) );
        }

        /**
         * @brief Answer one prompt and return, without opening a session.
         *
         * Standard output carries the answer and nothing else -- no banner, no spinner, no
         * session status -- so that `mila-chat -p "..." > answer.txt` yields an answer rather
         * than an answer wearing a welcome box.
         *
         * @param answer_out The real standard output. main.cpp has already pointed std::cout at
         *        standard error, because the diagnostics to be kept out of the answer include
         *        the library's own log records and those are not written through this class.
         *        Everything reaching std::cout below is therefore already diverted.
         *
         * @return An exit code from the contract in ChatConfiguration.md section 9.
         */
        int runOnce( const std::string& prompt, bool as_json, std::ostream& answer_out )
        {
            one_shot_ = true;
            renderer_.setQuiet( true );

            if ( config().model_name.empty() )
            {
                std::cerr << "No model is loaded. Name one with --model, "
                             "or install one with 'mila-store install <name>'.\n";

                return 4;
            }

            try
            {
                session_.loadConfiguredModel();
            }
            catch ( const std::exception& error )
            {
                std::cerr << std::format( "Could not load {}: {}\n",
                    config().model_name, error.what() );

                return 5;
            }

            session_.newConversation();
            respondTo( prompt );

            // The parsed answer is the final Assistant turn, so the last entry is the answer after any tool round
            // trip -- which is what a caller asked for, rather than the raw text with its channels still in it.
            const std::vector<ChatMessage>& history = session_.history();
            const std::string answer = history.empty() ? std::string{} : history.back().content;

            if ( as_json )
            {
                emitOneShotJson( answer, answer_out );
            }
            else
            {
                answer_out << answer << '\n';

                // A truncated answer is still an answer, so it is not a failure code. It is said
                // once on stderr, where it cannot corrupt the thing being piped.
                if ( session_.finishStatus() == GenerateStatus::MaxNewTokensReached )
                {
                    std::cerr << "Note: response hit the token cap without a stop token "
                                 "(finish: length).\n";
                }
            }

            return 0;
        }

        void run()
        {
            printBanner();

            if ( config().model_name.empty() )
            {
                // Said once, here: which model was configured and what the store does hold is a
                // startup fact, and it stops being true as soon as anything is installed.
                if ( !config().no_model_reason.empty() )
                {
                    renderer_.printInfo( config().no_model_reason );
                }

                reportNoModel();
            }
            else if ( !session_.open() )
            {
                reportNoModel();
            }

            printSessionStatus();

            session_.newConversation();

            std::string user_input;

            while ( true )
            {
                renderer_.printUserPrompt();

                // End of input ends the session, exactly as /exit does. getline clears the
                // string and sets failbit on EOF, so without this the empty-input continue
                // below spins the loop forever against a stream that will never yield again.
                if ( !std::getline( std::cin, user_input ) )
                {
                    renderer_.endUserPrompt();
                    break;
                }

                if ( user_input.empty() )
                    continue;

                if ( user_input.starts_with( '/' ) )
                {
                    std::string_view cmd{ user_input };
                    cmd.remove_prefix( 1 );

                    if ( cmd == "exit" )
                    {
                        break;
                    }

                    runCommand( cmd, user_input );
                    continue;
                }

                // Every generation path needs a model, so a turn with nothing resident says what
                // to do. The commands that fix it still work.
                if ( !session_.modelIsResident() )
                {
                    reportNoModel();
                    continue;
                }

                respondTo( user_input );
            }
        }

        // ── SessionEvents ─────────────────────────────────────────────────────

        void info( std::string_view text ) override
        {
            suspendStream();
            endTransferLine();
            renderer_.printInfo( text );
        }

        void error( std::string_view text ) override
        {
            suspendStream();
            endTransferLine();
            renderer_.printError( text );
        }

        void log( std::string_view text ) override
        {
            std::cout << text;
        }

        void loadBegan( std::string_view label ) override
        {
            // At detail All there is no spinner: its redraws and the log lines fight for the same line.
            if ( config().detail != DetailLevel::All )
            {
                renderer_.startSpinner( label );
            }
        }

        void loadEnded() override
        {
            renderer_.stopSpinner();
        }

        /// One line, redrawn in place: percent, bytes, and the time left once there is a rate to go on.
        void installProgress( std::uint64_t received, std::uint64_t total ) override
        {
            const auto now = std::chrono::steady_clock::now();

            // One callback serves every file in a pull, so this tracks which transfer is running as
            // well as how far along it is: a changed total, or a count that goes backwards, is the
            // next file starting rather than progress on this one.
            if ( !transfer_.active || total != transfer_.total
                || received < transfer_.first_received )
            {
                transfer_ = { total, received, now, -1, true };
            }

            const int percent = static_cast<int>( ( received * 100 ) / total );

            // One redraw per whole percent. Unthrottled this fired on every chunk, which at
            // 6.33 GB is tens of thousands of writes to a line that has 101 distinct states.
            if ( percent == transfer_.last_percent && received < total )
            {
                return;
            }

            transfer_.last_percent = percent;

            // Rate over what this run actually moved. A resumed transfer opens holding a
            // prefix it spent no time on, and counting those bytes makes the estimate a lie.
            const double elapsed =
                std::chrono::duration<double>( now - transfer_.started ).count();
            const std::uint64_t moved = received - transfer_.first_received;

            std::string remaining;

            if ( elapsed > 2.0 && moved > 0 && received < total )
            {
                const double rate = static_cast<double>( moved ) / elapsed;

                remaining = std::format( "  {} left", formatDuration(
                    static_cast<double>( total - received ) / rate ) );
            }

            const std::string line = std::format( "  {:>3}%  {}{}",
                percent, formatBytesOf( received, total ), remaining );

            // Erase to end of line rather than padding out to a fixed width. Padding writes
            // past the content, so on a terminal narrower than the pad the line wraps -- and
            // once it has wrapped, \r returns to the start of the last visual row, leaving
            // the earlier rows on screen for the next redraw to sit beside. That is what a
            // resize turned into fragments. Same idiom the renderer uses for the spinner.
            std::cout << "\r" << line << "\x1b[K" << std::flush;

            transfer_line_open_ = true;
        }

        void turnBegan( bool streams ) override
        {
            display_.reset();

            // Not in one shot: streaming paints tokens onto standard output as they arrive,
            // which is the whole value at a prompt and pure corruption of a piped answer.
            if ( streams && !one_shot_ )
            {
                display_ = std::make_unique<StreamingResponseDisplay>( renderer_, config().detail );
            }
        }

        void roundBegan() override
        {
            // Live token count on the spinner line: ticking = generating, frozen = hung.
            live_token_count_.store( 0, std::memory_order_relaxed );
            renderer_.startSpinner( {}, &live_token_count_ );
        }

        void tokenGenerated() override
        {
            live_token_count_.fetch_add( 1, std::memory_order_relaxed );

            // The library checks for a stop on each decode step, so a Ctrl+C seen here costs no latency.
            if ( interruptFlag().exchange( false ) )
            {
                session_.stop();
            }
        }

        void text( TextChannel channel, std::string_view text ) override
        {
            if ( display_ )
            {
                display_->text( channel, text );
            }
        }

        void thoughtBreak() override
        {
            if ( !display_ )
            {
                return;
            }

            if ( display_->showsThoughts() )
            {
                display_->thoughtBreak();
            }
            else
            {
                // Hidden reasoning streams nothing, so the spinner keeps the line: label it so the person
                // knows Mila is thinking. The first answer word stops it.
                renderer_.startSpinner( "Thinking...", &live_token_count_ );
            }
        }

        void thoughtClosed() override
        {
            if ( display_ )
            {
                display_->thoughtClosed();
            }
        }

        void toolCallOpened() override
        {
            if ( !display_ )
            {
                return;
            }

            // The display suspends for the suppressed tool span: the spinner owns the line until the round ends.
            display_->suspendForTrace();
            renderer_.startSpinner( "tool call", &live_token_count_ );
        }

        void roundEnded( const RoundStats& round ) override
        {
            renderer_.stopSpinner();

            // One shot says these on standard error instead, or in finish_reason.
            if ( one_shot_ )
            {
                return;
            }

            // A capped round is indistinguishable from a hang while a buffered response builds (pegged GPU,
            // silent spinner) -- say so explicitly.
            if ( round.finish_status == GenerateStatus::MaxNewTokensReached )
            {
                info( std::format(
                    "Response hit the {}-token cap without a stop token (finish: length).", round.token_budget ) );
            }
            else if ( round.finish_status == GenerateStatus::ContextOverflow )
            {
                info( "Response stopped at the context limit (finish: context_limit)." );
            }
            else if ( round.finish_status == GenerateStatus::ClientCancelled )
            {
                info( "Reply stopped." );
            }
        }

        /**
         * @brief Ask at the prompt line before a tool runs (ToolBroker.md section 5).
         *
         * Nothing but y or a runs the call: an empty answer, end of input and Ctrl+C all decline it, and Ctrl+C
         * also stops the reply. In one shot the question goes where everything else does, to standard error.
         */
        ToolDecision approveToolCall( const ToolCall& call ) override
        {
            renderer_.stopSpinner();
            suspendStream();

            const std::string summary = formatToolSummary(
                session_.tools(), call.name, call.arguments, "\x1b[1m", "\x1b[22m" );
            renderer_.printToolApproval( summary, call.arguments );

            std::string answer;

            while ( std::getline( std::cin, answer ) )
            {
                if ( interruptFlag().exchange( false ) )
                {
                    stopAtApproval();

                    return ToolDecision::Deny;
                }

                if ( const auto decision = parseToolDecision( answer ) )
                {
                    declined_ = *decision == ToolDecision::Deny;

                    return *decision;
                }

                renderer_.printToolApprovalChoices();
            }

            // A read cut short by Ctrl+C leaves the stream failed though more input may follow; at end of input the
            // next read fails again, so the session still ends there.
            std::cin.clear();
            renderer_.endUserPrompt();

            if ( interruptFlag().exchange( false ) )
            {
                stopAtApproval();
            }

            declined_ = true;

            return ToolDecision::Deny;
        }

        /**
         * @brief Render an answered tool call as an inline agentic-trace line.
         *
         * The human-readable intent (from the tool's summary template) always shows; the
         * raw result payload only at detail level All.
         */
        void toolCalled( const ToolCall& call, std::string_view result, bool is_error ) override
        {
            const bool declined = std::exchange( declined_, false );

            // A declined call is the person's own answer, not a failure to report.
            if ( is_error && !declined )
            {
                std::cerr << "\n[" << result << "]\n";
            }

            suspendStream();

            // ANSI bold on/off around each substituted argument value so the dynamic parts
            // stand out within the trace line; bold-off (not full reset) preserves the line
            // color the renderer applies around the whole summary.
            const std::string summary = formatToolSummary(
                session_.tools(), call.name, call.arguments, "\x1b[1m", "\x1b[22m" );

            renderer_.printToolCall( declined ? "Not run: " + summary : summary, result,
                config().detail == DetailLevel::All );
        }

        /**
         * @brief Show the reply, or check what streamed against it.
         *
         * The reasoning channel is rendered only at detail level Thoughts or above, and the verbatim raw output
         * at All; the final answer is always rendered. When the turn streamed live, the reply is not rendered
         * again: it is the oracle the streamed transcript is checked against (gate 1 of the streaming display).
         */
        void turnEnded( const ParsedResponse& display, std::string_view raw ) override
        {
            if ( display_ )
            {
                display_->finish();
            }

            if ( config().detail == DetailLevel::All )
                renderer_.printRaw( raw );

            // One shot renders nothing: the caller gets the answer on standard output in the
            // format it asked for, and a painted block would be in the middle of it.
            if ( one_shot_ )
            {
                display_.reset();

                return;
            }

            if ( display_ != nullptr && renderer_.streamHasOutput() )
            {
                validateStreamedDisplay( display );
            }
            else
            {
                if ( config().detail >= DetailLevel::Thoughts && !display.thinking.empty() )
                    renderer_.printThinking( display.thinking );

                renderer_.printMilaResponse( display.answer );
            }

            display_.reset();
        }

    private:

        const ChatConfig& config() const noexcept { return session_.config(); }

        const std::string& modelName() const noexcept { return session_.modelName(); }

        /// Close the live display's open line before an out-of-band line prints; nothing when none is open.
        void suspendStream()
        {
            if ( display_ )
            {
                display_->suspendForTrace();
            }
        }

        /// Ctrl+C at the tool question: no round is running to report the stop, so it is said here.
        void stopAtApproval()
        {
            session_.stop();
            info( "Reply stopped." );
        }

        /// The install's progress line ends before the next sentence, which is what the install said when it finished.
        void endTransferLine()
        {
            if ( transfer_line_open_ )
            {
                std::cout << "\n";
                transfer_line_open_ = false;
                transfer_.active = false;
            }
        }

        /// Set by the SIGINT handler, which may do nothing else; read on the turn's thread.
        static std::atomic<bool>& interruptFlag()
        {
            static std::atomic<bool> flag{ false };

            return flag;
        }

        static void onInterrupt( int )
        {
            interruptFlag().store( true );

            // The C runtime may restore the default handler before calling this one, which would let a second
            // Ctrl+C during the same reply end the process.
            std::signal( SIGINT, &ConsoleFrontEnd::onInterrupt );
        }

        /// One turn, with Ctrl+C stopping the reply instead of ending the process.
        void respondTo( const std::string& text )
        {
            interruptFlag().store( false );

            if ( !one_shot_ )
            {
                std::signal( SIGINT, &ConsoleFrontEnd::onInterrupt );
            }

            session_.respond( text );

            if ( !one_shot_ )
            {
                std::signal( SIGINT, SIG_DFL );
            }
        }

        void runCommand( std::string_view cmd, const std::string& user_input )
        {
            if ( cmd == "clear" )
            {
                session_.newConversation();
                renderer_.printInfo( "Conversation history cleared." );
                return;
            }
            if ( cmd == "help" )
            {
                printHelp();
                return;
            }
            if ( cmd == "stats" )
            {
                showTurnStats();
                return;
            }
            if ( cmd == "seed" || cmd.starts_with( "seed " ) )
            {
                if ( cmd == "seed" )
                {
                    renderer_.printInfo( "Usage: /seed <n>  (reseed the sampler; identical seed + prompt + history => identical tokens)." );
                    return;
                }

                const std::string_view arg = cmd.substr( 5 );
                std::uint64_t seed = 0;
                const auto result = std::from_chars( arg.data(), arg.data() + arg.size(), seed );

                if ( result.ec != std::errc{} )
                {
                    renderer_.printInfo( "Usage: /seed <n>  (n = non-negative integer)." );
                    return;
                }

                if ( !session_.seed( seed ) )
                {
                    reportNoModel();
                    return;
                }

                renderer_.printInfo( std::format( "Sampler reseeded ({}).", seed ) );
                return;
            }
            if ( cmd == "verbose" || cmd.starts_with( "verbose " ) )
            {
                if ( cmd == "verbose" )
                {
                    renderer_.printInfo( std::format(
                        "Detail: {}. Set with /verbose <off|thoughts|all>.",
                        detailLevelName( config().detail ) ) );
                    return;
                }

                const auto level = parseDetailLevel( cmd.substr( 8 ) );

                if ( !level )
                {
                    renderer_.printInfo( "Usage: /verbose <off|thoughts|all>." );
                    return;
                }

                setDetailLevel( *level );
                return;
            }
            if ( cmd == "effort" || cmd.starts_with( "effort " ) )
            {
                applyEffort( cmd );
                return;
            }

            if ( cmd == "context" || cmd.starts_with( "context " ) )
            {
                const std::vector<std::string_view> args =
                    cmd == "context" ? std::vector<std::string_view>{}
                                     : splitWhitespace( cmd.substr( 8 ) );

                if ( args.empty() )
                {
                    reportContext();
                    return;
                }

                if ( args.front() == "auto" )
                {
                    session_.setAutomaticContext();
                    return;
                }

                const std::string_view value = args.front();
                std::size_t length = 0;
                const auto result = std::from_chars(
                    value.data(), value.data() + value.size(), length );

                if ( result.ec != std::errc{}
                    || result.ptr != value.data() + value.size() || length == 0 )
                {
                    renderer_.printInfo(
                        "Usage: /context <n>|auto  (n = tokens; auto measures the largest "
                        "that fits this card)." );
                    return;
                }

                session_.setContextLength( length );
                return;
            }

            if ( cmd == "set" || cmd.starts_with( "set " ) )
            {
                const std::vector<std::string_view> args =
                    cmd == "set" ? std::vector<std::string_view>{}
                                 : splitWhitespace( cmd.substr( 4 ) );

                if ( args.empty() )
                {
                    reportSamplingSettings();
                    return;
                }

                if ( args.size() != 2 )
                {
                    renderer_.printInfo(
                        "Usage: /set <temperature|top_k|top_p|presence_penalty> <value|model>." );
                    return;
                }

                applySamplingSetting( args[ 0 ], args[ 1 ] );
                return;
            }

            if ( cmd == "thinking" || cmd.starts_with( "thinking " ) )
            {
                const std::vector<std::string_view> args =
                    cmd == "thinking" ? std::vector<std::string_view>{}
                                      : splitWhitespace( cmd.substr( 9 ) );

                applyThinking( args );
                return;
            }

            if ( cmd == "model" || cmd.starts_with( "model " ) )
            {
                const std::vector<std::string_view> args =
                    cmd == "model" ? std::vector<std::string_view>{}
                                   : splitWhitespace( cmd.substr( 6 ) );

                // A bare argument is a model NAME, not a verb, so the reserved words are a
                // closed set and everything else describes. That is what makes /model
                // <name> report rather than load: a load takes seconds, discards the KV
                // cache and is spelled out, so a half-remembered name costs a lookup
                // instead of a multi-gigabyte reload. It also means the noun form reads
                // the same way everywhere -- /model and /model <name> are one verb applied
                // to two objects, matching /context, /effort and /verbose.
                //
                // A model named `list` is unreachable. Repository names make that
                // implausible, and the alternative is a prefix nobody would type.
                if ( args.empty() )
                {
                    printModelInfo();
                }
                else if ( args.front() == "list" )
                {
                    listModelsCommand( std::span( args ).subspan( 1 ) );
                }
                else if ( args.front() == "load" )
                {
                    loadModelCommand( std::span( args ).subspan( 1 ) );
                }
                else if ( args.front() == "install" )
                {
                    installModelCommand( std::span( args ).subspan( 1 ) );
                }
                else if ( args.front() == "remove" )
                {
                    removeModelCommand( std::span( args ).subspan( 1 ) );
                }
                else
                {
                    describeModelCommand( args );
                }

                return;
            }

            renderer_.printInfo( std::format(
                "Unknown command: {}. Type /help for available commands.", user_input ) );
        }

        /**
         * @brief `/verbose <level>`: set the detail level, and say so.
         *
         * Warns when reasoning is requested while thinking mode is off, since nothing would be shown.
         */
        void setDetailLevel( DetailLevel level )
        {
            session_.setDetail( level );

            std::string message = std::format( "Detail set to {}.", detailLevelName( level ) );

            // The reasoning channel only exists when thinking mode is active. Named from the
            // resident model rather than a literal: model names come from the store, so any
            // spelled-out example goes stale the moment the store holds something else.
            if ( level >= DetailLevel::Thoughts && !config().show_thinking )
                message += std::format(
                    " (Thinking mode is off — enable it with /thinking on to see reasoning.)",
                    modelName() );

            renderer_.printInfo( message );
        }

        /// `/effort [1-5]`.
        void applyEffort( std::string_view cmd )
        {
            if ( !session_.templateHasEffortLevels() )
            {
                renderer_.printInfo( std::format(
                    "{}'s thinking has no effort levels: it is on or off (/thinking).", modelName() ) );
                return;
            }

            if ( cmd == "effort" )
            {
                const ThinkingEffort& e = thinkingEffort( config().thinking_effort );
                renderer_.printInfo( std::format(
                    "Thinking effort: {} ({}, ~{} tokens). Set with /effort <1-5>.",
                    config().thinking_effort, e.name, e.budget ) );
                return;
            }

            const std::string_view arg = cmd.substr( 7 );
            int level = 0;
            const auto result = std::from_chars( arg.data(), arg.data() + arg.size(), level );

            if ( result.ec != std::errc{} || !session_.setThinkingEffort( level ) )
            {
                renderer_.printInfo( "Usage: /effort <1-5>  (1 = minimal, 3 = balanced, 5 = exhaustive)." );
                return;
            }

            const ThinkingEffort& e = thinkingEffort( level );
            renderer_.printInfo( std::format(
                "Thinking effort set to {} ({}, ~{} tokens).", level, e.name, e.budget ) );
        }

        /**
         * @brief `/thinking [on|off]`.
         *
         * Its own command rather than an argument to a load. It rode on `/model <name> thinking`,
         * which needed a no-reload fast path to avoid a multi-gigabyte round trip for a boolean --
         * and once loading became explicit, `/model load <name> thinking` would have been a load
         * that deliberately does not load.
         */
        void applyThinking( std::span<const std::string_view> args )
        {
            if ( !config().thinking_capable )
            {
                renderer_.printInfo( "This model has no reasoning channel." );

                return;
            }

            if ( args.empty() )
            {
                renderer_.printInfo( std::format(
                    "Thinking: {}.", config().show_thinking ? "on" : "off" ) );

                return;
            }

            if ( args.size() != 1 || ( args.front() != "on" && args.front() != "off" ) )
            {
                renderer_.printInfo( "Usage: /thinking [on|off]." );

                return;
            }

            session_.setThinking( args.front() == "on" );

            renderer_.printInfo( std::format(
                "Thinking {}.", config().show_thinking ? "on" : "off" ) );
        }

        /**
         * @brief `/model list [--online]`.
         *
         * Installed is the default because it is the offline, instant answer, and because it is
         * the only one that says what can actually be loaded. The owner is hidden: --online means
         * the one Mila publishes into. An explicit owner still works, as the escape hatch for a
         * second publisher, but nothing advertises it.
         */
        void listModelsCommand( std::span<const std::string_view> args )
        {
            try
            {
                const auto hub_owner = [&]() -> std::string
                    {
                        const std::string argument( args.front() );

                        return ( argument == "--online" || argument == "online" )
                            ? std::string( Mila::Distribution::kDefaultHubOwner )
                            : argument;
                    };

                // Against the card's TOTAL memory rather than what is free.
                //
                // The listing answers "which of these could this machine run", and that is a
                // property of the card, not of this second. Free memory was tried and is wrong
                // here twice over: the resident model's own report understates what releasing it
                // returns (it excludes the 6-13% residual Gate B measured), and whatever the
                // desktop holds gets charged to every candidate. Both push the same way, and a 3B
                // was marked as not fitting on a card with room for three. The live picture
                // belongs on /model, which measures it directly.
                const DeviceMemoryInfo memory = queryDeviceMemory( config().device_index );

                std::optional<FootprintBudget> budget;

                // No device to ask means no column: a listing claiming "0 MB" would be stating a
                // measurement it does not have.
                if ( memory.total_bytes > 0 )
                {
                    FootprintBudget costed;

                    // Zero under auto, which asks the listing to answer each row at the largest
                    // context THAT model would get. Pricing every row at the resident model's
                    // auto-derived number is what made three of six rows warn falsely: Gemma
                    // affords 56320 because most of its layers are sliding-window, and no Llama
                    // would ever be given it. A context the user NAMED is different -- it really
                    // does apply to every row, because it is what loading any of them would use.
                    costed.fixed_context_length = config().context_is_automatic
                        ? dim_t{ 0 }
                        : static_cast<dim_t>( config().context_length );

                    costed.available_bytes = memory.total_bytes;
                    costed.device_name = queryDeviceName( config().device_index );
                    costed.device_index = config().device_index;
                    costed.resident_model =
                        session_.modelIsResident() ? modelName() : std::string{};

                    budget = std::move( costed );
                }

                if ( args.empty() )
                {
                    const ModelListing listing = describeInstalledModels( budget );

                    // Plain, like /help and /model: a table is the content the command was run
                    // to produce.
                    for ( const auto& line : listing.table )
                    {
                        std::cout << line << "\n";
                    }

                    std::cout << "\n";

                    for ( const auto& line : listing.notes )
                    {
                        renderer_.printInfo( "  " + line );
                    }

                    return;
                }

                // Capacity, matching the installed listing: the question is what this card can
                // run, not what is free while a model is resident.
                for ( const auto& line : describeHubModels(
                    hub_owner(), memory.total_bytes,
                    queryDeviceName( config().device_index ) ) )
                {
                    std::cout << line << "\n";
                }
            }
            catch ( const std::exception& error )
            {
                renderer_.printError( std::format(
                    "Could not list models: {}", error.what() ) );
            }
        }

        /**
         * @brief `/model load <name> [none|fp8|fp4|q4_0]`.
         *
         * Spelled out rather than reached by a bare name, because it is the one command here that
         * costs seconds and megabytes: it releases the resident weights, builds a new graph and
         * clears the conversation.
         */
        void loadModelCommand( std::span<const std::string_view> args )
        {
            if ( args.empty() )
            {
                renderer_.printInfo(
                    "Usage: /model load <name> [none|fp8|fp4|q4_0]." );

                return;
            }

            std::optional<QuantizationMode> requested_quantization;

            for ( std::size_t index = 1; index < args.size(); ++index )
            {
                const auto parsed = parseQuantization( args[ index ] );

                if ( !parsed )
                {
                    renderer_.printInfo( std::format(
                        "Unknown option '{}'. Use none, fp8, fp4 or q4_0.", args[ index ] ) );

                    return;
                }

                requested_quantization = *parsed;
            }

            session_.loadModel( args.front(), requested_quantization );
        }

        /// `/model install <name>`.
        void installModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model install <name> -- one name, as /model list --online shows it." );

                return;
            }

            session_.installModel( std::string( args.front() ) );
        }

        /// `/model remove <name>`.
        void removeModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model remove <name> -- one name, as /model list shows it." );

                return;
            }

            session_.removeModel( std::string( args.front() ) );
        }

        /// `/model <name>` -- the facts about one model, installed or only published.
        void describeModelCommand( std::span<const std::string_view> args )
        {
            if ( args.size() != 1 )
            {
                renderer_.printInfo(
                    "Usage: /model <name>, or /model list|load|install|remove." );

                return;
            }

            const std::string name = ChatSession::foldModelName( args.front() );

            // The resident model answers from the session instead, because the live deployment --
            // the context it was built at, what it is actually holding -- is strictly more than
            // the record can say, and it is the same model either way.
            if ( session_.isCurrentModel( name, std::nullopt ) )
            {
                printModelInfo();

                return;
            }

            try
            {
                for ( const auto& line : describeModel( name ) )
                {
                    std::cout << line << "\n";
                }
            }
            catch ( const std::exception& error )
            {
                renderer_.printError( std::format(
                    "Could not describe {}: {}", name, error.what() ) );
            }
        }

        /**
         * @brief Report the context, where it came from, and what auto would choose now.
         *
         * The last of those is what makes the command self-teaching: the number is on screen
         * before the user commits to it, so /context auto is never a leap in the dark.
         */
        void reportContext()
        {
            const std::string basis = config().context_is_automatic
                ? std::string( "auto" )
                : ( config().context_origin.empty()
                    ? std::string( layerName( SettingsLayer::FamilyInvariants ) )
                    : config().context_origin );

            std::cout << std::format( "  {:<16}{}\n", "Context window:", config().context_length );
            std::cout << std::format( "  {:<16}{}\n", "Set by:", basis );
            std::cout << std::format( "  {:<16}{}\n", "KV cache:", kvCacheName( config().kv_cache ) );

            const ContextFloor floor = session_.contextFloor();

            if ( floor.minimum > 0 )
            {
                std::cout << std::format( "  {:<16}{} (transcript {}, reasoning {}, answer {})\n",
                    "Minimum here:", floor.minimum, floor.transcript, floor.reasoning,
                    kAnswerHeadroomTokens );
            }

            if ( !session_.modelIsResident() )
            {
                return;
            }

            // From the plan this model's own load ran, not a fresh one: a plan made with the model
            // resident would describe a card that still holds it.
            const std::optional<DeploymentPlan>& plan = session_.deploymentPlan();

            if ( !plan || plan->contextLimitedBy() == DeploymentPlan::ContextLimit::FixedByCaller )
            {
                std::cout << std::format( "  {:<16}not measured (set /context auto to measure it)\n",
                    "Largest fit:" );

                return;
            }

            // Why it stopped short is worth one clause here, where the reader asked about context
            // specifically -- it is the one place the prefill bound is not noise.
            std::cout << std::format( "  {:<16}{}{}\n",
                "Largest fit:", plan->contextLength(),
                plan->contextLimitedBy() == DeploymentPlan::ContextLimit::FullPrefillChunk
                    ? "  (held back to keep a full prefill chunk)" : "" );
        }

        /// The sampling the next reply draws with, and which knobs are the user's rather than the model's.
        void reportSamplingSettings() const
        {
            const Dnn::SamplingParams sampling = sessionSampling( config() );

            const auto origin = []( bool yours ) { return yours ? "  (yours)" : ""; };

            std::cout << std::format( "  {:<18}{}{}\n", "temperature:", sampling.temperature,
                origin( config().temperature.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "top_k:", sampling.top_k,
                origin( config().top_k.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "top_p:", sampling.top_p,
                origin( config().top_p.has_value() ) );
            std::cout << std::format( "  {:<18}{}{}\n", "presence_penalty:", sampling.presence_penalty,
                origin( config().presence_penalty.has_value() ) );

            renderer_.printInfo(
                "  Set with /set <key> <value>, or /set <key> model for the model's own. temperature 0 is "
                "greedy, top_k 0 and top_p 1 each disable that filter." );
        }

        /**
         * @brief Set one sampling knob.
         *
         * The bounds are the sampler's rather than a matter of taste: a negative temperature or a
         * top_p outside [0,1] is not an adventurous setting, it is a value nothing downstream has
         * a meaning for. No reload -- these are read per generate call, unlike context.
         */
        void applySamplingSetting( std::string_view key, std::string_view value )
        {
            // Whole-field parses only. from_chars stops at the first character it cannot use, so
            // without the end test "0.8abc" and "40x" would both be accepted silently.
            const auto parseFloat = [value]( float& out ) -> bool
                {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), out );

                    return result.ec == std::errc{}
                        && result.ptr == value.data() + value.size();
                };

            const auto parseInt = [value]( int& out ) -> bool
                {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), out );

                    return result.ec == std::errc{}
                        && result.ptr == value.data() + value.size();
                };

            if ( value == "model" )
            {
                if ( key == "temperature" )
                {
                    session_.setTemperature( std::nullopt );
                }
                else if ( key == "top_k" )
                {
                    session_.setTopK( std::nullopt );
                }
                else if ( key == "top_p" )
                {
                    session_.setTopP( std::nullopt );
                }
                else if ( key == "presence_penalty" )
                {
                    session_.setPresencePenalty( std::nullopt );
                }
                else
                {
                    renderer_.printInfo( std::format(
                        "'{}' is not a setting. Use temperature, top_k, top_p or presence_penalty.", key ) );

                    return;
                }

                renderer_.printInfo( std::format( "{} follows the model.", key ) );

                return;
            }

            if ( key == "temperature" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < 0.0f || parsed > 5.0f )
                {
                    renderer_.printInfo( "Usage: /set temperature <0..5>  (0 is greedy)." );

                    return;
                }

                session_.setTemperature( parsed );
                renderer_.printInfo( std::format( "temperature set to {}.", parsed ) );

                return;
            }

            if ( key == "top_k" )
            {
                int parsed = 0;

                if ( !parseInt( parsed ) || parsed < 0 )
                {
                    renderer_.printInfo( "Usage: /set top_k <n>  (0 disables it)." );

                    return;
                }

                session_.setTopK( parsed );
                renderer_.printInfo( std::format( "top_k set to {}.", parsed ) );

                return;
            }

            if ( key == "top_p" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < 0.0f || parsed > 1.0f )
                {
                    renderer_.printInfo( "Usage: /set top_p <0..1>  (1 disables it)." );

                    return;
                }

                session_.setTopP( parsed );
                renderer_.printInfo( std::format( "top_p set to {}.", parsed ) );

                return;
            }

            if ( key == "presence_penalty" )
            {
                float parsed = 0.0f;

                if ( !parseFloat( parsed ) || parsed < -2.0f || parsed > 2.0f )
                {
                    renderer_.printInfo( "Usage: /set presence_penalty <-2..2>  (0 disables it)." );

                    return;
                }

                session_.setPresencePenalty( parsed );
                renderer_.printInfo( std::format( "presence_penalty set to {}.", parsed ) );

                return;
            }

            renderer_.printInfo( std::format(
                "'{}' is not a setting. Use temperature, top_k, top_p or presence_penalty.", key ) );
        }

        /**
         * @brief Say what to type next. Carries no reason: startup already gave it.
         *
         * The reason names the configured model and lists the store, which is worth saying once.
         * Repeating it on every prompt buries the one line that changes anything -- and it goes
         * stale the moment something is installed, while this does not.
         */
        void reportNoModel() const
        {
            renderer_.printInfo(
                "No model is loaded. /model list --online shows what can be installed, "
                "/model install <name> installs one, and /model list shows what is here." );
        }

        void printModelInfo() const
        {
            // Without a model the fields below are defaults, not facts -- printing them would
            // report a precision and a context for weights that are not there.
            if ( config().model_name.empty() )
            {
                reportNoModel();

                return;
            }

            const ThinkingEffort& effort = thinkingEffort( config().thinking_effort );

            std::cout << std::format( "  Model:          {}\n", modelName() );

            // Lineage before the deployment axes: a redistributed model's attribution belongs
            // beside its identity, not below its settings.
            if ( !config().base_model.empty() )
            {
                std::cout << std::format( "  Base model:     {}\n", config().base_model );
            }

            if ( !config().license.empty() )
            {
                std::cout << std::format( "  License:        {}\n", config().license );
            }

            const std::string_view attribution = requiredAttribution( config().license );

            if ( !attribution.empty() )
            {
                std::cout << std::format( "  Attribution:    {}\n", attribution );
            }

            // Whether it was derived, not the derivation. "auto, 11.99 GB device, held to a full
            // 1024-row prefill chunk" answers a question nobody in a chat session is asking; that
            // it was not a number they chose is the part that changes how they read it. The full
            // account still reaches a caller through the -p JSON payload's context_source.
            const std::string context_line = config().context_is_automatic
                ? std::format( "{} (auto)", config().context_length )
                : std::format( "{}", config().context_length );

            std::cout << std::format(
                "  Quantization:   {}\n"
                "  Context window: {}\n"
                "  Instruct:       {}\n",
                quantizationName( config().quantization_mode ),
                context_line,
                config().is_instruct ? "yes" : "no" );

            // Reported only by a model that has the channel. An effort level beside a model with
            // no reasoning mode describes a budget for tokens it will never emit.
            if ( config().thinking_capable )
            {
                std::cout << std::format( "  Thinking:       {}\n", config().show_thinking ? "on" : "off" );

                if ( session_.templateHasEffortLevels() )
                {
                    std::cout << std::format( "  Effort:         {} ({}, ~{} tokens)\n",
                        config().thinking_effort, effort.name, effort.budget );
                }
            }

            std::cout << std::format( "  Detail:         {}\n",
                detailLevelName( config().detail ) );

            // Measured off the device rather than taken from the model's own report: what is
            // actually resident includes the allocator rounding and lazily grown scratch that
            // a footprint cannot model, and that residual is the whole reason to look here.
            const DeviceMemoryInfo memory = queryDeviceMemory( config().device_index );

            if ( memory.total_bytes > 0 )
            {
                std::cout << std::format( "  VRAM:           {}\n",
                    formatBytesOf( memory.total_bytes - memory.free_bytes, memory.total_bytes ) );
            }
        }

        /**
         * @brief Show the per-round timing breakdown for the most recent turn on demand.
         *
         * A turn is one or more rounds; a tool call ends a round and starts another with its
         * own re-prefill. Each round is reported separately (prefill / decode throughput /
         * tokens, plus the tool it invoked) because a single lumped figure across rounds mixes
         * decode with re-prefill and is meaningless. A totals line closes multi-round turns.
         */
        void showTurnStats() const
        {
            const std::vector<RoundStats>& rounds = session_.lastTurnRounds();

            if ( rounds.empty() )
            {
                renderer_.printInfo( "No generation yet." );
                return;
            }

            std::vector<std::string> lines;
            const bool multi_round = rounds.size() > 1;

            lines.push_back( std::format( "  Last turn: {} round{}",
                rounds.size(), multi_round ? "s" : "" ) );

            int total_tokens = 0;
            float total_prefill_ms = 0.0f;

            for ( std::size_t i = 0; i < rounds.size(); ++i )
            {
                const RoundStats& round = rounds[ i ];
                total_tokens += round.tokens_generated;
                total_prefill_ms += round.prefill_time_ms;

                std::string line = std::format(
                    "  Round {}  prefill {:.0f} ms  \xe2\x94\x82  ",
                    i + 1, round.prefill_time_ms );

                if ( round.decode_tokens_per_second > 0.0f )
                    line += std::format( "{:.1f} tok/s  \xe2\x94\x82  {} tokens",
                        round.decode_tokens_per_second, round.tokens_generated );
                else
                    line += std::format( "{} token{}",
                        round.tokens_generated, round.tokens_generated == 1 ? "" : "s" );

                if ( round.ended_in_tool_call )
                    line += std::format( "  \xe2\x94\x82  \xe2\x86\x92 {}", round.tool_name );
                else if ( round.finish_status != GenerateStatus::Success )
                    line += std::format( "  \xe2\x94\x82  ended: {}", to_string( round.finish_status ) );

                lines.push_back( std::move( line ) );

                // Inter-token gap census (see RoundStats): outlier gaps localize a
                // mid-response stall inside generate(); uniform gaps exonerate the loop.
                if ( round.tokens_generated > 1 )
                {
                    std::string gap_line = std::format(
                        "           gaps med {:.1f} ms  \xe2\x94\x82  p99 {:.1f} ms  \xe2\x94\x82  max {:.1f} ms @ token {}",
                        round.median_gap_ms, round.p99_gap_ms,
                        round.max_gap_ms, round.max_gap_token_index );

                    if ( round.stall_count > 0 )
                        gap_line += std::format( "  \xe2\x94\x82  {} stall{} ({:.0f} ms)",
                            round.stall_count, round.stall_count == 1 ? "" : "s",
                            round.stall_total_ms );

                    lines.push_back( std::move( gap_line ) );
                }
            }

            if ( multi_round )
                lines.push_back( std::format(
                    "  Total     {} tokens  \xe2\x94\x82  {:.0f} ms prefill",
                    total_tokens, total_prefill_ms ) );

            renderer_.printStatsDetail( lines );
        }

        /**
         * @brief The one-shot answer as a JSON object on standard output.
         *
         * finish_reason is the field that earns this format: a response cut off at the token
         * cap reads as a complete answer in plain text and is self-announcing here.
         */
        void emitOneShotJson( const std::string& answer, std::ostream& answer_out ) const
        {
            int tokens = 0;
            float prefill_ms = 0.0f;

            for ( const RoundStats& round : session_.lastTurnRounds() )
            {
                tokens += round.tokens_generated;
                prefill_ms += round.prefill_time_ms;
            }

            nlohmann::json payload;
            payload[ "content" ] = answer;
            payload[ "model" ] = config().model_name;
            payload[ "context_length" ] = config().context_length;

            // The scripted half of the provenance the startup line prints. A caller that asked for
            // no context and got 83968 has the same right to know why as a reader of the banner.
            payload[ "context_source" ] = config().context_is_automatic ? "auto" : "configured";

            // What the plan was decided against and what it traded, so a script can reproduce the
            // choice: the free memory is the input that moves between runs on a card driving a display.
            if ( config().context_is_automatic && session_.deploymentPlan() )
            {
                const DeploymentPlan& plan = *session_.deploymentPlan();

                payload[ "context_measurement" ] = {
                    { "device", config().device_index },
                    { "device_total_bytes", plan.reading().total_bytes },
                    { "device_free_bytes", plan.reading().free_bytes },
                    { "prefill_chunk_rows", plan.prefillChunkRows() },
                    { "unconstrained_chunk_rows", plan.prefillChunking().unconstrained_chunk_rows },
                    { "bounded_by_prefill", plan.contextLimitedBy() == DeploymentPlan::ContextLimit::FullPrefillChunk },
                    { "context_limit", std::string( DeploymentPlan::nameOf( plan.contextLimitedBy() ) ) } };
            }

            payload[ "tokens_generated" ] = tokens;

            // The interactive session reads this off /stats; a scripted caller measuring a prompt
            // has only wall time without it, which carries the model load and the tokenizer too.
            payload[ "prefill_ms" ] = prefill_ms;
            payload[ "rounds" ] = session_.lastTurnRounds().size();
            payload[ "finish_reason" ] = finishReasonName( session_.finishStatus() );

            answer_out << payload.dump( 2 ) << '\n';
        }

        /**
         * @brief Wire names for the generation outcome, in the vocabulary a caller expects.
         */
        static const char* finishReasonName( GenerateStatus status )
        {
            switch ( status )
            {
                case GenerateStatus::MaxNewTokensReached: return "length";
                case GenerateStatus::ContextOverflow:     return "context_limit";
                default:                                  return "stop";
            }
        }

        /**
         * @brief Gate 1 of the streaming display: the streamed transcript must
         *        equal the buffered render of the same response.
         *
         * Line-exact comparison against the shared formatRich + wordWrap pipeline
         * (built at the wrap width the stream captured); when the stream had
         * forced visual breaks (tool trace lines interleaving the block) the
         * comparison falls back to whitespace-normalized text. Any divergence is
         * a bug in the incremental path and is reported loudly.
         */
        void validateStreamedDisplay( const ParsedResponse& parsed ) const
        {
            const int width = renderer_.streamWrapWidth();

            validateStreamedChannel( "answer", renderer_.streamedAnswerLines(),
                RichText::wordWrap( RichText::formatRich( parsed.answer ), width ) );

            if ( config().detail >= DetailLevel::Thoughts )
                validateStreamedChannel( "thinking", renderer_.streamedThinkingLines(),
                    RichText::wordWrap( RichText::formatRich( parsed.thinking ), width ) );
        }

        void validateStreamedChannel( std::string_view channel,
            const std::vector<std::string>& streamed,
            const std::vector<std::string>& oracle ) const
        {
            if ( !renderer_.streamForcedBreak() )
            {
                if ( streamed == oracle )
                    return;

                std::size_t line = 0;
                const std::size_t common = std::min( streamed.size(), oracle.size() );

                while ( line < common && streamed[ line ] == oracle[ line ] )
                    ++line;

                renderer_.printInfo( std::format(
                    "[stream validator] {} display diverged from the buffered render "
                    "(first difference at line {}; {} streamed / {} buffered lines).",
                    channel, line + 1, streamed.size(), oracle.size() ) );
                return;
            }

            if ( normalizeWhitespace( joinLines( streamed ) ) != normalizeWhitespace( joinLines( oracle ) ) )
                renderer_.printInfo( std::format(
                    "[stream validator] {} display diverged from the buffered render.",
                    channel ) );
        }

        static std::string joinLines( const std::vector<std::string>& lines )
        {
            std::string joined;

            for ( const auto& line : lines )
            {
                joined += line;
                joined += '\n';
            }

            return joined;
        }

        /// Collapse whitespace runs to single spaces and trim the ends, so text
        /// comparison survives the extra line breaks a forced suspend introduces.
        static std::string normalizeWhitespace( std::string_view text )
        {
            std::string out;
            out.reserve( text.size() );
            bool in_whitespace = true;

            for ( const char c : text )
            {
                if ( c == ' ' || c == '\t' || c == '\n' || c == '\r' )
                {
                    in_whitespace = true;
                    continue;
                }

                if ( in_whitespace && !out.empty() )
                    out += ' ';

                in_whitespace = false;
                out += c;
            }

            return out;
        }

        /// y or yes runs the call once, a or all for the conversation; empty, n or no declines; anything else is no answer.
        static std::optional<ToolDecision> parseToolDecision( std::string_view answer )
        {
            std::string word;

            for ( const char c : answer )
            {
                if ( c != ' ' && c != '\t' && c != '\r' )
                    word += static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
            }

            if ( word == "y" || word == "yes" )
                return ToolDecision::AllowOnce;

            if ( word == "a" || word == "all" )
                return ToolDecision::AllowConversation;

            if ( word.empty() || word == "n" || word == "no" )
                return ToolDecision::Deny;

            return std::nullopt;
        }

        static std::vector<std::string_view> splitWhitespace( std::string_view text )
        {
            std::vector<std::string_view> tokens;

            std::size_t i = 0;

            while ( i < text.size() )
            {
                while ( i < text.size() && text[ i ] == ' ' )
                    ++i;

                if ( i >= text.size() )
                    break;

                const std::size_t start = i;

                while ( i < text.size() && text[ i ] != ' ' )
                    ++i;

                tokens.push_back( text.substr( start, i - start ) );
            }

            return tokens;
        }

        void printBanner() const
        {
            renderer_.printWelcomeBox( std::format( "Mila Chat v{}", getAPIVersion().toString() ) );
        }

        /**
         * @brief What the session actually ended up with, printed after the load rather than before.
         *
         * The config names the model that *will* be loaded as readily as the one that is, so a
         * status line printed ahead of the load asserts as fact what the spinner on the next line
         * then shows to be still in progress -- and spends the model's name twice to do it.
         * Printed here it is a statement about weights that are on the device.
         */
        void printSessionStatus() const
        {
            if ( session_.modelIsResident() )
            {
                // "(auto)" says the number was not one they chose, which changes how they read it.
                // How it was derived does not, and is not shown anywhere.
                const std::string context_display = config().context_is_automatic
                    ? std::format( "{} (auto)", config().context_length )
                    : std::format( "{}", config().context_length );

                // The thinking clause is omitted entirely for a model without the channel, rather
                // than shown as "off" -- off reads as a setting the user could turn on.
                if ( config().thinking_capable )
                {
                    const std::string thinking_display = !config().show_thinking
                        ? "off"
                        : session_.templateHasEffortLevels() ? std::string( thinkingEffort( config().thinking_effort ).name ) : "on";

                    renderer_.printInfo( std::format( "  Model: {}  ·  Context window: {}  ·  Thinking: {}",
                        modelName(), context_display, thinking_display ) );
                }
                else
                {
                    renderer_.printInfo( std::format( "  Model: {}  ·  Context window: {}",
                        modelName(), context_display ) );
                }
            }

            std::cout << "  Type /help for commands, /exit to quit.\n\n";
        }

        void printHelp() const
        {
            std::cout << R"(
Available commands:
  /help                              Show this help message
  /clear                             Clear conversation history
  /model                             Show the loaded model
  /model <name>                      Show a model, installed or published
  /model list [--online]             List installed models, or what can be installed
  /model load <name> [quant]         Load a model (clears history). quant quantizes
                                     unquantized weights on load: none, fp8, fp4, q4_0.
  /model install <name>              Install a published model
  /model remove <name>               Remove an installed model and reclaim its blobs
  /context                           Show the context window and the largest that fits
  /context <n>|auto                  Set the context window (rebuilds, keeps the conversation)
  /thinking [on|off]                 Show or set the reasoning channel
  /set                               Show the sampling the next reply uses
  /set <key> <value>|model           Set temperature, top_k, top_p or presence_penalty,
                                     or return it to the model's own
  /effort [1-5]                      Show or set the thinking token-budget level
  /verbose [off|thoughts|all]        Show or set display detail (reasoning, raw + logs)
  /stats                             Show per-round timing for the last turn
  /seed <n>                          Reseed the sampler for reproducible generation
  /exit                              Exit the application

Models:         a bare /model <name> reports; loading is spelled out, because it takes
                seconds and clears the conversation.
Quantization:   a name ending -fp4/-fp8/-q4_0 is already quantized. For an unquantized model,
                /model load <name> fp4 quantizes it on load -- same weights, less
                VRAM, but the full file is still read.
Thinking:       Gemma's <|think|> mode. Toggling it does not reload weights. Effort
                (length) is set with /effort 1-5.
Sampling:       each model opens at its publisher's recommended setting, which can differ
                with thinking on and off. What you /set stays until you set it to model.
Context:        the primary VRAM lever, and it sizes buffers at build time -- so /context
                rebuilds the model, though the conversation survives. auto measures the
                largest that fits this card; a number you give is used exactly.
Detail:         tool calls always show as an agentic trace. /verbose thoughts adds the
                reasoning channel, all adds raw output + logging. The model reasons even
                when detail is off.

Examples:
)";

            // Named from the resident model rather than spelled out: a model name is whatever
            // the store holds, so a literal example stops resolving as soon as it holds
            // something else.
            std::cout << std::format( "  /model {}\n", modelName() );

            std::cout << R"(  /model list --online
  /thinking on
  /effort 5
  /context auto
  /set temperature 0.6
)" << "\n";
        }

        ConsoleRenderer renderer_;
        ChatSession session_;

        // The turn's live display, when the turn streams; created by turnBegan(), consumed by turnEnded().
        std::unique_ptr<StreamingResponseDisplay> display_;

        // Read by the spinner thread, written on the turn's thread.
        std::atomic<int> live_token_count_{ 0 };

        // Which file of an install is downloading, and how far along it is, for the progress line.
        struct TransferProgress
        {
            std::uint64_t total{ 0 };
            std::uint64_t first_received{ 0 };
            std::chrono::steady_clock::time_point started{};
            int last_percent{ -1 };
            bool active{ false };
        };

        TransferProgress transfer_;
        bool transfer_line_open_{ false };

        // Set when the person declines a call at the question, read by the toolCalled() that reports it.
        bool declined_{ false };

        // Set by runOnce: suppresses every display path, so standard output holds the answer
        // alone. Not a detail level -- detail says how much to show, this says show nothing.
        bool one_shot_{ false };
    };
}
