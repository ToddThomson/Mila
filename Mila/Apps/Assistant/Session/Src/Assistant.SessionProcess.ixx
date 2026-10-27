/**
 * @file Assistant.SessionProcess.ixx
 * @brief The Mila app's session process: a Mila::AI agent and its session, spoken to as JSON lines on stdio
 *        (MilaApp.md 6.1).
 */

export module Assistant.SessionProcess;

import std;
import nlohmann.json;
import Assistant.ConversationStore;
import Assistant.FileTools;
import Assistant.Models;
import Assistant.PromptFile;
import Assistant.PromptTool;
import Assistant.ResidentAgent;
import Assistant.RoundTimer;
import Assistant.SearchKeyFile;
import Assistant.Settings;
import Assistant.SettingsFile;
import Assistant.TaskPlan;
import Assistant.WebPageReader;
import Assistant.WebSearch;

import Mila;

namespace Mila::Assistant
{
    using Mila::Dnn::dim_t;
    using Mila::Dnn::GenerateStatus;
    using Mila::Dnn::WeightQuantization;
    using Mila::Dnn::Conversation::Role;
    using Mila::Dnn::Conversation::ToolCall;
    using Mila::Dnn::Conversation::Turn;
    using Mila::Deployment::DeploymentRequest;
    using Mila::Deployment::DeploymentRefusedError;
    using Mila::Deployment::InstalledModel;
    using Mila::AI::AgentEvent;
    using Mila::AI::ToolAnswer;

    /**
     * @brief Requests on their way from the reading thread to the session's.
     *
     * Closing it ends the session: pop() then returns nothing, and whatever was still waiting is dropped.
     */
    class RequestQueue
    {
    public:

        void push( nlohmann::json request )
        {
            {
                std::lock_guard<std::mutex> lock( mutex_ );
                requests_.push_back( std::move( request ) );
            }

            ready_.notify_one();
        }

        void close()
        {
            {
                std::lock_guard<std::mutex> lock( mutex_ );
                closed_ = true;
            }

            ready_.notify_one();
        }

        std::optional<nlohmann::json> pop()
        {
            std::unique_lock<std::mutex> lock( mutex_ );
            ready_.wait( lock, [this] { return closed_ || !requests_.empty(); } );

            if ( closed_ )
            {
                return std::nullopt;
            }

            nlohmann::json request = std::move( requests_.front() );
            requests_.pop_front();

            return request;
        }

    private:

        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<nlohmann::json> requests_;
        bool closed_{ false };
    };

    /**
     * @brief The Mila app's session: section 4.2's requests in, section 4.1's events out.
     *
     * One agent at a time over one session, which holds the activity's prompt, tools, rules, grant and limits and
     * the conversation, and keeps them when the model changes. Two threads: the one that called run() does every
     * piece of session work, model loads included, and a second reads requests, so a stop or a tool decision is
     * heard while a reply is being written. Every event is one line on the protocol stream, written under a lock
     * because both threads write.
     */
    export class SessionProcess final
    {
    public:

        /**
         * @param settings The settings as read and overridden by the command line; written back to @p settings_file
         *                 as each one changes.
         * @param problems What could not be read from the settings file, said once the session starts.
         * @param prompt_directory The activities' prompts: `ask.json`, `files.json`, `research.json` and `tasks.json`,
         *                         the last two each with a `-search` version for when a search key is set.
         */
        SessionProcess( Settings settings, SettingsFile settings_file, std::vector<std::string> problems, int device,
            std::filesystem::path conversation_directory, std::filesystem::path prompt_directory, std::ostream& protocol_out )
            : settings_( std::move( settings ) ), settings_file_( std::move( settings_file ) ),
              settings_problems_( std::move( problems ) ), device_( device ),
              store_( std::move( conversation_directory ) ), prompts_( std::move( prompt_directory ) ), out_( protocol_out ),
              search_key_file_( store_.directory().parent_path() / "search.json" )
        {
            session().onAsk( [this]( const Mila::AI::ToolQuestion& question ) { return askAboutCall( question ); } );
            session().withSampling( settings_.sampling );
            useDetail( settings_.detail );

            // A plan's change is shown at once, while the reply that made it is still running.
            registerTool( std::string( TaskPlan::kProposeName ), {}, [this]( const std::string& arguments, const Mila::AI::ToolGrant& )
                {
                    std::string result = plan_.propose( arguments );
                    sendState();

                    return result;
                } );

            registerTool( std::string( TaskPlan::kUpdateName ), {}, [this]( const std::string& arguments, const Mila::AI::ToolGrant& )
                {
                    std::string result = plan_.update( arguments );
                    sendState();

                    return result;
                } );

            Mila::AI::ToolNeeds network;
            network.uses_network = true;

            registerTool( std::string( WebPageReader::kName ), network,
                []( const std::string& arguments, const Mila::AI::ToolGrant& grant )
                {
                    return WebPageReader::read( arguments, grant );
                } );

            // Registered once; each call reaches the folder of the conversation it belongs to, through its grant.
            for ( const std::string_view name : FileTools::kNames )
            {
                registerTool( std::string( name ), FileTools::needsOf( name ),
                    [name]( const std::string& arguments, const Mila::AI::ToolGrant& grant )
                    {
                        return FileTools::run( name, arguments, grant );
                    } );
            }

            if ( const std::optional<std::string> key = search_key_file_.read() )
            {
                useSearchKey( *key );
            }
        }

        /**
         * @brief Open the model the settings name, then answer requests until @p requests ends.
         *
         * @return The process's exit code.
         */
        int run( std::istream& requests )
        {
            std::jthread reader( [this, &requests] { readRequests( requests ); } );

            for ( const std::string& problem : settings_problems_ )
            {
                sendError( problem, nullptr );
            }

            useActivity( "ask", {} );

            if ( !settings_.model.empty() )
            {
                openRemembered();
            }

            while ( const std::optional<nlohmann::json> request = queue_.pop() )
            {
                try
                {
                    handle( *request );
                }
                catch ( const std::exception& failure )
                {
                    // A request that fails says so and leaves the session answering the next.
                    work_ = Work::None;
                    busy_ = false;
                    sendError( std::format( "Something went wrong: {}", failure.what() ), nullptr );
                    sendState();
                }
            }

            return 0;
        }

    private:

        static constexpr std::string_view kStreamId = "stream-1";

        /// Ask & Explore's limits, numbers no measurement has set.
        static constexpr int kAskRounds = 4;
        static constexpr dim_t kAskReplyTokens = 2048;

        /// What the session's thread is doing, for an error to say where it happened.
        enum class Work
        {
            None,
            Turn,
            Load,
            Install,
            Settings,
        };

        /// A call waiting on the person, answered from the reading thread.
        struct Approval
        {
            std::string turn_id;
            std::string call_id;
            std::promise<ToolAnswer> answer;
        };

        using Handler = std::function<std::string( const std::string& arguments, const Mila::AI::ToolGrant& grant )>;

        Mila::AI::Session& session() noexcept
        {
            return model_.session();
        }

        const Mila::AI::Session& session() const noexcept
        {
            return model_.session();
        }

        // ---- The reading thread -----------------------------------------------------------

        /**
         * @brief Read one request per line until the input ends, which ends the session.
         *
         * A stop and a tool decision are acted on here, because the session's thread is inside the reply they are
         * about. A second turn while one is running is refused here, so that the refusal does not wait for it.
         */
        void readRequests( std::istream& requests )
        {
            std::string line;

            while ( std::getline( requests, line ) )
            {
                if ( !line.empty() && line.back() == '\r' )
                {
                    line.pop_back();
                }

                if ( line.empty() )
                {
                    continue;
                }

                nlohmann::json request;

                try
                {
                    request = nlohmann::json::parse( line );
                }
                catch ( const nlohmann::json::parse_error& )
                {
                    std::cerr << "Ignored a request line that is not JSON.\n";
                    continue;
                }

                if ( !request.is_object() || !request.contains( "type" ) || !request[ "type" ].is_string() )
                {
                    std::cerr << "Ignored a request with no type.\n";
                    continue;
                }

                const std::string type = request[ "type" ].get<std::string>();

                if ( type == "turn.stop" )
                {
                    stopReply();
                }
                else if ( type == "turn.tool_decision" )
                {
                    decideToolCall( request );
                }
                else if ( type == "turn.send" )
                {
                    if ( busy_.exchange( true ) )
                    {
                        sendError( "A reply is still being written. Stop it first, or wait for it to finish.", "turn" );
                        continue;
                    }

                    stop_asked_ = false;
                    queue_.push( std::move( request ) );
                }
                else
                {
                    queue_.push( std::move( request ) );
                }
            }

            queue_.close();
            stopReply();
        }

        void stopReply()
        {
            std::lock_guard<std::mutex> lock( approval_mutex_ );

            stop_asked_ = true;

            {
                std::lock_guard<std::mutex> stopping( stop_mutex_ );
                stop_.request_stop();
            }

            // The agent reads the stop before it runs the call, so this answer is never acted on.
            if ( approval_.has_value() )
            {
                approval_->answer.set_value( ToolAnswer::Deny );
                approval_.reset();
            }
        }

        void decideToolCall( const nlohmann::json& request )
        {
            const std::string turn_id = request.value( "turn_id", std::string{} );
            const std::string call_id = request.value( "call_id", std::string{} );
            const std::string decision = request.value( "decision", std::string{} );

            std::optional<ToolAnswer> parsed;

            if ( decision == "allow_once" )
            {
                parsed = ToolAnswer::Once;
            }
            else if ( decision == "allow_conversation" )
            {
                parsed = ToolAnswer::ForConversation;
            }
            else if ( decision == "deny" )
            {
                parsed = ToolAnswer::Deny;
            }

            std::lock_guard<std::mutex> lock( approval_mutex_ );

            if ( !parsed.has_value() || !approval_.has_value()
                || approval_->turn_id != turn_id || approval_->call_id != call_id )
            {
                std::cerr << "Ignored a tool decision for a call that is not waiting.\n";

                return;
            }

            approval_->answer.set_value( *parsed );
            approval_.reset();
        }

        // ---- The session's thread ---------------------------------------------------------

        void handle( const nlohmann::json& request )
        {
            const std::string type = request[ "type" ].get<std::string>();

            if ( type == "ready" )
            {
                sendState();
                sendConversations();
                sendModels( false );
            }
            else if ( type == "turn.send" )
            {
                respond( request );
            }
            else if ( type == "conversation.new" )
            {
                const std::string folder = request.contains( "folder" ) && request[ "folder" ].is_string()
                    ? request[ "folder" ].get<std::string>() : std::string{};

                if ( useActivity( request.value( "activity", std::string( "ask" ) ), folder ) )
                {
                    startNewConversation();
                    sendState();
                }
            }
            else if ( type == "conversation.open" )
            {
                openConversation( request.value( "id", std::string{} ) );
            }
            else if ( type == "conversation.rename" )
            {
                renameConversation( request.value( "id", std::string{} ), request.value( "title", std::string{} ) );
            }
            else if ( type == "conversation.delete" )
            {
                deleteConversation( request.value( "id", std::string{} ) );
            }
            else if ( type == "models.refresh" )
            {
                sendModels( request.value( "online", false ) );
            }
            else if ( type == "model.load" )
            {
                loadModel( request );
            }
            else if ( type == "model.install" )
            {
                installModel( request.value( "name", std::string{} ) );
            }
            else if ( type == "model.remove" )
            {
                removeModel( request.value( "name", std::string{} ) );
            }
            else if ( type == "settings.set" )
            {
                work_ = Work::Settings;
                changeSetting( request );
                work_ = Work::None;
                sendState();
            }
            else if ( type == "attachment.add" )
            {
                refuseAttachment( request );
            }
            else if ( type == "attachment.remove" || type == "link.open" )
            {
                // Nothing is held for an attachment, which is refused when added; a link is the window's.
            }
            else
            {
                std::cerr << std::format( "Ignored a request of type {}.\n", type );
            }
        }

        // ---- A turn -----------------------------------------------------------------------

        void respond( const nlohmann::json& request )
        {
            const std::string text = request.value( "text", std::string{} );

            if ( !model_.resident() )
            {
                busy_ = false;
                sendError( "No model is loaded. Choose one from the model menu.", "turn" );
                sendState();

                return;
            }

            if ( conversation_id_.empty() )
            {
                conversation_id_ = store_.newIdentifier();
                conversation_title_ = titleFor( text );
                conversation_created_ = timestamp();
                conversation_messages_ = nlohmann::json::array();
            }

            turn_id_ = std::format( "turn-{}", ++turn_number_ );
            round_ = 0;
            turn_ = TurnRecord{};
            turn_calls_ = nlohmann::json::array();
            announced_call_id_.clear();

            const std::stop_token stop = [this]
                {
                    std::lock_guard<std::mutex> lock( stop_mutex_ );

                    // A fresh source per turn, so a stop asked for during one reply never ends the next; one asked for
                    // since this turn was accepted stops it at once.
                    stop_ = std::stop_source{};

                    if ( stop_asked_.load() )
                    {
                        stop_.request_stop();
                    }

                    return stop_.get_token();
                }();

            work_ = Work::Turn;
            sendState();

            send( "turn.begin", { { "stream_id", kStreamId }, { "turn_id", turn_id_ },
                { "conversation_id", conversation_id_ } } );

            Mila::AI::Agent& agent = model_.agent();
            RoundTimer timer;

            const Mila::AI::Response answered = agent.respond( text,
                [&]( const AgentEvent& event ) { onAgentEvent( event, timer, stop ); }, stop );

            if ( answered.rounds == 0 && answered.status == GenerateStatus::ContextOverflow )
            {
                send( "info", { { "text", std::format( "The prompt is {} tokens and the context is {}, so there is no "
                    "room to answer. Start a new conversation, or choose a longer context.",
                    agent.model().tokenizer().encode( agent.prompt( text ).text ).size(), agent.plan().contextLength() ) } } );
            }

            send( "turn.end", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "finish", turnFinish() },
                { "answer", turn_.answer } } );

            work_ = Work::None;

            recordTurn( text );
            busy_ = false;

            sendState();
            sendConversations();
        }

        void onAgentEvent( const AgentEvent& event, RoundTimer& timer, const std::stop_token& stop )
        {
            using Kind = AgentEvent::Kind;

            switch ( event.kind )
            {
                case Kind::RoundBegan:
                    ++round_;
                    timer.begin();
                    break;

                case Kind::Token:
                    timer.token();
                    break;

                case Kind::ReasoningOpened:
                    send( "turn.thought_break", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ } } );
                    break;

                case Kind::Reasoning:
                    sendText( "thought", event.text );
                    break;

                case Kind::Answer:
                    sendText( "answer", event.text );
                    break;

                case Kind::ToolCalled:
                    callAnswered( *event.call, event.text, event.is_error );
                    break;

                case Kind::MalformedToolCall:
                    // A call the template's grammar could not read is said, never dropped: the model meant to call
                    // something.
                    sendError( std::format( "[tool call not understood: {}]", event.text ), "turn" );
                    break;

                case Kind::RoundEnded:
                    endRound( timer, event, stop );
                    break;

                case Kind::ReasoningClosed:
                case Kind::ToolCallOpened:
                    break;
            }
        }

        void sendText( std::string_view channel, std::string_view text )
        {
            send( "turn.text", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
                { "channel", channel }, { "text", text } } );
        }

        /// A round's measurements, and its reasoning and text for the turn as stored.
        void endRound( const RoundTimer& timer, const AgentEvent& event, const std::stop_token& stop )
        {
            const Mila::Dnn::Conversation::Reply& reply = *event.reply;
            const Mila::AI::Agent& agent = model_.agent();

            const int thought_tokens = reply.reasoning.empty()
                ? 0 : static_cast<int>( agent.model().tokenizer().encode( reply.reasoning ).size() );

            // The agent runs a round's calls when the round finished them.
            const bool called = !reply.tool_calls.empty() && !session().tools().empty()
                && event.status == GenerateStatus::Success && !stop.stop_requested();

            send( "turn.round_end", {
                { "stream_id", kStreamId },
                { "turn_id", turn_id_ },
                { "round", round_ },
                { "prefill_ms", std::round( timer.prefillMilliseconds() ) },
                { "tokens_per_second", std::round( timer.decodeRate() * 10.0 ) / 10.0 },
                { "tokens", timer.tokens() },
                { "thought_tokens", thought_tokens },
                { "finish", called ? std::string_view( "tool_call" ) : finishName( event.status ) } } );

            ++turn_.rounds;
            turn_.prefill_ms += timer.prefillMilliseconds();
            turn_.tokens += timer.tokens();
            turn_.decoded_tokens += std::max( timer.tokens() - 1, 0 );
            turn_.decode_seconds += timer.decodeSeconds();
            turn_.last_status = event.status;
            turn_.last_called = called;

            const auto append = []( std::string& into, const std::string& part )
                {
                    if ( part.empty() )
                    {
                        return;
                    }

                    into += into.empty() ? "" : "\n\n";
                    into += part;
                };

            append( turn_.thought, reply.reasoning );
            append( turn_.answer, reply.content );
        }

        /// How the person is asked: the call is shown, and the reply waits for their answer from the reading thread.
        ToolAnswer askAboutCall( const Mila::AI::ToolQuestion& question )
        {
            const std::string call_id = announceCall( question.name, question.arguments );

            std::promise<ToolAnswer> answer;
            std::future<ToolAnswer> decision = answer.get_future();

            {
                std::lock_guard<std::mutex> lock( approval_mutex_ );

                if ( stop_asked_.load() )
                {
                    return ToolAnswer::Deny;
                }

                approval_.emplace( Approval{ turn_id_, call_id, std::move( answer ) } );
            }

            send( "turn.tool_approval", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
                { "choices", { "allow_once", "allow_conversation", "deny" } } } );

            const ToolAnswer decided = decision.get();

            // A stop answers the wait with Deny that the person never gave, so the stored call keeps no decision.
            if ( !stop_asked_.load() )
            {
                storedCall( call_id )[ "decision" ] = answerName( decided );
            }

            return decided;
        }

        void callAnswered( const ToolCall& call, std::string_view result, bool is_error )
        {
            // A call the rules decided was not asked about, so it has not been announced yet.
            const std::string call_id = announced_call_id_.empty() ? announceCall( call.name, call.arguments ) : announced_call_id_;
            announced_call_id_.clear();

            send( "turn.tool_result", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
                { "result", result }, { "is_error", is_error } } );

            nlohmann::json& stored = storedCall( call_id );
            stored[ "result" ] = result;
            stored[ "is_error" ] = is_error;
        }

        std::string announceCall( std::string_view name, std::string_view arguments )
        {
            announced_call_id_ = std::format( "call-{}", ++call_number_ );
            const std::string summary = callSummary( prompt_tools_, name, arguments );

            send( "turn.tool_call", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
                { "call_id", announced_call_id_ }, { "name", name }, { "summary", summary },
                { "arguments", arguments } } );

            // Stored as announced, so a call the person stopped before it ran is kept with no result (6.3).
            turn_calls_.push_back( { { "call_id", announced_call_id_ }, { "name", name }, { "summary", summary },
                { "arguments", arguments }, { "decision", nullptr }, { "result", nullptr }, { "is_error", false } } );

            return announced_call_id_;
        }

        /// The stored record of a call this turn announced.
        nlohmann::json& storedCall( const std::string& call_id )
        {
            for ( nlohmann::json& stored : turn_calls_ )
            {
                if ( stored[ "call_id" ] == call_id )
                {
                    return stored;
                }
            }

            throw std::logic_error( std::format( "No call {} was announced in this turn.", call_id ) );
        }

        /**
         * @brief How the turn ended, from its last round.
         *
         * A reply stopped while a call waited ends on a round that finished its call, so the stop is what says it
         * was cancelled. A turn with no round had no room in the context to start one.
         */
        std::string_view turnFinish() const
        {
            if ( turn_.rounds == 0 )
            {
                return "context";
            }

            if ( stop_asked_.load() && turn_.last_called )
            {
                return "cancelled";
            }

            return finishName( turn_.last_status );
        }

        /// The turn joins the stored conversation as the page shows it, and as the model read it.
        void recordTurn( const std::string& text )
        {
            conversation_messages_.push_back( { { "role", "user" }, { "content", text }, { "thought", nullptr },
                { "tool_calls", nullptr }, { "attachments", nullptr }, { "stats", nullptr } } );

            const double rate = turn_.decode_seconds > 0.0
                ? std::round( 10.0 * turn_.decoded_tokens / turn_.decode_seconds ) / 10.0 : 0.0;

            const nlohmann::json stats = { { "prefill_ms", std::round( turn_.prefill_ms ) }, { "tokens_per_second", rate },
                { "tokens", turn_.tokens }, { "rounds", turn_.rounds }, { "finish", turnFinish() } };

            conversation_messages_.push_back( {
                { "role", "assistant" },
                { "content", turn_.answer },
                { "thought", turn_.thought.empty() ? nlohmann::json( nullptr ) : nlohmann::json( turn_.thought ) },
                { "tool_calls", turn_calls_.empty() ? nlohmann::json( nullptr ) : turn_calls_ },
                { "attachments", nullptr },
                { "stats", stats } } );

            saveConversation();
        }

        // ---- Conversations ----------------------------------------------------------------

        void saveConversation()
        {
            nlohmann::json turns = nlohmann::json::array();

            for ( const Turn& turn : session().conversation() )
            {
                if ( turn.role != Role::System )
                {
                    turns.push_back( turnToJson( turn ) );
                }
            }

            const nlohmann::json conversation = {
                { "v", 1 },
                { "id", conversation_id_ },
                { "title", conversation_title_ },
                { "model", model_.name() },
                { "activity", conversation_activity_ },
                { "folder", conversationFolder() },
                { "plan", plan_.toJson() },
                { "created", conversation_created_ },
                { "updated", timestamp() },
                { "messages", conversation_messages_ },
                { "turns", std::move( turns ) } };

            try
            {
                store_.save( conversation );
            }
            catch ( const std::exception& failure )
            {
                sendError( std::format( "This conversation could not be saved: {}", failure.what() ), nullptr );
            }
        }

        void startNewConversation()
        {
            // Ends the tools allowed for the conversation with it.
            session().newConversation();
            forgetConversation();
        }

        /// The session's conversation no longer belongs to a stored one; the next turn starts one.
        void forgetConversation()
        {
            conversation_id_.clear();
            conversation_title_.clear();
            conversation_created_.clear();
            conversation_messages_ = nlohmann::json::array();
        }

        void openConversation( const std::string& id )
        {
            const std::optional<nlohmann::json> conversation = store_.load( id );

            if ( !conversation.has_value() )
            {
                sendError( "That conversation is no longer stored.", nullptr );

                return;
            }

            const std::string activity = conversation->value( "activity", std::string( "ask" ) );
            const nlohmann::json stored_folder = conversation->value( "folder", nlohmann::json( nullptr ) );
            const std::string folder = stored_folder.is_string() ? stored_folder.get<std::string>() : std::string{};

            if ( !useActivity( activity, folder ) )
            {
                return;
            }

            plan_.restore( conversation->value( "plan", nlohmann::json( nullptr ) ) );

            std::vector<Turn> turns;

            for ( const nlohmann::json& turn : conversation->value( "turns", nlohmann::json::array() ) )
            {
                turns.push_back( turnFromJson( turn ) );
            }

            // No tool is allowed for it yet: what the person allowed when it was written is asked about again.
            session().openConversation( std::move( turns ) );

            conversation_id_ = id;
            conversation_title_ = conversation->value( "title", std::string{} );
            conversation_created_ = conversation->value( "created", std::string{} );
            conversation_messages_ = conversation->value( "messages", nlohmann::json::array() );

            send( "conversation.loaded", { { "id", id }, { "title", conversation_title_ },
                { "model", conversation->value( "model", std::string{} ) }, { "activity", conversation_activity_ },
                { "folder", conversationFolder() }, { "messages", conversation_messages_ } } );

            sendState();
        }

        void renameConversation( const std::string& id, std::string title )
        {
            const auto first = title.find_first_not_of( " \t\r\n" );
            const auto last = title.find_last_not_of( " \t\r\n" );

            if ( first == std::string::npos )
            {
                return;
            }

            title = title.substr( first, last - first + 1 );

            if ( store_.rename( id, title ) && id == conversation_id_ )
            {
                conversation_title_ = title;
            }

            sendConversations();
        }

        void deleteConversation( const std::string& id )
        {
            store_.remove( id );

            if ( id == conversation_id_ )
            {
                startNewConversation();
                sendState();
            }

            sendConversations();
        }

        bool conversationHasUserTurn() const
        {
            return std::ranges::any_of( session().conversation(), []( const Turn& turn ) { return turn.role == Role::User; } );
        }

        // ---- Activities and their tools ---------------------------------------------------

        /**
         * @brief Give the session an activity's system prompt, tools, limits, rules and grant (MilaApp.md 3.1).
         *
         * Ask & Explore has no tools, no grant, and limits of rounds and reply tokens. The others run with no round or reply
         * limit, bounded by the context and the person's Stop: Build & Code with the file tools, granted its folder
         * to read and write, Research & Analyze reading web pages, granted the network, Get Things Done with both and
         * its plan, whose updates run without asking.
         *
         * @return False, having said why, for an activity this session does not offer or a folder that does not
         *         exist; nothing has changed then.
         */
        bool useActivity( const std::string& activity, const std::string& folder )
        {
            const bool known = activity == "ask" || activity == "build" || activity == "research" || activity == "tasks";

            if ( !known )
            {
                sendError( std::format( "There is no activity called {}.", activity ), "conversation" );

                return false;
            }

            std::vector<Mila::AI::GrantFolder> folders;

            if ( activity == "build" || activity == "tasks" )
            {
                if ( folder.empty() )
                {
                    sendError( "This works in a folder; choose one to start.", "conversation" );

                    return false;
                }

                folders.push_back( { std::filesystem::path( std::u8string( folder.begin(), folder.end() ) ),
                    Mila::AI::FolderAccess::ReadWrite } );
            }

            std::optional<Mila::AI::ToolGrant> grant;

            try
            {
                grant.emplace( std::move( folders ), activity == "research" || activity == "tasks" );
            }
            catch ( const std::invalid_argument& )
            {
                sendError( std::format( "The folder {} is not there any more.", folder ), "conversation" );

                return false;
            }

            usePrompt( promptFor( activity ) );

            session().withLimits( activity == "ask"
                ? Mila::AI::AgentLimits{ .rounds = kAskRounds, .reply_tokens = kAskReplyTokens }
                : Mila::AI::AgentLimits{} );

            Mila::AI::ToolRules rules;

            if ( activity == "tasks" )
            {
                rules.add( { std::string( TaskPlan::kUpdateName ), Mila::AI::ToolPermission::Allow, {} } );
            }

            session().withRules( std::move( rules ) );

            conversation_folder_.reset();

            if ( !grant->folders().empty() )
            {
                const std::u8string text = grant->folders().front().folder.u8string();
                conversation_folder_.emplace( text.begin(), text.end() );
            }

            session().withGrant( std::move( *grant ) );
            plan_.clear();
            conversation_activity_ = activity;

            return true;
        }

        /**
         * @brief An activity's own prompt. Research & Analyze and Get Things Done have one that names web search and
         *        one that does not, since a prompt naming a tool not offered draws calls of it.
         */
        std::filesystem::path promptFor( const std::string& activity ) const
        {
            if ( activity == "ask" )
            {
                return prompts_ / "ask.json";
            }

            if ( activity == "research" )
            {
                return prompts_ / ( web_search_ ? "research-search.json" : "research.json" );
            }

            if ( activity == "tasks" )
            {
                return prompts_ / ( web_search_ ? "tasks-search.json" : "tasks.json" );
            }

            return prompts_ / "files.json";
        }

        /// The prompt's system prompt and its tools, from the next response on; the model and conversation stay.
        void usePrompt( const std::filesystem::path& path )
        {
            PromptFile prompt = PromptFile::load( path );

            session().withSystemPrompt( std::move( prompt.system_prompt ) );
            prompt_tools_ = std::move( prompt.tools );
            offerTools();
        }

        /// A key given or removed changes the open conversation's prompt and tools from its next response.
        void refreshSearchPrompt()
        {
            if ( conversation_activity_ == "research" || conversation_activity_ == "tasks" )
            {
                usePrompt( promptFor( conversation_activity_ ) );
            }
        }

        /// The code that runs a tool, offered whenever the activity's prompt describes it.
        void registerTool( std::string name, Mila::AI::ToolNeeds needs, Handler handler )
        {
            handlers_.insert_or_assign( std::move( name ), std::pair{ needs, std::move( handler ) } );
        }

        /**
         * @brief Offer exactly the tools the prompt describes and the session runs.
         *
         * Describing a tool nothing runs primes the model to call it, so one with no handler is left out.
         */
        void offerTools()
        {
            Mila::AI::Session& offering = session();
            offering.withoutTools();

            for ( const PromptTool& described : prompt_tools_ )
            {
                const auto handler = handlers_.find( described.name );

                if ( handler == handlers_.end() )
                {
                    continue;
                }

                Mila::AI::Tool tool{
                    described.name,
                    described.description,
                    described.parameters,
                    [run = handler->second.second]( std::string_view arguments, const Mila::AI::ToolGrant& grant, std::stop_token )
                    {
                        return run( std::string( arguments ), grant );
                    } };

                tool.needs = handler->second.first;
                offering.withTool( std::move( tool ) );
            }

            applyTemplateVariables();
        }

        // ---- Models -----------------------------------------------------------------------

        /// The request a load sends: this device, the context fixed or the longest that fits, and a quantization to
        /// apply as the weights load when one is asked for.
        DeploymentRequest requestFor( std::optional<WeightQuantization> quantization ) const
        {
            DeploymentRequest request;
            request.withDevice( Mila::Dnn::Compute::DeviceId{ Mila::Dnn::Compute::DeviceType::Cuda, device_ } )
                .withKvCacheCompression( Mila::Dnn::KvCacheCompression::None );

            if ( quantization && *quantization != WeightQuantization::None )
            {
                request.withWeightQuantization( *quantization );
            }

            if ( settings_.context_length )
            {
                request.withContextLength( *settings_.context_length );
            }
            else
            {
                request.withAutomaticContextLength();
            }

            return request;
        }

        /// The model the settings remember, opened as the session starts; one that will not open is said in the
        /// page's no-model state, where the person chooses another.
        void openRemembered()
        {
            const DeploymentRequest request = requestFor( std::nullopt );

            if ( auto refusal = Mila::AI::Agent::check( settings_.model, request ) )
            {
                no_model_reason_ = refusalText( *refusal );

                return;
            }

            work_ = Work::Load;
            open( settings_.model, std::nullopt, request );
            work_ = Work::None;
        }

        void loadModel( const nlohmann::json& request )
        {
            std::optional<WeightQuantization> quantization;

            if ( request.contains( "quantization" ) && request[ "quantization" ].is_string() )
            {
                quantization = parseLoadQuantization( request[ "quantization" ].get<std::string>() );

                if ( !quantization.has_value() )
                {
                    sendError( "Quantization is none, fp8, fp4 or q4_0.", "load" );

                    return;
                }
            }

            const std::string asked = request.value( "name", std::string{} );
            const std::string name = InstalledModel::find( asked )
                .transform( []( const InstalledModel& model ) { return model.name(); } )
                .value_or( asked );

            if ( model_.resident() && model_.name() == name && ( !quantization || quantization == loaded_quantization_ ) )
            {
                send( "info", { { "text", std::format( "{} is already loaded.", name ) } } );

                return;
            }

            work_ = Work::Load;
            switchModel( name, quantization );
            work_ = Work::None;

            sendState();
        }

        /**
         * @brief Change model, clearing the conversation, as the person asked: a model's name, checked before the
         *        one resident is released, so a name that will not open leaves it working.
         *
         * A model that does not fit is found only once the card is free; the one it replaced is then loaded again.
         */
        void switchModel( const std::string& name, std::optional<WeightQuantization> quantization )
        {
            const bool had_model = model_.resident();
            const std::string previous = model_.name();
            const std::optional<WeightQuantization> previous_quantization = loaded_quantization_;
            const DeploymentRequest previous_request = had_model ? model_.request() : DeploymentRequest{};

            const DeploymentRequest request = requestFor( quantization );

            if ( auto refusal = Mila::AI::Agent::check( name, request ) )
            {
                sendError( refusalText( *refusal ), workName() );

                return;
            }

            if ( !open( name, quantization, request ) )
            {
                if ( had_model && open( previous, previous_quantization, previous_request ) )
                {
                    send( "info", { { "text", std::format( "{} is loaded again.", previous ) } } );
                }

                return;
            }

            startNewConversation();

            send( "info", { { "text", had_model ? "Model switched. Conversation history cleared." : "Model loaded." } } );

            if ( !settings_.context_length )
            {
                send( "info", { { "text", std::format( "Context {} (auto).", model_.agent().plan().contextLength() ) } } );
            }
        }

        /**
         * @brief Open a model on a request, saying how it goes: a load from start to end, and why it failed if it did.
         *
         * @return False, having said why, when nothing could be opened; nothing is resident then, and the session is
         *         held.
         */
        bool open( const std::string& name, std::optional<WeightQuantization> quantization, const DeploymentRequest& request )
        {
            const bool applied_at_load = quantization && *quantization != WeightQuantization::None;

            // The quantization is quoted only when it is applied as the weights load: a pre-quantized model already
            // carries it in its name.
            load_label_ = applied_at_load
                ? std::format( "Loading {} ({})", name, quantizationWord( *quantization ) )
                : std::format( "Loading {}", name );

            sendState();
            send( "progress", { { "task", loadTask() }, { "label", load_label_ }, { "done", nullptr }, { "total", nullptr } } );

            std::string failure;

            try
            {
                model_.open( name, request );
            }
            catch ( const Mila::AI::OpenRefusedError& error )
            {
                failure = refusalText( error.refusal() );
            }
            catch ( const DeploymentRefusedError& error )
            {
                const std::optional<InstalledModel> installed = InstalledModel::find( name );
                const bool quantizable = !applied_at_load && installed
                    && installed->storedWeightQuantization() == WeightQuantization::None;

                failure = refusalText( error.refusal(), quantizable );
            }
            catch ( const std::exception& error )
            {
                failure = error.what();
            }

            send( "progress", { { "task", loadTask() }, { "label", load_label_ }, { "done", 1 }, { "total", 1 } } );

            if ( !failure.empty() )
            {
                loaded_quantization_.reset();
                sendError( std::format( "Could not load {}: {}", name, failure ), workName() );

                return false;
            }

            loaded_quantization_ = quantization;
            no_model_reason_.clear();
            settings_.model = model_.name();
            saveSettings();

            applyTemplateVariables();

            if ( settings_.detail == "all" )
            {
                send( "log", { { "level", "info" }, { "text", model_.agent().model().model().toString() } } );
                send( "log", { { "level", "info" }, { "text", model_.agent().model().model().getMemoryStats().toString() } } );
            }

            return true;
        }

        void installModel( const std::string& name )
        {
            install_name_ = name;
            install_total_ = 0;
            install_percent_ = -1;

            work_ = Work::Install;
            sendState();

            bool installed = false;

            try
            {
                for ( const std::string& line : Mila::Assistant::installModel( name,
                    [this]( std::uint64_t received, std::uint64_t total ) { installProgress( received, total ); } ) )
                {
                    send( "info", { { "text", line } } );
                }

                installed = true;
            }
            catch ( const std::exception& error )
            {
                sendError( std::format( "Install failed: {}", error.what() ), "install" );
            }

            // A session with nothing resident wants the model it just installed; a load that fails is said as part of
            // the install.
            if ( installed && !model_.resident() )
            {
                switchModel( name, std::nullopt );
            }

            work_ = Work::None;

            sendModels( false );
            sendState();
        }

        void installProgress( std::uint64_t received, std::uint64_t total )
        {
            // One message per whole percent of each file, and every file's last.
            const int percent = static_cast<int>( ( received * 100 ) / total );

            if ( total == install_total_ && percent == install_percent_ && received < total )
            {
                return;
            }

            install_total_ = total;
            install_percent_ = percent;

            send( "progress", { { "task", "install" }, { "label", std::format( "Installing {}", install_name_ ) },
                { "done", received }, { "total", total } } );
        }

        void removeModel( const std::string& name )
        {
            try
            {
                for ( const std::string& line : Mila::Assistant::removeModel( name ) )
                {
                    send( "info", { { "text", line } } );
                }
            }
            catch ( const std::exception& error )
            {
                sendError( std::format( "Remove failed: {}", error.what() ), nullptr );
            }

            sendModels( false );
        }

        /**
         * @brief Open the resident model again on another request, the conversation kept: a context sizes the
         *        model's buffers as it is built, so a new one is a new agent over the same session.
         *
         * A request that does not fit puts the model back at the context it had.
         */
        void reopen( const DeploymentRequest& request, std::string_view done )
        {
            const std::string name = model_.name();
            const DeploymentRequest previous = model_.request();

            if ( open( name, loaded_quantization_, request ) )
            {
                send( "info", { { "text", std::format( "Context {}{}. Conversation kept.", model_.agent().plan().contextLength(), done ) } } );

                return;
            }

            if ( open( name, loaded_quantization_, previous ) )
            {
                send( "info", { { "text", std::format( "{} is loaded again at context {}.", name,
                    model_.agent().plan().contextLength() ) } } );
            }
        }

        // ---- Settings ---------------------------------------------------------------------

        /// One change per request (MilaApp.md 4.2); what the session says about it arrives as its own events.
        void changeSetting( const nlohmann::json& request )
        {
            if ( request.contains( "context" ) )
            {
                changeContext( request[ "context" ] );
            }
            else if ( request.contains( "thinking" ) )
            {
                changeThinking( request[ "thinking" ] );
            }
            else if ( request.contains( "effort" ) )
            {
                changeEffort( request[ "effort" ] );
            }
            else if ( request.contains( "detail" ) )
            {
                const nlohmann::json& value = request[ "detail" ];

                if ( !value.is_string() || !std::ranges::contains( kDetailLevels, value.get<std::string>() ) )
                {
                    sendError( "Detail is off, thoughts or all.", "settings" );

                    return;
                }

                settings_.detail = value.get<std::string>();
                useDetail( settings_.detail );
                saveSettings();
            }
            else if ( request.contains( "sampling" ) )
            {
                changeSampling( request[ "sampling" ] );
            }
            else if ( request.contains( "seed" ) )
            {
                changeSeed( request[ "seed" ] );
            }
            else if ( request.contains( "search_key" ) )
            {
                changeSearchKey( request[ "search_key" ] );
            }
            else
            {
                sendError( "That is not a setting this session has.", "settings" );
            }
        }

        void changeContext( const nlohmann::json& value )
        {
            if ( value.is_string() && value.get<std::string>() == "auto" )
            {
                settings_.context_length.reset();
                saveSettings();

                if ( !model_.resident() )
                {
                    send( "info", { { "text", "Context set to auto; it is measured when a model loads." } } );

                    return;
                }

                // Measured again: the card's free memory may have changed since the model loaded.
                reopen( requestFor( loaded_quantization_ ), " (auto)" );

                return;
            }

            if ( !value.is_number_unsigned() || value.get<std::uint64_t>() == 0
                || value.get<std::uint64_t>() > static_cast<std::uint64_t>( std::numeric_limits<dim_t>::max() ) )
            {
                sendError( "A context window is a number of tokens, or auto.", "settings" );

                return;
            }

            const dim_t length = value.get<dim_t>();

            if ( !model_.resident() )
            {
                settings_.context_length = length;
                saveSettings();
                send( "info", { { "text", std::format( "Context set to {}; it applies when a model loads.", length ) } } );

                return;
            }

            Mila::AI::Agent& agent = model_.agent();

            if ( length == agent.plan().contextLength() )
            {
                const bool was_fixed = settings_.context_length.has_value();

                settings_.context_length = length;
                saveSettings();

                // The buffers are this size already, so pinning the number auto chose rebuilds nothing.
                send( "info", { { "text", was_fixed
                    ? std::format( "Context is already {}.", length )
                    : std::format( "Context {} is now pinned.", length ) } } );

                return;
            }

            // Refused below what the conversation already holds, which would load and then have no room to answer.
            const dim_t held = static_cast<dim_t>( agent.model().tokenizer().encode( agent.prompt().text ).size() );

            if ( length <= held )
            {
                sendError( std::format( "This conversation is already {} tokens, so a context of {} has no room to "
                    "answer. Start a new conversation, or choose a longer context.", held, length ), "settings" );

                return;
            }

            const std::optional<dim_t> previous = settings_.context_length;
            settings_.context_length = length;

            const DeploymentRequest request = requestFor( loaded_quantization_ );

            reopen( request, "" );

            // A context that did not fit is not the one the person has.
            if ( !model_.resident() || model_.agent().plan().contextLength() != length )
            {
                settings_.context_length = previous;
            }

            saveSettings();
        }

        void changeThinking( const nlohmann::json& value )
        {
            if ( !value.is_boolean() || !hasThinkingSwitch() )
            {
                sendError( "This model has no reasoning channel.", "settings" );

                return;
            }

            settings_.thinking = value.get<bool>();
            saveSettings();
            applyTemplateVariables();
        }

        void changeEffort( const nlohmann::json& value )
        {
            const Mila::Dnn::Conversation::TemplateVariable* effort = effortVariable();

            if ( effort == nullptr )
            {
                sendError( "This model's thinking is on or off; it has no effort levels.", "settings" );

                return;
            }

            if ( !value.is_string() || !std::ranges::contains( effort->values, value.get<std::string>() ) )
            {
                std::string levels;

                for ( const std::string& level : effort->values )
                {
                    levels += levels.empty() ? "" : ", ";
                    levels += level;
                }

                sendError( std::format( "Effort is one of {}.", levels ), "settings" );

                return;
            }

            settings_.effort = value.get<std::string>();
            saveSettings();
            applyTemplateVariables();
        }

        void changeSampling( const nlohmann::json& change )
        {
            const std::string key = change.is_object() ? change.value( "key", std::string{} ) : std::string{};
            const nlohmann::json value = change.is_object() && change.contains( "value" ) ? change[ "value" ] : nlohmann::json{};
            const bool follow_model = value.is_string() && value.get<std::string>() == "model";

            if ( !std::ranges::contains( kSamplingKeys, key ) )
            {
                sendError( "That is not a sampling setting.", "settings" );

                return;
            }

            const bool whole = key != "top_k" || value.is_number_integer();

            if ( !follow_model && ( !value.is_number() || !whole || !samplingValueAllowed( key, value.get<double>() ) ) )
            {
                sendError( samplingBounds( key ), "settings" );

                return;
            }

            Mila::AI::SamplingOverride& sampling = settings_.sampling;

            if ( key == "temperature" )
                sampling.temperature = follow_model ? std::nullopt : std::optional<float>( value.get<float>() );
            else if ( key == "top_k" )
                sampling.top_k = follow_model ? std::nullopt : std::optional<int>( value.get<int>() );
            else if ( key == "top_p" )
                sampling.top_p = follow_model ? std::nullopt : std::optional<float>( value.get<float>() );
            else
                sampling.presence_penalty = follow_model ? std::nullopt : std::optional<float>( value.get<float>() );

            session().withSampling( sampling );
            saveSettings();
        }

        static std::string_view samplingBounds( std::string_view key ) noexcept
        {
            if ( key == "temperature" )
                return "Temperature is from 0 to 5; 0 always picks the likeliest word.";

            if ( key == "top_k" )
                return "top_k is a whole number, 0 or more; 0 turns it off.";

            if ( key == "top_p" )
                return "top_p is from 0 to 1; 1 turns it off.";

            return "presence_penalty is from -2 to 2; 0 turns it off.";
        }

        void changeSeed( const nlohmann::json& value )
        {
            if ( !value.is_number_unsigned() )
            {
                sendError( "A seed is a whole number, 0 or more.", "settings" );

                return;
            }

            if ( !model_.resident() )
            {
                sendError( "No model is loaded.", "settings" );

                return;
            }

            const std::uint64_t seed = value.get<std::uint64_t>();
            model_.agent().model().model().seedSampler( seed );

            send( "info", { { "text", std::format( "Sampler reseeded ({}).", seed ) } } );
        }

        /// A key is kept and searching offered from the next response; null or empty removes both.
        void changeSearchKey( const nlohmann::json& value )
        {
            const std::string key = value.is_string() ? value.get<std::string>() : std::string{};
            const auto first = key.find_first_not_of( " \t\r\n" );
            const auto last = key.find_last_not_of( " \t\r\n" );
            const std::string trimmed = first == std::string::npos ? std::string{} : key.substr( first, last - first + 1 );

            if ( trimmed.empty() )
            {
                search_key_file_.remove();
                web_search_.reset();
                handlers_.erase( std::string( WebSearch::kName ) );
                refreshSearchPrompt();

                return;
            }

            search_key_file_.write( trimmed );
            useSearchKey( trimmed );
            refreshSearchPrompt();
        }

        void useSearchKey( const std::string& key )
        {
            web_search_.emplace( key );

            Mila::AI::ToolNeeds network;
            network.uses_network = true;

            // Registered once; a later key replaces the search it runs.
            registerTool( std::string( WebSearch::kName ), network,
                [this]( const std::string& arguments, const Mila::AI::ToolGrant& grant )
                {
                    if ( !web_search_ )
                    {
                        throw std::runtime_error( "web search is not set up" );
                    }

                    return web_search_->search( arguments, grant );
                } );
        }

        void useDetail( const std::string& detail )
        {
            Mila::Logging::Logger::defaultLogger().setLevel( detail == "all"
                ? Mila::Logging::LogLevel::Info
                : Mila::Logging::LogLevel::Warning );
        }

        /// Kept for the next launch; a file that cannot be written is said, and the setting still applies now.
        void saveSettings()
        {
            try
            {
                settings_file_.write( settings_ );
            }
            catch ( const std::exception& failure )
            {
                sendError( std::format( "This setting applies now but could not be kept: {}", failure.what() ), "settings" );
            }
        }

        // ---- The template's own variables -------------------------------------------------

        const Mila::Dnn::Conversation::TemplateVariable* templateVariable( std::string_view name ) const
        {
            if ( !model_.resident() )
            {
                return nullptr;
            }

            const auto variables = model_.agent().model().protocol()->templateVariables();
            const auto found = std::ranges::find( variables, name, &Mila::Dnn::Conversation::TemplateVariable::name );

            return found == variables.end() ? nullptr : &*found;
        }

        /// Whether the resident model reasons, and its template can turn that off.
        bool hasThinkingSwitch() const
        {
            return model_.resident() && model_.agent().model().capabilities().thinking
                && templateVariable( "enable_thinking" ) != nullptr;
        }

        /// The template's effort variable, where its thinking has levels and is on.
        const Mila::Dnn::Conversation::TemplateVariable* effortVariable() const
        {
            return hasThinkingSwitch() ? templateVariable( "reasoning_effort" ) : nullptr;
        }

        /**
         * @brief The settings as the resident model's template variables, each only where the template has it.
         *
         * Tools go in the system message where the template can place them there. Llama 3.1's template puts them in
         * the first user message by default, under an instruction to answer with a call, and Llama 3.1 8B then called
         * a tool for every question; with them in the system message and a system prompt saying when to call, it
         * answered 10 of 10 questions that needed none (measured 2026-10-25).
         */
        void applyTemplateVariables()
        {
            if ( !model_.resident() )
            {
                return;
            }

            std::map<std::string, std::string> variables;

            if ( templateVariable( "enable_thinking" ) != nullptr )
            {
                variables[ "enable_thinking" ] = settings_.thinking ? "true" : "false";
            }

            if ( const auto* effort = templateVariable( "reasoning_effort" );
                effort != nullptr && settings_.thinking && std::ranges::contains( effort->values, settings_.effort ) )
            {
                variables[ "reasoning_effort" ] = settings_.effort;
            }

            if ( templateVariable( "tools_in_user_message" ) != nullptr && !session().tools().empty() )
            {
                variables[ "tools_in_user_message" ] = "false";
            }

            model_.agent().withTemplateVariables( std::move( variables ) );
        }

        // ---- Attachments ------------------------------------------------------------------

        /**
         * @brief Every attachment is refused, by name: no model this build runs takes an image or audio yet
         *        (InstalledModel's isSupportedModality), so session.state lists no modality to accept one for.
         */
        void refuseAttachment( const nlohmann::json& request )
        {
            const std::string kind = request.value( "kind", std::string{} );

            std::string reason;

            if ( !model_.resident() )
            {
                reason = "No model is loaded to read it.";
            }
            else if ( kind == "image" || kind == "audio" )
            {
                reason = std::format( "{} cannot take {}.", model_.name(), kind == "image" ? "images" : "audio" );
            }
            else
            {
                reason = "Only PNG and JPEG images and WAV audio can be attached.";
            }

            send( "attachment.refused", { { "attachment_id", request.value( "attachment_id", std::string{} ) },
                { "reason", reason } } );
        }

        // ---- Events of the process's own --------------------------------------------------

        void sendState()
        {
            const bool resident = model_.resident();
            const bool thinking_capable = hasThinkingSwitch();
            const auto* effort = effortVariable();

            nlohmann::ordered_json effort_levels = nullptr;
            nlohmann::ordered_json effort_in_effect = nullptr;
            std::string thinking = "off";

            if ( effort != nullptr )
            {
                effort_levels = effort->values;
                effort_in_effect = std::string( Mila::Dnn::Conversation::resolveTemplateVariable(
                    model_.agent().model().protocol()->templateVariables(), session().templateVariables(), "reasoning_effort" ) );
            }

            if ( thinking_capable && model_.agent().thinks() )
            {
                thinking = effort != nullptr ? effort_in_effect.get<std::string>() : "on";
            }

            nlohmann::ordered_json sampling = nlohmann::ordered_json::object();

            if ( resident )
            {
                const Mila::Dnn::SamplingParams effective = model_.agent().sampling();
                const Mila::AI::SamplingOverride& chosen = settings_.sampling;

                sampling[ "temperature" ] = { { "value", effective.temperature }, { "is_model_default", !chosen.temperature } };
                sampling[ "top_k" ] = { { "value", effective.top_k }, { "is_model_default", !chosen.top_k } };
                sampling[ "top_p" ] = { { "value", effective.top_p }, { "is_model_default", !chosen.top_p } };
                sampling[ "presence_penalty" ] = { { "value", effective.presence_penalty },
                    { "is_model_default", !chosen.presence_penalty } };
            }

            // The modalities this build runs for the resident model, from the library's own answer.
            nlohmann::ordered_json modalities = nlohmann::ordered_json::array();

            if ( resident )
            {
                for ( const std::string& modality : model_.agent().model().capabilities().modalities )
                {
                    modalities.push_back( modality );
                }
            }

            const bool busy = busy_.load() || work_ != Work::None;

            const nlohmann::ordered_json stream = {
                { "stream_id", kStreamId },
                { "parent_stream_id", nullptr },
                { "conversation_id", conversation_id_.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( conversation_id_ ) },
                { "busy", busy } };

            const dim_t context_length = resident ? model_.agent().plan().contextLength() : settings_.context_length.value_or( 0 );

            send( "session.state", {
                { "model", resident ? nlohmann::ordered_json( model_.name() ) : nlohmann::ordered_json( nullptr ) },
                { "no_model_reason", no_model_reason_.empty() || resident
                    ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( no_model_reason_ ) },
                { "context_length", context_length },
                { "context_automatic", !settings_.context_length.has_value() },
                { "thinking_capable", thinking_capable },
                { "thinking", thinking },
                { "effort", std::move( effort_in_effect ) },
                { "effort_levels", std::move( effort_levels ) },
                { "detail", settings_.detail },
                { "sampling", std::move( sampling ) },
                { "modalities", std::move( modalities ) },
                { "activities", nlohmann::ordered_json::array( { "ask", "build", "research", "tasks" } ) },
                { "activity", conversation_activity_ },
                { "folder", conversationFolder() },
                { "plan", plan_.toJson() },
                { "web_search", web_search_.has_value() },
                { "busy", busy },
                { "streams", nlohmann::ordered_json::array( { stream } ) } } );
        }

        nlohmann::json conversationFolder() const
        {
            return conversation_folder_ ? nlohmann::json( *conversation_folder_ ) : nlohmann::json( nullptr );
        }

        void sendConversations()
        {
            send( "conversations.list", { { "items", store_.list() } } );
        }

        /// What is installed, and, when asked for, what the publisher offers.
        void sendModels( bool online )
        {
            nlohmann::json published = nullptr;

            if ( online )
            {
                try
                {
                    published = publishedModels();
                }
                catch ( const std::exception& failure )
                {
                    published = nullptr;
                    sendError( std::format( "Could not list the models that can be installed: {}", failure.what() ), nullptr );
                }
            }

            send( "models.list", { { "installed", installedModels() }, { "published", std::move( published ) } } );
        }

        // ---- Writing ----------------------------------------------------------------------

        void sendError( std::string_view text, nlohmann::ordered_json during )
        {
            send( "error", { { "text", text }, { "during", std::move( during ) } } );
        }

        /// One message, one line. Text the model wrote that is not valid UTF-8 is replaced rather than ending the line.
        void send( std::string_view type, nlohmann::ordered_json fields )
        {
            nlohmann::ordered_json message = { { "v", 1 }, { "type", type } };

            for ( auto field = fields.begin(); field != fields.end(); ++field )
            {
                message[ field.key() ] = std::move( field.value() );
            }

            const std::string line = message.dump( -1, ' ', false, nlohmann::ordered_json::error_handler_t::replace );

            std::lock_guard<std::mutex> lock( output_mutex_ );
            out_ << line << '\n' << std::flush;
        }

        // ---- Small facts ------------------------------------------------------------------

        /// Where an error happened, for the page to show it there.
        nlohmann::ordered_json workName() const
        {
            switch ( work_ )
            {
                case Work::Turn:     return "turn";
                case Work::Load:     return "load";
                case Work::Install:  return "install";
                case Work::Settings: return "settings";
                case Work::None:     break;
            }

            return nullptr;
        }

        /// A context change rebuilds the model it already holds; anything else that loads, loads.
        std::string_view loadTask() const
        {
            return work_ == Work::Settings ? "rebuild" : "load";
        }

        static std::string_view finishName( GenerateStatus status )
        {
            switch ( status )
            {
                case GenerateStatus::Success:             return "stop";
                case GenerateStatus::MaxNewTokensReached: return "length";
                case GenerateStatus::ContextOverflow:     return "context";
                case GenerateStatus::ClientCancelled:     return "cancelled";
            }

            return "error";
        }

        static std::string_view answerName( ToolAnswer answer )
        {
            switch ( answer )
            {
                case ToolAnswer::Once:            return "allow_once";
                case ToolAnswer::ForConversation: return "allow_conversation";
                case ToolAnswer::Deny:            break;
            }

            return "deny";
        }

        /// The first user message, cut at a word under 60 characters (MilaApp.md 6.3).
        static std::string titleFor( const std::string& text )
        {
            std::string title = text.substr( 0, text.find( '\n' ) );

            if ( title.size() < 60 )
            {
                return title;
            }

            const auto cut = title.rfind( ' ', 59 );

            title.resize( cut == std::string::npos || cut == 0 ? 59 : cut );

            // Not inside a UTF-8 sequence: back off continuation bytes, then the lead byte they belonged to.
            while ( !title.empty() && ( static_cast<unsigned char>( title.back() ) & 0xC0 ) == 0x80 )
            {
                title.pop_back();
            }

            if ( !title.empty() && static_cast<unsigned char>( title.back() ) >= 0xC0 )
            {
                title.pop_back();
            }

            return title;
        }

        /// To the millisecond: the store lists conversations by it, and two can change within a second.
        static std::string timestamp()
        {
            const auto now = std::chrono::floor<std::chrono::milliseconds>( std::chrono::system_clock::now() );

            return std::format( "{:%FT%TZ}", now );
        }

        static std::string_view roleName( Role role )
        {
            switch ( role )
            {
                case Role::System:    return "system";
                case Role::User:      return "user";
                case Role::Assistant: return "assistant";
                case Role::Tool:      return "tool";
            }

            return "user";
        }

        static nlohmann::json turnToJson( const Turn& turn )
        {
            nlohmann::json calls = nlohmann::json::array();

            for ( const ToolCall& call : turn.tool_calls )
            {
                calls.push_back( { { "name", call.name }, { "arguments", call.arguments } } );
            }

            return { { "role", roleName( turn.role ) }, { "content", turn.content }, { "tool_calls", std::move( calls ) },
                { "tool_name", turn.tool_name }, { "reasoning", turn.reasoning } };
        }

        static Turn turnFromJson( const nlohmann::json& stored )
        {
            const std::string role = stored.value( "role", std::string{} );

            Turn turn;
            turn.role = role == "assistant" ? Role::Assistant
                : role == "tool" ? Role::Tool
                : Role::User;
            turn.content = stored.value( "content", std::string{} );
            turn.tool_name = stored.value( "tool_name", std::string{} );
            turn.reasoning = stored.value( "reasoning", std::string{} );

            for ( const nlohmann::json& call : stored.value( "tool_calls", nlohmann::json::array() ) )
            {
                turn.tool_calls.push_back( { call.value( "name", std::string{} ), call.value( "arguments", std::string{} ) } );
            }

            return turn;
        }

        /// The turn in flight, as its rounds come to it.
        struct TurnRecord
        {
            int rounds{ 0 };
            double prefill_ms{ 0.0 };
            int tokens{ 0 };
            int decoded_tokens{ 0 };
            double decode_seconds{ 0.0 };
            GenerateStatus last_status{ GenerateStatus::Success };
            bool last_called{ false };
            std::string thought;
            std::string answer;
        };

        Settings settings_;
        SettingsFile settings_file_;
        std::vector<std::string> settings_problems_;
        int device_;

        ResidentAgent model_;

        // The quantization asked for when the resident model loaded, applied again when it reloads.
        std::optional<WeightQuantization> loaded_quantization_;

        // Why the model the settings remember did not open; empty once one does.
        std::string no_model_reason_;

        ConversationStore store_;
        std::filesystem::path prompts_;
        std::ostream& out_;
        SearchKeyFile search_key_file_;

        // Every tool the session can run, by name, with what it declares it needs; offered when the activity's
        // prompt describes it.
        std::map<std::string, std::pair<Mila::AI::ToolNeeds, Handler>> handlers_;

        // The tools the activity's prompt describes, for the line each call is shown with.
        std::vector<PromptTool> prompt_tools_;

        // Set while the person has given a search key; the web_search tool is offered only then.
        std::optional<WebSearch> web_search_;
        std::mutex output_mutex_;

        RequestQueue queue_;

        // Set by the reading thread when it accepts a turn, cleared by this thread when the turn is done.
        std::atomic<bool> busy_{ false };
        std::atomic<bool> stop_asked_{ false };

        // Replaced at the start of each turn and requested by the reading thread.
        std::mutex stop_mutex_;
        std::stop_source stop_;

        // Read by the reading thread only under approval_mutex_.
        std::mutex approval_mutex_;
        std::optional<Approval> approval_;

        Work work_{ Work::None };
        std::string load_label_;

        std::string install_name_;
        std::uint64_t install_total_{ 0 };
        int install_percent_{ -1 };

        // The conversation's activity, and the folder its grant holds, in UTF-8, for those that have one
        // (MilaApp.md 3.1).
        std::string conversation_activity_ = "ask";
        std::optional<std::string> conversation_folder_;
        TaskPlan plan_;

        // The conversation the session's history belongs to; empty until its first turn.
        std::string conversation_id_;
        std::string conversation_title_;
        std::string conversation_created_;
        nlohmann::json conversation_messages_ = nlohmann::json::array();

        // The turn in flight.
        int turn_number_{ 0 };
        int call_number_{ 0 };
        std::string turn_id_;
        int round_{ 0 };
        TurnRecord turn_;
        nlohmann::json turn_calls_ = nlohmann::json::array();
        std::string announced_call_id_;
    };
}
