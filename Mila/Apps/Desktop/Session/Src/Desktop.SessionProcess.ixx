/**
 * @file Desktop.SessionProcess.ixx
 * @brief The desktop app's session process: Chat's session, spoken to as JSON lines on stdio (ChatDesktop.md 6.1).
 */

export module Desktop.SessionProcess;

import std;
import Chat.ParsedResponse;
import Chat.Json;
import Chat.ModelCatalog;
import Chat.Session;
import Chat.TextChannel;
import Chat.ToolDecision;
import Desktop.ConversationStore;

import Mila;

namespace Mila::DesktopApp
{
    using namespace Mila::ChatApp;
    using Mila::Dnn::GenerateStatus;
    using Mila::Deployment::InstalledModel;

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
     * @brief Chat's session as a program a window starts: section 4.2's requests in, section 4.1's events out.
     *
     * It translates and holds no logic of its own, so that v0.22.0 makes it a layer over Mila::AI (ChatDesktop.md
     * 6.1). Two threads: the one that called run() does every piece of session work, model loads included, and a
     * second reads requests, so a stop or a tool decision is heard while a reply is being written. Every event is
     * one line on the protocol stream, written under a lock because both threads write.
     */
    export class SessionProcess final : public SessionEvents
    {
    public:

        SessionProcess( ChatConfig config, std::filesystem::path conversation_directory, std::ostream& protocol_out )
            : session_( std::move( config ), *this ), store_( std::move( conversation_directory ) ), out_( protocol_out )
        {
        }

        /// @copydoc ChatSession::registerTool
        void registerTool( std::string name, std::function<std::string( const std::string& )> handler )
        {
            session_.registerTool( std::move( name ), std::move( handler ) );
        }

        /**
         * @brief Load the configured model, then answer requests until @p requests ends.
         *
         * @return The process's exit code.
         */
        int run( std::istream& requests )
        {
            std::jthread reader( [this, &requests] { readRequests( requests ); } );

            if ( !session_.config().model_name.empty() )
            {
                activity_ = Activity::Load;
                session_.open();
                activity_ = Activity::None;
            }

            session_.newConversation();

            while ( const std::optional<nlohmann::json> request = queue_.pop() )
            {
                try
                {
                    handle( *request );
                }
                catch ( const std::exception& failure )
                {
                    // A request that fails says so and leaves the session answering the next.
                    activity_ = Activity::None;
                    busy_ = false;
                    sendError( std::format( "Something went wrong: {}", failure.what() ), nullptr );
                    sendState();
                }
            }

            return 0;
        }

        // ---- SessionEvents ----------------------------------------------------------------

        void info( std::string_view text ) override
        {
            send( "info", { { "text", text } } );
        }

        void error( std::string_view text ) override
        {
            sendError( text, activityName() );
        }

        void log( std::string_view text ) override
        {
            send( "log", { { "level", "info" }, { "text", text } } );
        }

        void loadBegan( std::string_view label ) override
        {
            load_label_ = label;
            sendState();
            send( "progress", { { "task", loadTask() }, { "label", load_label_ },
                { "done", nullptr }, { "total", nullptr } } );
        }

        void loadEnded() override
        {
            send( "progress", { { "task", loadTask() }, { "label", load_label_ }, { "done", 1 }, { "total", 1 } } );
        }

        void installProgress( std::uint64_t received, std::uint64_t total ) override
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

        void turnBegan( bool ) override
        {
        }

        void roundBegan() override
        {
            ++round_;
        }

        void tokenGenerated() override
        {
            // A stop asked for before the turn's own stop source existed reaches it here, on the next token.
            if ( stop_asked_.load() )
            {
                session_.stop();
            }
        }

        void text( TextChannel channel, std::string_view text ) override
        {
            send( "turn.text", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
                { "channel", channel == TextChannel::Thought ? "thought" : "answer" }, { "text", text } } );
        }

        void thoughtBreak() override
        {
            send( "turn.thought_break", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ } } );
        }

        void thoughtClosed() override
        {
        }

        void toolCallOpened() override
        {
        }

        void roundEnded( const RoundStats& round ) override
        {
            turn_rounds_.push_back( round );

            send( "turn.round_end", {
                { "stream_id", kStreamId },
                { "turn_id", turn_id_ },
                { "round", round_ },
                { "prefill_ms", std::round( round.prefill_time_ms ) },
                { "tokens_per_second", std::round( round.decode_tokens_per_second * 10.0f ) / 10.0f },
                { "tokens", round.tokens_generated },
                { "thought_tokens", round.reasoning_tokens },
                { "finish", round.ended_in_tool_call ? std::string_view( "tool_call" ) : finishName( round.finish_status ) } } );
        }

        ToolDecision approveToolCall( const ToolCall& call ) override
        {
            const std::string call_id = announceCall( call );

            std::promise<ToolDecision> answer;
            std::future<ToolDecision> decision = answer.get_future();

            {
                std::lock_guard<std::mutex> lock( approval_mutex_ );

                if ( stop_asked_.load() )
                {
                    return ToolDecision::Deny;
                }

                approval_.emplace( Approval{ turn_id_, call_id, std::move( answer ) } );
            }

            send( "turn.tool_approval", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
                { "choices", { "allow_once", "allow_conversation", "deny" } } } );

            const ToolDecision decided = decision.get();

            // A stop answers the wait with Deny that the person never gave, so the stored call keeps no decision.
            if ( !stop_asked_.load() )
            {
                storedCall( call_id )[ "decision" ] = decisionName( decided );
            }

            return decided;
        }

        void toolCalled( const ToolCall& call, std::string_view result, bool is_error ) override
        {
            // A tool allowed for the conversation was not asked about, so it has not been announced yet.
            const std::string call_id = announced_call_id_.empty() ? announceCall( call ) : announced_call_id_;
            announced_call_id_.clear();

            send( "turn.tool_result", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
                { "result", result }, { "is_error", is_error } } );

            nlohmann::json& stored = storedCall( call_id );
            stored[ "result" ] = result;
            stored[ "is_error" ] = is_error;
        }

        void turnEnded( const ParsedResponse& display, std::string_view ) override
        {
            turn_display_ = display;

            send( "turn.end", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "finish", turnFinish() },
                { "answer", display.answer } } );
        }

    private:

        static constexpr std::string_view kStreamId = "stream-1";

        enum class Activity
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
            std::promise<ToolDecision> answer;
        };

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
            session_.stop();

            // The session reads the stop before it runs the call, so this answer is never acted on.
            if ( approval_.has_value() )
            {
                approval_->answer.set_value( ToolDecision::Deny );
                approval_.reset();
            }
        }

        void decideToolCall( const nlohmann::json& request )
        {
            const std::string turn_id = request.value( "turn_id", std::string{} );
            const std::string call_id = request.value( "call_id", std::string{} );
            const std::string decision = request.value( "decision", std::string{} );

            std::optional<ToolDecision> parsed;

            if ( decision == "allow_once" )
            {
                parsed = ToolDecision::AllowOnce;
            }
            else if ( decision == "allow_conversation" )
            {
                parsed = ToolDecision::AllowConversation;
            }
            else if ( decision == "deny" )
            {
                parsed = ToolDecision::Deny;
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
                startNewConversation();
                sendState();
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
                session_.removeModel( request.value( "name", std::string{} ) );
                sendModels( false );
            }
            else if ( type == "settings.set" )
            {
                activity_ = Activity::Settings;
                changeSetting( request );
                activity_ = Activity::None;
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

        void respond( const nlohmann::json& request )
        {
            const std::string text = request.value( "text", std::string{} );

            if ( !session_.modelIsResident() )
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
            turn_rounds_.clear();
            turn_calls_ = nlohmann::json::array();
            turn_display_ = ParsedResponse{};
            announced_call_id_.clear();

            activity_ = Activity::Turn;
            sendState();

            send( "turn.begin", { { "stream_id", kStreamId }, { "turn_id", turn_id_ },
                { "conversation_id", conversation_id_ } } );

            session_.respond( text );

            activity_ = Activity::None;

            recordTurn( text );
            busy_ = false;

            sendState();
            sendConversations();
        }

        /// The turn joins the stored conversation as the page shows it, and as the model read it.
        void recordTurn( const std::string& text )
        {
            conversation_messages_.push_back( { { "role", "user" }, { "content", text }, { "thought", nullptr },
                { "tool_calls", nullptr }, { "attachments", nullptr }, { "stats", nullptr } } );

            float prefill_ms = 0.0f;
            int tokens = 0;
            double decode_seconds = 0.0;

            for ( const RoundStats& round : turn_rounds_ )
            {
                prefill_ms += round.prefill_time_ms;
                tokens += round.tokens_generated;

                if ( round.decode_tokens_per_second > 0.0f )
                {
                    decode_seconds += round.tokens_generated / static_cast<double>( round.decode_tokens_per_second );
                }
            }

            const double rate = decode_seconds > 0.0 ? std::round( 10.0 * tokens / decode_seconds ) / 10.0 : 0.0;

            const nlohmann::json stats = { { "prefill_ms", std::round( prefill_ms ) }, { "tokens_per_second", rate },
                { "tokens", tokens }, { "rounds", turn_rounds_.size() }, { "finish", turnFinish() } };

            conversation_messages_.push_back( {
                { "role", "assistant" },
                { "content", turn_display_.answer },
                { "thought", turn_display_.thinking.empty() ? nlohmann::json( nullptr ) : nlohmann::json( turn_display_.thinking ) },
                { "tool_calls", turn_calls_.empty() ? nlohmann::json( nullptr ) : turn_calls_ },
                { "attachments", nullptr },
                { "stats", stats } } );

            saveConversation();
        }

        void saveConversation()
        {
            nlohmann::json turns = nlohmann::json::array();

            for ( const ChatMessage& turn : session_.history() )
            {
                if ( turn.role != MessageRole::System )
                {
                    turns.push_back( turnToJson( turn ) );
                }
            }

            const nlohmann::json conversation = {
                { "v", 1 },
                { "id", conversation_id_ },
                { "title", conversation_title_ },
                { "model", session_.modelName() },
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
            session_.newConversation();
            forgetConversation();
        }

        /// The session's history no longer belongs to a stored conversation; the next turn starts one.
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

            std::vector<ChatMessage> turns;

            for ( const nlohmann::json& turn : conversation->value( "turns", nlohmann::json::array() ) )
            {
                turns.push_back( turnFromJson( turn ) );
            }

            session_.openConversation( std::move( turns ) );

            conversation_id_ = id;
            conversation_title_ = conversation->value( "title", std::string{} );
            conversation_created_ = conversation->value( "created", std::string{} );
            conversation_messages_ = conversation->value( "messages", nlohmann::json::array() );

            send( "conversation.loaded", { { "id", id }, { "title", conversation_title_ },
                { "model", conversation->value( "model", std::string{} ) }, { "messages", conversation_messages_ } } );

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

        void loadModel( const nlohmann::json& request )
        {
            std::optional<QuantizationMode> quantization;

            if ( request.contains( "quantization" ) && request[ "quantization" ].is_string() )
            {
                quantization = parseQuantization( request[ "quantization" ].get<std::string>() );

                if ( !quantization.has_value() )
                {
                    sendError( "Quantization is none, fp8, fp4 or q4_0.", "load" );

                    return;
                }
            }

            activity_ = Activity::Load;
            session_.loadModel( request.value( "name", std::string{} ), quantization );
            activity_ = Activity::None;

            // A load that switched models cleared the history, and with it the conversation it belonged to.
            if ( !historyHasUserTurn() )
            {
                forgetConversation();
            }

            sendState();
        }

        void installModel( const std::string& name )
        {
            install_name_ = name;
            install_total_ = 0;
            install_percent_ = -1;

            activity_ = Activity::Install;
            sendState();
            session_.installModel( name );
            activity_ = Activity::None;

            if ( !historyHasUserTurn() )
            {
                forgetConversation();
            }

            sendModels( false );
            sendState();
        }

        bool historyHasUserTurn() const
        {
            return std::ranges::any_of( session_.history(),
                []( const ChatMessage& turn ) { return turn.role == MessageRole::User; } );
        }

        /// One change per request (ChatDesktop.md 4.2); what the session says about it arrives as its own events.
        void changeSetting( const nlohmann::json& request )
        {
            if ( request.contains( "context" ) )
            {
                changeContext( request[ "context" ] );
            }
            else if ( request.contains( "thinking" ) )
            {
                if ( !request[ "thinking" ].is_boolean() || !session_.setThinking( request[ "thinking" ].get<bool>() ) )
                {
                    sendError( "This model has no reasoning channel.", "settings" );
                }
            }
            else if ( request.contains( "effort" ) )
            {
                changeEffort( request[ "effort" ] );
            }
            else if ( request.contains( "detail" ) )
            {
                const auto level = request[ "detail" ].is_string()
                    ? parseDetailLevel( request[ "detail" ].get<std::string>() )
                    : std::nullopt;

                if ( !level.has_value() )
                {
                    sendError( "Detail is off, thoughts or all.", "settings" );

                    return;
                }

                session_.setDetail( *level );
            }
            else if ( request.contains( "sampling" ) )
            {
                changeSampling( request[ "sampling" ] );
            }
            else if ( request.contains( "seed" ) )
            {
                changeSeed( request[ "seed" ] );
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
                session_.setAutomaticContext();

                return;
            }

            if ( !value.is_number_unsigned() || value.get<std::size_t>() == 0 )
            {
                sendError( "A context window is a number of tokens, or auto.", "settings" );

                return;
            }

            session_.setContextLength( value.get<std::size_t>() );
        }

        void changeEffort( const nlohmann::json& value )
        {
            if ( !session_.config().thinking_capable || !session_.templateHasEffortLevels() )
            {
                sendError( "This model's thinking is on or off; it has no effort levels.", "settings" );

                return;
            }

            if ( !value.is_number_integer() || !session_.setThinkingEffort( value.get<int>() ) )
            {
                sendError( "Effort is a level from 1 to 5.", "settings" );
            }
        }

        /// The bounds are the sampler's, as the console's /set has them.
        void changeSampling( const nlohmann::json& change )
        {
            const std::string key = change.is_object() ? change.value( "key", std::string{} ) : std::string{};
            const nlohmann::json value = change.is_object() && change.contains( "value" ) ? change[ "value" ] : nlohmann::json{};
            const bool follow_model = value.is_string() && value.get<std::string>() == "model";

            const auto number = [&value]( double low, double high ) -> std::optional<float>
                {
                    if ( !value.is_number() || value.get<double>() < low || value.get<double>() > high )
                    {
                        return std::nullopt;
                    }

                    return value.get<float>();
                };

            if ( key == "temperature" )
            {
                const auto parsed = number( 0.0, 5.0 );

                if ( !follow_model && !parsed )
                {
                    sendError( "Temperature is from 0 to 5; 0 always picks the likeliest word.", "settings" );

                    return;
                }

                session_.setTemperature( follow_model ? std::nullopt : parsed );
            }
            else if ( key == "top_k" )
            {
                if ( !follow_model && ( !value.is_number_integer() || value.get<long long>() < 0 ) )
                {
                    sendError( "top_k is a whole number, 0 or more; 0 turns it off.", "settings" );

                    return;
                }

                session_.setTopK( follow_model ? std::nullopt : std::optional<int>( value.get<int>() ) );
            }
            else if ( key == "top_p" )
            {
                const auto parsed = number( 0.0, 1.0 );

                if ( !follow_model && !parsed )
                {
                    sendError( "top_p is from 0 to 1; 1 turns it off.", "settings" );

                    return;
                }

                session_.setTopP( follow_model ? std::nullopt : parsed );
            }
            else if ( key == "presence_penalty" )
            {
                const auto parsed = number( -2.0, 2.0 );

                if ( !follow_model && !parsed )
                {
                    sendError( "presence_penalty is from -2 to 2; 0 turns it off.", "settings" );

                    return;
                }

                session_.setPresencePenalty( follow_model ? std::nullopt : parsed );
            }
            else
            {
                sendError( "That is not a sampling setting.", "settings" );
            }
        }

        void changeSeed( const nlohmann::json& value )
        {
            if ( !value.is_number_unsigned() )
            {
                sendError( "A seed is a whole number, 0 or more.", "settings" );

                return;
            }

            const std::uint64_t seed = value.get<std::uint64_t>();

            if ( !session_.seed( seed ) )
            {
                sendError( "No model is loaded.", "settings" );

                return;
            }

            info( std::format( "Sampler reseeded ({}).", seed ) );
        }

        /**
         * @brief Every attachment is refused, by name: no model this build runs takes an image or audio yet
         *        (InstalledModel's isSupportedModality), so session.state lists no modality to accept one for.
         */
        void refuseAttachment( const nlohmann::json& request )
        {
            const std::string kind = request.value( "kind", std::string{} );

            std::string reason;

            if ( !session_.modelIsResident() )
            {
                reason = "No model is loaded to read it.";
            }
            else if ( kind == "image" || kind == "audio" )
            {
                reason = std::format( "{} cannot take {}.", session_.modelName(), kind == "image" ? "images" : "audio" );
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
            const ChatConfig& config = session_.config();
            const bool resident = session_.modelIsResident();
            const bool effort_levels = resident && config.thinking_capable && session_.templateHasEffortLevels();

            std::string thinking = "off";

            if ( resident && config.thinking_capable && config.show_thinking )
            {
                thinking = effort_levels ? std::string( thinkingEffort( config.thinking_effort ).name ) : "on";
            }

            nlohmann::ordered_json sampling = nlohmann::ordered_json::object();

            if ( resident )
            {
                const Dnn::SamplingParams effective = sessionSampling( config );

                sampling[ "temperature" ] = { { "value", effective.temperature },
                    { "is_model_default", !config.temperature.has_value() } };
                sampling[ "top_k" ] = { { "value", effective.top_k }, { "is_model_default", !config.top_k.has_value() } };
                sampling[ "top_p" ] = { { "value", effective.top_p }, { "is_model_default", !config.top_p.has_value() } };
                sampling[ "presence_penalty" ] = { { "value", effective.presence_penalty },
                    { "is_model_default", !config.presence_penalty.has_value() } };
            }

            // The modalities this build runs for the resident model, from the library's own answer.
            nlohmann::ordered_json modalities = nlohmann::ordered_json::array();

            if ( resident )
            {
                if ( const std::optional<InstalledModel> installed = InstalledModel::find( session_.modelName() ) )
                {
                    for ( const std::string& modality : installed->capabilities().modalities )
                    {
                        modalities.push_back( modality );
                    }
                }
            }

            const bool busy = busy_.load() || activity_ != Activity::None;

            const nlohmann::ordered_json stream = {
                { "stream_id", kStreamId },
                { "parent_stream_id", nullptr },
                { "conversation_id", conversation_id_.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( conversation_id_ ) },
                { "busy", busy } };

            send( "session.state", {
                { "model", resident ? nlohmann::ordered_json( session_.modelName() ) : nlohmann::ordered_json( nullptr ) },
                { "no_model_reason", config.no_model_reason.empty() || resident
                    ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( config.no_model_reason ) },
                { "context_length", config.context_length },
                { "context_automatic", config.context_is_automatic },
                { "thinking_capable", resident && config.thinking_capable },
                { "thinking", thinking },
                { "effort", effort_levels ? nlohmann::ordered_json( config.thinking_effort ) : nlohmann::ordered_json( nullptr ) },
                { "effort_levels", effort_levels },
                { "detail", detailLevelName( config.detail ) },
                { "sampling", std::move( sampling ) },
                { "modalities", std::move( modalities ) },
                { "busy", busy },
                { "streams", nlohmann::ordered_json::array( { stream } ) } } );
        }

        void sendConversations()
        {
            send( "conversations.list", { { "items", store_.list() } } );
        }

        /// What is installed, and, when asked for, what the publisher offers.
        void sendModels( bool online )
        {
            nlohmann::json installed = nlohmann::json::array();

            for ( const InstalledModel& model : InstalledModel::list() )
            {
                const std::optional<QuantizationMode> stored = model.storedWeightQuantization();

                std::string family = Mila::Dnn::toString( model.family() );
                std::ranges::transform( family, family.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

                installed.push_back( {
                    { "name", model.name() },
                    { "size_bytes", model.stored().bytes_on_disk },
                    { "quantization", stored.has_value()
                        ? std::string( quantizationName( *stored ) ) : model.stored().record.weight_quantization },
                    { "family", family } } );
            }

            nlohmann::json published = nullptr;

            if ( online )
            {
                try
                {
                    published = nlohmann::json::array();

                    for ( const PublishedModel& model : listPublishedModels( std::string( Mila::Distribution::kDefaultHubOwner ) ) )
                    {
                        published.push_back( {
                            { "name", model.name },
                            { "size_bytes", model.download_bytes },
                            { "quantization", model.quantization },
                            { "installed", model.installed } } );
                    }
                }
                catch ( const std::exception& failure )
                {
                    published = nullptr;
                    sendError( std::format( "Could not list the models that can be installed: {}", failure.what() ), nullptr );
                }
            }

            send( "models.list", { { "installed", std::move( installed ) }, { "published", std::move( published ) } } );
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

        nlohmann::ordered_json activityName() const
        {
            switch ( activity_ )
            {
                case Activity::Turn:     return "turn";
                case Activity::Load:     return "load";
                case Activity::Install:  return "install";
                case Activity::Settings: return "settings";
                case Activity::None:     break;
            }

            return nullptr;
        }

        /// A context change rebuilds the model it already holds; anything else that loads, loads.
        std::string_view loadTask() const
        {
            return activity_ == Activity::Settings ? "rebuild" : "load";
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

        /**
         * @brief How the turn ended, from its last round.
         *
         * A reply stopped while a call waited ends on a round that finished its call, so the stop is what says it
         * was cancelled. A turn with no round had no room in the context to start one.
         */
        std::string_view turnFinish() const
        {
            const std::vector<RoundStats>& rounds = session_.lastTurnRounds();

            if ( rounds.empty() )
            {
                return "context";
            }

            if ( stop_asked_.load() && rounds.back().ended_in_tool_call )
            {
                return "cancelled";
            }

            return finishName( rounds.back().finish_status );
        }

        std::string announceCall( const ToolCall& call )
        {
            announced_call_id_ = std::format( "call-{}", ++call_number_ );
            const std::string summary = formatToolSummary( session_.tools(), call.name, call.arguments );

            send( "turn.tool_call", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
                { "call_id", announced_call_id_ }, { "name", call.name }, { "summary", summary },
                { "arguments", call.arguments } } );

            // Stored as announced, so a call the person stopped before it ran is kept with no result (6.3).
            turn_calls_.push_back( { { "call_id", announced_call_id_ }, { "name", call.name }, { "summary", summary },
                { "arguments", call.arguments }, { "decision", nullptr }, { "result", nullptr }, { "is_error", false } } );

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

        static std::string_view decisionName( ToolDecision decision )
        {
            switch ( decision )
            {
                case ToolDecision::AllowOnce:
                    return "allow_once";

                case ToolDecision::AllowConversation:
                    return "allow_conversation";

                case ToolDecision::Deny:
                    return "deny";
            }

            return "deny";
        }

        /// The first user message, cut at a word under 60 characters (ChatDesktop.md 6.3).
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

        static std::string timestamp()
        {
            const auto now = std::chrono::floor<std::chrono::seconds>( std::chrono::system_clock::now() );

            return std::format( "{:%FT%TZ}", now );
        }

        static std::string_view roleName( MessageRole role )
        {
            switch ( role )
            {
                case MessageRole::System:    return "system";
                case MessageRole::User:      return "user";
                case MessageRole::Assistant: return "assistant";
                case MessageRole::Tool:      return "tool";
            }

            return "user";
        }

        static nlohmann::json turnToJson( const ChatMessage& turn )
        {
            nlohmann::json calls = nlohmann::json::array();

            for ( const ToolCall& call : turn.tool_calls )
            {
                calls.push_back( { { "name", call.name }, { "arguments", call.arguments } } );
            }

            return { { "role", roleName( turn.role ) }, { "content", turn.content }, { "tool_calls", std::move( calls ) },
                { "tool_name", turn.tool_name }, { "reasoning", turn.reasoning } };
        }

        static ChatMessage turnFromJson( const nlohmann::json& stored )
        {
            const std::string role = stored.value( "role", std::string{} );

            ChatMessage turn;
            turn.role = role == "assistant" ? MessageRole::Assistant
                : role == "tool" ? MessageRole::Tool
                : MessageRole::User;
            turn.content = stored.value( "content", std::string{} );
            turn.tool_name = stored.value( "tool_name", std::string{} );
            turn.reasoning = stored.value( "reasoning", std::string{} );

            for ( const nlohmann::json& call : stored.value( "tool_calls", nlohmann::json::array() ) )
            {
                turn.tool_calls.push_back( { call.value( "name", std::string{} ), call.value( "arguments", std::string{} ) } );
            }

            return turn;
        }

        ChatSession session_;
        ConversationStore store_;
        std::ostream& out_;
        std::mutex output_mutex_;

        RequestQueue queue_;

        // Set by the reading thread when it accepts a turn, cleared by this thread when the turn is done.
        std::atomic<bool> busy_{ false };
        std::atomic<bool> stop_asked_{ false };

        // Read by the reading thread only under approval_mutex_.
        std::mutex approval_mutex_;
        std::optional<Approval> approval_;

        Activity activity_{ Activity::None };
        std::string load_label_;

        std::string install_name_;
        std::uint64_t install_total_{ 0 };
        int install_percent_{ -1 };

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
        std::vector<RoundStats> turn_rounds_;
        nlohmann::json turn_calls_ = nlohmann::json::array();
        ParsedResponse turn_display_;
        std::string announced_call_id_;
    };
}
