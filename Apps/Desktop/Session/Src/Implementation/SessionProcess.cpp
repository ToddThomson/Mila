/**
 * @file SessionProcess.cpp
 * @brief The bodies of SessionProcess's members; the interface is Desktop.SessionProcess.ixx.
 */

module Desktop.SessionProcess;

import std;
import nlohmann.json;
import Desktop.CommandTool;
import Desktop.ConversationStore;
import Desktop.ConversationSummary;
import Desktop.Creations;
import Desktop.EventLog;
import Desktop.ProjectStore;
import Desktop.WorkspaceFolder;
import Desktop.ExposureRecord;
import Desktop.MemoryHolder;
import Desktop.MilaLicense;
import Desktop.Models;
import Desktop.OfferAnswer;
import Desktop.Offers;
import Desktop.PromptFile;
import Desktop.ProgramSetup;
import Desktop.PromptTool;
import Desktop.ResidentAgent;
import Desktop.RoundTimer;
import Desktop.SearchKeyFile;
import Desktop.SearchKeyStore;
import Desktop.Settings;
import Desktop.SettingsFile;
import Desktop.StoreFile;
import Desktop.TaskPlan;
import Desktop.Understanding;
import Desktop.WebSearch;
import Mila;

namespace Mila::Desktop
{
    using Mila::Dnn::dim_t;
    using Mila::Dnn::GenerateStatus;
    using Mila::Dnn::WeightQuantization;
    using Mila::Dnn::Compute::DeviceOutOfMemoryError;
    using Mila::Dnn::Conversation::Role;
    using Mila::Dnn::Conversation::ToolCall;
    using Mila::Dnn::Conversation::Turn;
    using Mila::Deployment::DeploymentRequest;
    using Mila::Deployment::DeploymentRefusedError;
    using Mila::Deployment::InstalledModel;
    using Mila::Agents::AgentEvent;
    using Mila::Agents::ToolAnswer;

    namespace
    {
        /// The text without the white space around it.
        std::string trimmed( std::string_view text )
        {
            const auto first = text.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            const auto last = text.find_last_not_of( " \t\r\n" );

            return std::string( text.substr( first, last - first + 1 ) );
        }

        /// A project's folder entry's access: `read_write` is *Read and change*, anything else reads.
        Mila::Agents::FolderAccess folderAccessOf( const nlohmann::json& folder )
        {
            return folder.is_object() && folder.value( "access", std::string{} ) == "read_write"
                ? Mila::Agents::FolderAccess::ReadWrite : Mila::Agents::FolderAccess::Read;
        }

        /**
         * @brief A project's folders as the page sent them, checked: each a folder on this computer, named once, with
         *        its access `read` or `read_write`.
         *
         * @return The folders as the record keeps them, or nothing when any entry is not one.
         */
        std::optional<nlohmann::json> projectFolders( const nlohmann::json& sent )
        {
            if ( !sent.is_array() )
            {
                return std::nullopt;
            }

            nlohmann::json folders = nlohmann::json::array();
            std::set<std::string> named;

            for ( const nlohmann::json& entry : sent )
            {
                if ( !entry.is_object() || !entry.contains( "path" ) || !entry[ "path" ].is_string() )
                {
                    return std::nullopt;
                }

                const std::string path = entry[ "path" ].get<std::string>();
                const std::string access = entry.value( "access", std::string( "read" ) );
                std::error_code missing;

                if ( path.empty() || ( access != "read" && access != "read_write" ) || !named.insert( path ).second
                    || !std::filesystem::is_directory( std::filesystem::path( std::u8string( path.begin(), path.end() ) ), missing ) )
                {
                    return std::nullopt;
                }

                folders.push_back( { { "path", path }, { "access", access } } );
            }

            return folders;
        }
    }

    SessionProcess::SessionProcess( Settings settings, SettingsFile settings_file, std::vector<std::string> problems,
        std::expected<MilaLicense, std::string> license, Mila::Deployment::AllowedDevices devices,
        std::filesystem::path workspace_directory, std::filesystem::path prompt_directory, std::ostream& protocol_out )
        : settings_( std::move( settings ) ), settings_file_( std::move( settings_file ) ),
          settings_problems_( std::move( problems ) ), license_( std::move( license ) ), devices_( std::move( devices ) ),
          store_( conversationsFolder( workspace_directory ) ), projects_( projectsFolder( workspace_directory ) ),
          activity_( activityFile( workspace_directory ) ), prompts_( std::move( prompt_directory ) ), out_( protocol_out ),
          search_key_file_( workspace_directory / "search.json" )
    {
        session().onAsk( [this]( const Mila::Agents::ToolQuestion& question ) { return askAboutCall( question ); } );
        session().withSampling( settings_.sampling );
        useDetail( settings_.detail );

        // A plan's change is shown at once, while the reply that made it is still running.
        // The plan's tools and request_folder answer in Mila's own words, so they expose nothing (ToolCalling.md 3.3).
        registerTool( std::string( TaskPlan::kProposeName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                std::string result = plan_.propose( arguments );
                sendState();

                return result;
            }, Mila::Agents::ResultOrigin::Application );

        registerTool( std::string( TaskPlan::kUpdateName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                std::string result = plan_.update( arguments );
                sendState();

                return result;
            }, Mila::Agents::ResultOrigin::Application );

        useMiloTool( Mila::Agents::WebPageReader::tool() );

        // Registered once; each call reaches the folder of the conversation it belongs to, through its grant.
        for ( Mila::Agents::Tool& tool : Mila::Agents::FileTools::tools() )
        {
            useMiloTool( std::move( tool ) );
        }

        // Runs only once the person has chosen the folder, which their answer to the call carries.
        registerTool( std::string( kRequestFolderName ), {}, [this]( const std::string&, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return grantChosenFolder();
            }, Mila::Agents::ResultOrigin::Application );

        // The workspace's own records: open_place answers in Mila's words, and find_records says per call whether a
        // match came from a chat that read from outside (AssistantInterface.md 5.2 and 5.6).
        registerTool( std::string( kOpenPlaceName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return openPlace( arguments );
            }, Mila::Agents::ResultOrigin::Application );

        registerTool( std::string( kFindRecordsName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return findRecordsIn( arguments );
            }, Mila::Agents::ResultOrigin::Application );

        // Mila proposes and the person decides (AssistantInterface.md 6): both answer in Mila's own words.
        registerTool( std::string( kProposeProjectName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return proposeProject( arguments );
            }, Mila::Agents::ResultOrigin::Application );

        registerTool( std::string( kMoveToProjectName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return moveToProject( arguments );
            }, Mila::Agents::ResultOrigin::Application );

        // A note is written in Mila's own words about the store; the notes read say per call where they came from (5.6).
        registerTool( std::string( kNoteUnderstandingName ), {}, [this]( const std::string& arguments, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return noteUnderstanding( arguments );
            }, Mila::Agents::ResultOrigin::Application );

        registerTool( std::string( kReadProjectNotesName ), {}, [this]( const std::string&, const Mila::Agents::ToolGrant&, std::stop_token )
            {
                return readProjectNotes();
            }, Mila::Agents::ResultOrigin::Outside );

        // A key kept in search.json before the credential store held it moves there once, and the file goes. Refused,
        // the file stays and its key is still used, so the person's key is never lost.
        if ( const std::optional<std::string> kept_in_file = search_key_file_.read() )
        {
            try
            {
                search_key_store_.write( *kept_in_file );
                search_key_file_.remove();
            }
            catch ( const std::runtime_error& refused )
            {
                Mila::Logging::Logger::info( std::format( "The search key stays in search.json: {}", refused.what() ) );
            }
        }

        std::optional<std::string> key = search_key_store_.read();

        if ( !key )
        {
            key = search_key_file_.read();
        }

        if ( key )
        {
            useSearchKey( *key );
        }

        if ( settings_.run_commands )
        {
            useCommands();
        }
    }

    int SessionProcess::run( std::istream& requests )
    {
        std::jthread reader( [this, &requests] { readRequests( requests ); } );

        for ( const std::string& problem : settings_problems_ )
        {
            sendError( problem, nullptr );
        }

        useChat( {} );

        if ( !settings_.model.empty() && licenseAccepted() )
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

    Mila::Agents::Session& SessionProcess::session() noexcept
    {
        return model_.session();
    }

    const Mila::Agents::Session& SessionProcess::session() const noexcept
    {
        return model_.session();
    }

    void SessionProcess::readRequests( std::istream& requests )
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
            else if ( type == "app.active" )
            {
                // Here rather than in the queue, so that a reply being written does not delay it.
                memory_holder_.activate();
            }
            else if ( type == "app.background" )
            {
                memory_holder_.deactivate();
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

    void SessionProcess::stopReply()
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

    void SessionProcess::decideToolCall( const nlohmann::json& request )
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
            || approval_->turn_id != turn_id || approval_->call_id != call_id
            || ( *parsed == ToolAnswer::ForConversation && !approval_->conversation_offered ) )
        {
            std::cerr << "Ignored a tool decision for a call that is not waiting.\n";

            return;
        }

        if ( !approval_->places.empty() && *parsed != ToolAnswer::Deny )
        {
            // Only one of the places the person was shown, named by its view and id.
            const nlohmann::json named = request.value( "place", nlohmann::json( nullptr ) );
            const auto offered = std::ranges::find_if( approval_->places, [&named]( const nlohmann::json& place )
                {
                    return named.is_object() && named.value( "view", nlohmann::json( nullptr ) ) == place[ "view" ]
                        && named.value( "id", nlohmann::json( nullptr ) ) == place[ "id" ];
                } );

            if ( offered == approval_->places.end() )
            {
                std::cerr << "Ignored a place's answer that names none of the places offered.\n";

                return;
            }

            chosen_place_ = *offered;
        }

        if ( approval_->folder_asked && *parsed != ToolAnswer::Deny )
        {
            const std::string folder = request.contains( "folder" ) && request[ "folder" ].is_string()
                ? request[ "folder" ].get<std::string>() : std::string{};

            if ( folder.empty() )
            {
                std::cerr << "Ignored a folder request's answer that names no folder.\n";

                return;
            }

            chosen_folder_ = folder;
        }

        approval_->answer.set_value( *parsed );
        approval_.reset();
    }

    void SessionProcess::handle( const nlohmann::json& request )
    {
        const std::string type = request[ "type" ].get<std::string>();

        if ( type == "ready" )
        {
            sendLicense();
            sendState();
            sendProjects();
            sendConversations();
            sendActivity( {} );
            sendModels( false );
        }
        else if ( type == "license.accept" )
        {
            acceptLicense( request.value( "sha256", std::string{} ) );
        }
        else if ( type == "turn.send" )
        {
            respond( request );
        }
        else if ( type == "conversation.new" )
        {
            newConversation( request );
        }
        else if ( type == "conversation.move" )
        {
            moveConversation( request );
        }
        else if ( type == "project.new" )
        {
            newProject( request );
        }
        else if ( type == "project.update" )
        {
            updateProject( request );
        }
        else if ( type == "project.delete" )
        {
            deleteProject( request );
        }
        else if ( type == "suggestion.decide" )
        {
            decideSuggestion( request );
        }
        else if ( type == "project.undo" )
        {
            undoProject( request );
        }
        else if ( type == "note.decide" )
        {
            decideNote( request );
        }
        else if ( type == "activity.open" )
        {
            const nlohmann::json before = request.value( "before", nlohmann::json( nullptr ) );
            sendActivity( before.is_string() ? before.get<std::string>() : std::string{} );
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
        else if ( type == "reach.remove" )
        {
            removeReach( request );
        }
        else if ( type == "creation.open" )
        {
            openCreation( request );
        }
        else if ( type == "creation.keep" )
        {
            keepReply( request );
        }
        else if ( type == "conversation.link_opened" )
        {
            recordLinkOpened( request.value( "url", std::string{} ) );
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

    void SessionProcess::respond( const nlohmann::json& request )
    {
        const std::string text = request.value( "text", std::string{} );

        if ( !licenseAccepted() )
        {
            busy_ = false;
            sendError( std::string( kLicenseFirst ), "turn" );
            sendState();

            return;
        }

        // An answer to Mila's offer is read here, by app code, and needs no reply (AssistantInterface.md 5.4).
        if ( answerOffer( text ) )
        {
            busy_ = false;
            sendState();

            return;
        }

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

        // Stored at once with the person's message, so the list shows the conversation on send and a quit during
        // the reply keeps what they wrote; the reply is stored when it ends (Todd, 2026-11-13).
        conversation_messages_.push_back( { { "role", "user" }, { "content", text }, { "thought", nullptr },
            { "tool_calls", nullptr }, { "attachments", nullptr }, { "stats", nullptr } } );

        Turn sent;
        sent.role = Role::User;
        sent.content = text;
        saveConversation( &sent );
        sendConversations();

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

        Mila::Agents::Agent& agent = model_.agent();
        RoundTimer timer;

        const auto began = std::chrono::steady_clock::now();
        memory_holder_.use( agent.model().model().getDeviceId() );

        // What a person waits for before the round starts, which the round's own prefill time leaves out.
        Mila::Logging::Logger::info( std::format( "{}: the model's memory held after {:.0f} ms", turn_id_,
            std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - began ).count() ) );

        // After the hold, so the page shows "Getting ready" in the turn's place while memory comes back.
        send( "turn.begin", { { "stream_id", kStreamId }, { "turn_id", turn_id_ },
            { "conversation_id", conversation_id_ } } );

        std::optional<Mila::Agents::Response> responded;

        // A new chat in a project is given the project's note after its first message, as data (5.6).
        const std::vector<ToolCall> placed = placedCalls();
        placing_notes_ = !placed.empty();

        try
        {
            responded = agent.respond( text, placed,
                [&]( const AgentEvent& event ) { onAgentEvent( event, timer, stop ); }, stop );
            placing_notes_ = false;
        }
        catch ( const DeviceOutOfMemoryError& )
        {
            placing_notes_ = false;
            // The agent left the conversation as it was and the model loaded, so the message stored on send is
            // taken back out -- with the conversation, when it was its first -- and the person can send it again.
            memory_holder_.idle();
            conversation_messages_.erase( conversation_messages_.size() - 1 );

            if ( conversation_messages_.empty() )
            {
                store_.remove( conversation_id_ );
                forgetConversation();
            }
            else
            {
                saveConversation();
            }

            sendConversations();
            send( "turn.end", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "finish", "error" },
                { "answer", turn_.answer } } );
            sendError( "Another application is using the GPU's memory, so Mila could not answer. Your message "
                "was not added to the conversation; send it again once that application is done.", "turn" );

            work_ = Work::None;

            if ( std::exchange( tools_changed_in_call_, false ) )
            {
                offerTools();
            }
            busy_ = false;
            sendState();

            return;
        }

        memory_holder_.idle();

        const Mila::Agents::Response& answered = *responded;

        if ( answered.rounds == 0 && answered.status == GenerateStatus::ContextOverflow )
        {
            send( "info", { { "text", std::format( "The prompt is {} tokens and the context is {}, so there is no "
                "room to answer. Start a new conversation, or choose a longer context.",
                agent.model().tokenizer().encode( agent.prompt( text ).text ).size(), agent.plan().contextLength() ) } } );
        }

        // The reply is stored next, as this message of the conversation, which is how the page keeps it.
        send( "turn.end", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "finish", turnFinish() },
            { "answer", turn_.answer }, { "message", conversation_messages_.size() } } );

        work_ = Work::None;

        // A call stopped before its result came in still changed what the chat is offered.
        if ( std::exchange( tools_changed_in_call_, false ) )
        {
            offerTools();
        }

        recordTurn();
        busy_ = false;

        sendState();
        sendConversations();
    }

    void SessionProcess::onAgentEvent( const AgentEvent& event, RoundTimer& timer, const std::stop_token& stop )
    {
        using Kind = AgentEvent::Kind;

        switch ( event.kind )
        {
            case Kind::RoundBegan:
                ++round_;
                context_used_ = event.prompt_tokens;
                sendState();
                timer.begin();
                break;

            case Kind::Token:
                timer.token();
                break;

            case Kind::ReasoningOpened:
                send( "turn.thought_break", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ } } );
                break;

            case Kind::Reasoning:
                // With the round's count so far: the page cannot count tokens from text, and a long thought
                // otherwise shows the person nothing moving.
                send( "turn.text", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
                    { "channel", "thought" }, { "text", event.text }, { "tokens", timer.tokens() } } );
                break;

            case Kind::Answer:
                sendText( "answer", event.text );
                break;

            case Kind::ToolCalled:
                callAnswered( *event.call, event.text, event.is_error, event.exposes );
                break;

            case Kind::MalformedToolCall:
                // A call the template's grammar could not read is said, never dropped: the model meant to call
                // something.
                sendError( std::format( "[tool call not understood: {}]", event.text ), "turn" );
                break;

            case Kind::RoundEnded:
                endRound( timer, event, stop );
                break;

            case Kind::CompactionBegan:
                send( "info", { { "text", std::format(
                    "The conversation holds {} tokens, so it is being compacted.", *event.tokens_before ) } } );
                break;

            case Kind::Compacted:
                context_used_ = event.tokens_after;
                sendState();
                send( "info", { { "text", event.text.empty() && event.dropped_turns > 0
                    ? std::format( "The {} oldest turns were dropped unread, taking the conversation from {} tokens to {}.",
                        event.dropped_turns, *event.tokens_before, *event.tokens_after )
                    : std::format( "Compacted from {} tokens to {}.", *event.tokens_before, *event.tokens_after ) } } );
                break;

            case Kind::CompactionFailed:
                send( "info", { { "text", std::format( "The conversation could not be compacted: {}.", event.text ) } } );
                break;

            case Kind::ReasoningClosed:
            case Kind::ToolCallOpened:
                break;
        }
    }

    void SessionProcess::sendText( std::string_view channel, std::string_view text )
    {
        send( "turn.text", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
            { "channel", channel }, { "text", text } } );
    }

    void SessionProcess::endRound( const RoundTimer& timer, const AgentEvent& event, const std::stop_token& stop )
    {
        const Mila::Dnn::Conversation::Reply& reply = *event.reply;
        const Mila::Agents::Agent& agent = model_.agent();

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

        Mila::Logging::Logger::info( std::format( "{} round {}: first token after {:.0f} ms, {} tokens at {:.1f} per second, {}",
            turn_id_, round_, timer.prefillMilliseconds(), timer.tokens(), timer.decodeRate(),
            called ? std::string_view( "tool_call" ) : finishName( event.status ) ) );

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

    ToolAnswer SessionProcess::askAboutCall( const Mila::Agents::ToolQuestion& question )
    {
        const std::string call_id = announceCall( question.name, question.arguments );

        std::promise<ToolAnswer> answer;
        std::future<ToolAnswer> decision = answer.get_future();

        // A chat that has a folder is not asked for another: the call says which it has.
        const bool folder_asked = question.name == kRequestFolderName;

        if ( folder_asked && conversation_folder_ )
        {
            return ToolAnswer::Once;
        }

        // A file call in a chat with no folder can reach nothing, since the grant holds none, so the person is not
        // asked about it: the grant refuses it, and the prompt has the model ask for a folder.
        const bool needs_folder = question.needs.reads_files || question.needs.writes_files || question.needs.runs_programs;

        // The workspace's own tools run unasked in a chat that has not read from outside (AssistantInterface.md 5.6). In
        // one that has, find_records is asked with its words shown, and open_place asks only when several places fit,
        // for the person to choose, so their names never reach the model.
        nlohmann::json places = nlohmann::json::array();

        if ( question.name == kFindRecordsName && question.exposure.empty() )
        {
            return ToolAnswer::Once;
        }

        // An offer is never an approval: the person answers it, in words or in the window, after the reply. A move is
        // offered or done and said in a chat that has not read from outside, and asked every time in one that has (5.6).
        if ( question.name == kProposeProjectName || ( question.name == kMoveToProjectName && question.exposure.empty() ) )
        {
            return ToolAnswer::Once;
        }

        // A note runs in every chat, waiting for the person when this one has read from outside; reading the project's
        // note changes nothing (5.6).
        if ( question.name == kNoteUnderstandingName || question.name == kReadProjectNotesName )
        {
            return ToolAnswer::Once;
        }

        if ( question.name == kOpenPlaceName )
        {
            places = question.exposure.empty() ? nlohmann::json::array() : placesFor( std::string( question.arguments ) );

            if ( places.size() < 2 )
            {
                return ToolAnswer::Once;
            }
        }

        if ( needs_folder && !conversation_folder_ )
        {
            return ToolAnswer::Once;
        }

        // What goes out is approved call by call (AssistantInterface.md 5.7), and a folder is chosen each time it is
        // asked for: no answer covers the conversation. Nor does one in a chat that has read from outside, for a
        // call that writes or runs (5.6), as the gate decides.
        const bool exposed_and_acting = !question.exposure.empty() && ( question.needs.writes_files || question.needs.runs_programs );
        const bool every_time = question.needs.uses_network || folder_asked || exposed_and_acting
            || question.name == kFindRecordsName || question.name == kMoveToProjectName || !places.empty();

        // A folder starts read-only, so the first call that changes a file also asks to change files there.
        const bool changes_folder = question.needs.writes_files && conversation_folder_
            && conversation_folder_access_ == Mila::Agents::FolderAccess::Read;

        const std::vector<std::string> choices = every_time
            ? std::vector<std::string>{ "allow_once", "deny" }
            : std::vector<std::string>{ "allow_once", "allow_conversation", "deny" };

        {
            std::lock_guard<std::mutex> lock( approval_mutex_ );

            if ( stop_asked_.load() )
            {
                return ToolAnswer::Deny;
            }

            chosen_folder_.reset();
            chosen_place_.reset();
            approval_.emplace( Approval{ turn_id_, call_id, std::move( answer ), !every_time, folder_asked, places } );
        }

        nlohmann::ordered_json approval = { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
            { "choices", choices }, { "reaches", reaches( question ) },
            { "changes_folder", changes_folder ? conversationFolder() : nlohmann::json( nullptr ) },
            { "has_read", hasRead( question ) },
            { "not_in_your_words", notInPersonsWords( question ) } };

        // A chat in a project of several folders is offered them to choose among (AssistantInterface.md 10).
        if ( folder_asked )
        {
            nlohmann::json offered = nlohmann::json::array();

            if ( const std::optional<nlohmann::json> project = projectOf( conversation_project_ ) )
            {
                offered = project->value( "folders", nlohmann::json::array() );
            }

            approval[ "folders" ] = std::move( offered );
        }

        if ( !places.empty() )
        {
            approval[ "places" ] = places;
        }

        send( "turn.tool_approval", std::move( approval ) );

        const ToolAnswer decided = decision.get();

        // A stop answers the wait with Deny that the person never gave, so the stored call keeps no decision.
        if ( !stop_asked_.load() )
        {
            storedCall( call_id )[ "decision" ] = answerName( decided );
        }

        if ( changes_folder && decided != ToolAnswer::Deny && !stop_asked_.load() )
        {
            allowChangesInFolder();
        }

        return decided;
    }

    void SessionProcess::allowChangesInFolder()
    {
        conversation_folder_access_ = Mila::Agents::FolderAccess::ReadWrite;

        try
        {
            grantChat();
        }
        catch ( const std::invalid_argument& )
        {
            // The folder has gone: the grant is unchanged, so the call fails and says why.
            conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
        }

        sendState();
    }

    nlohmann::json SessionProcess::reaches( const Mila::Agents::ToolQuestion& question )
    {
        const auto shown = []( const std::filesystem::path& path )
            {
                const std::u8string text = path.u8string();

                return std::string( text.begin(), text.end() );
            };

        nlohmann::json folders = nlohmann::json::array();

        for ( const Mila::Agents::GrantFolder& folder : question.grant.folders() )
        {
            folders.push_back( { { "path", shown( folder.folder ) },
                { "access", folder.access == Mila::Agents::FolderAccess::ReadWrite ? "read_write" : "read" } } );
        }

        nlohmann::json program_folders = nlohmann::json::array();

        for ( const std::filesystem::path& folder : question.grant.programFolders() )
        {
            // Windows' long-path prefix, which the person never wrote; a POSIX path never carries it.
            std::filesystem::path::string_type plain = folder.native();
            const std::filesystem::path::string_type long_path_prefix = std::filesystem::path( R"(\\?\)" ).native();

            if ( plain.starts_with( long_path_prefix ) )
            {
                plain.erase( 0, long_path_prefix.size() );
            }

            program_folders.push_back( shown( std::filesystem::path( plain ) ) );
        }

        const bool network = question.grant.network()
            && ( question.name != CommandTool::kName || CommandTool::asksForNetwork( question.arguments ) );

        return { { "folders", std::move( folders ) }, { "program_folders", std::move( program_folders ) },
            { "network", network }, { "runs_programs", question.grant.runsPrograms() } };
    }

    nlohmann::json SessionProcess::hasRead( const Mila::Agents::ToolQuestion& question ) const
    {
        if ( question.exposure.empty() )
        {
            return nullptr;
        }

        nlohmann::json sources = nlohmann::json::array();

        for ( auto source = question.exposure.rbegin(); source != question.exposure.rend() && sources.size() < 3; ++source )
        {
            // A conversation reopened from before the mark was stored knows only the tool.
            sources.push_back( {
                { "name", source->tool },
                { "summary", source->arguments.empty() ? source->tool : callSummary( prompt_tools_, source->tool, source->arguments ) },
                { "time", source->when ? nlohmann::json( std::format( "{:%FT%TZ}",
                    std::chrono::floor<std::chrono::milliseconds>( *source->when ) ) ) : nlohmann::json( nullptr ) } } );
        }

        return { { "count", question.exposure.size() }, { "latest", std::move( sources ) } };
    }

    nlohmann::json SessionProcess::notInPersonsWords( const Mila::Agents::ToolQuestion& question ) const
    {
        if ( question.exposure.empty() )
        {
            return nullptr;
        }

        const std::string_view argument = question.name == WebSearch::kName ? "query"
            : question.name == "read_web_page" ? "url" : std::string_view{};

        if ( argument.empty() )
        {
            return nullptr;
        }

        const nlohmann::json arguments = nlohmann::json::parse( question.arguments, nullptr, false );

        if ( !arguments.is_object() || !arguments.contains( argument ) || !arguments[ argument ].is_string() )
        {
            return nullptr;
        }

        const std::string sent = arguments[ argument ].get<std::string>();

        for ( const nlohmann::json& message : conversation_messages_ )
        {
            if ( message.value( "role", std::string{} ) == "user"
                && message.value( "content", std::string{} ).find( sent ) != std::string::npos )
            {
                return nullptr;
            }
        }

        return sent;
    }

    void SessionProcess::callAnswered( const ToolCall& call, std::string_view result, bool is_error, bool exposes )
    {
        // A call the rules decided was not asked about, so it has not been announced yet.
        const std::string call_id = announced_call_id_.empty() ? announceCall( call.name, call.arguments ) : announced_call_id_;
        announced_call_id_.clear();

        send( "turn.tool_result", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
            { "result", result }, { "is_error", is_error } } );

        nlohmann::json& stored = storedCall( call_id );
        stored[ "result" ] = result;
        stored[ "is_error" ] = is_error;

        // Where the result came from, so the stored call says what exposed the chat (AssistantInterface.md 5.6).
        stored[ "exposes" ] = exposes;

        // What a write made, read straight after it, so the drawer can say later whether the file changed and who
        // made it in which step (AssistantInterface.md 4.9).
        if ( const std::optional<std::string> path = writtenPath( stored ) )
        {
            const std::filesystem::path named( std::u8string( path->begin(), path->end() ) );
            const std::filesystem::path file = conversation_folder_
                ? std::filesystem::path( std::u8string( conversation_folder_->begin(), conversation_folder_->end() ) ) / named
                : named;

            stored[ "written" ] = fileRecord( file );
            stored[ "model" ] = model_.name();
            stored[ "step" ] = plan_.currentStep();
        }

        // The call is over and the gate holds none of its tools now, so a change it asked for can replace them.
        if ( std::exchange( tools_changed_in_call_, false ) )
        {
            offerTools();
        }
    }

    void SessionProcess::offerToolsWhenSafe()
    {
        // A call running is one of the gate's tools: replacing them under it would destroy the function being run.
        if ( work_ == Work::Turn )
        {
            tools_changed_in_call_ = true;

            return;
        }

        offerTools();
    }

    std::string SessionProcess::announceCall( std::string_view name, std::string_view arguments )
    {
        announced_call_id_ = std::format( "call-{}", ++call_number_ );
        const std::string summary = callSummary( prompt_tools_, name, arguments );

        send( "turn.tool_call", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "round", round_ },
            { "call_id", announced_call_id_ }, { "name", name }, { "summary", summary },
            { "arguments", arguments } } );

        // Stored as announced, so a call the person stopped before it ran is kept with no result (6.3).
        turn_calls_.push_back( { { "call_id", announced_call_id_ }, { "name", name }, { "summary", summary },
            { "arguments", arguments }, { "time", timestamp() }, { "decision", nullptr }, { "result", nullptr },
            { "is_error", false } } );

        return announced_call_id_;
    }

    nlohmann::json& SessionProcess::storedCall( const std::string& call_id )
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

    std::string_view SessionProcess::turnFinish() const
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

    void SessionProcess::recordTurn()
    {
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
            { "stats", stats },
            { "model", model_.name() } } );

        saveConversation();
    }

    void SessionProcess::saveConversation( const Turn* sent )
    {
        nlohmann::json turns = nlohmann::json::array();

        for ( const Turn& turn : session().conversation() )
        {
            if ( turn.role != Role::System )
            {
                turns.push_back( turnToJson( turn ) );
            }
        }

        if ( sent != nullptr )
        {
            turns.push_back( turnToJson( *sent ) );
        }

        const nlohmann::json conversation = {
            { "v", 1 },
            { "id", conversation_id_ },
            { "title", conversation_title_ },
            { "model", model_.name() },
            { "folder", conversationFolder() },
            { "folder_from_project", conversation_folder_ && conversation_folder_from_project_ },
            { "project", conversation_project_.empty() ? nlohmann::json( nullptr ) : nlohmann::json( conversation_project_ ) },
            { "plan", plan_.toJson() },
            { "created", conversation_created_ },
            { "updated", timestamp() },
            { "messages", conversation_messages_ },
            { "links_opened", conversation_links_ },
            { "exposure", exposureToJson( session().exposure() ) },
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

    void SessionProcess::startNewConversation()
    {
        // Ends the tools allowed for the conversation with it.
        session().newConversation();
        forgetConversation();
    }

    void SessionProcess::forgetConversation()
    {
        conversation_id_.clear();
        conversation_title_.clear();
        conversation_created_.clear();
        conversation_project_.clear();
        conversation_messages_ = nlohmann::json::array();
        conversation_links_ = nlohmann::json::array();
    }

    void SessionProcess::recordLinkOpened( const std::string& url )
    {
        if ( url.empty() || conversation_id_.empty() )
        {
            return;
        }

        conversation_links_.push_back( { { "url", url }, { "time", timestamp() } } );
        saveConversation();
        sendConversations();
    }

    void SessionProcess::removeReach( const nlohmann::json& request )
    {
        const std::string kind = request.value( "kind", std::string{} );

        if ( kind == "allowed" )
        {
            session().gate().withdrawAllowance( request.value( "name", std::string{} ) );
            sendState();

            return;
        }

        if ( kind != "folder" )
        {
            return;
        }

        const std::string id = request.value( "id", std::string{} );

        if ( id.empty() || id == conversation_id_ )
        {
            conversation_folder_.reset();
            conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
            refreshTools();

            if ( !conversation_id_.empty() )
            {
                saveConversation();
            }
        }
        else if ( std::optional<nlohmann::json> stored = store_.load( id ) )
        {
            ( *stored )[ "folder" ] = nullptr;
            store_.save( *stored );
        }

        sendState();
        sendConversations();
    }

    void SessionProcess::openConversation( const std::string& id )
    {
        const std::optional<nlohmann::json> conversation = store_.load( id );

        if ( !conversation.has_value() )
        {
            sendError( "That conversation is no longer stored.", nullptr );

            return;
        }

        // A conversation stored with an activity opens as a chat like any other, its folder in its grant
        // (AssistantInterface.md 5.5); the activity is not read.
        const nlohmann::json stored_folder = conversation->value( "folder", nlohmann::json( nullptr ) );
        std::string folder = stored_folder.is_string() ? stored_folder.get<std::string>() : std::string{};

        // A chat in a project opens with the project's folder as it stands now (AssistantInterface.md 10): one the
        // project let go is gone from it, one with none is given the project's one folder, and a project's folder is
        // held at the project's access. Any other folder opens read-only, as a chat reopened does.
        const nlohmann::json stored_project = conversation->value( "project", nlohmann::json( nullptr ) );
        const std::optional<nlohmann::json> project = projectOf( stored_project.is_string() ? stored_project.get<std::string>() : std::string{} );
        const bool stored_from_project = conversation->value( "folder_from_project", false );

        Mila::Agents::FolderAccess access = Mila::Agents::FolderAccess::Read;
        bool from_project = false;

        if ( project.has_value() )
        {
            const std::optional<Mila::Agents::FolderAccess> held = projectAccess( *project, folder );

            if ( !folder.empty() && held.has_value() )
            {
                access = *held;
                from_project = true;
            }
            else if ( !folder.empty() && stored_from_project )
            {
                folder.clear();
            }

            if ( folder.empty() )
            {
                if ( const auto starting = startingFolder( *project ) )
                {
                    folder = starting->first;
                    access = starting->second;
                    from_project = true;
                }
            }
        }
        else if ( stored_from_project )
        {
            // Its project has gone; a chat it held a folder for keeps the folder as its own, read-only.
            from_project = false;
        }

        useChat( folder, access, from_project );

        plan_.restore( conversation->value( "plan", nlohmann::json( nullptr ) ) );

        std::vector<Turn> turns;

        for ( const nlohmann::json& turn : conversation->value( "turns", nlohmann::json::array() ) )
        {
            turns.push_back( turnFromJson( turn ) );
        }

        // No tool is allowed for it yet: what the person allowed when it was written is asked about again. A
        // conversation stored before its exposure was counts every tool result in it as read from outside.
        if ( conversation->contains( "exposure" ) )
        {
            session().openConversation( std::move( turns ), exposureFromJson( ( *conversation )[ "exposure" ] ) );
        }
        else
        {
            session().openConversation( std::move( turns ) );
        }

        conversation_id_ = id;
        conversation_title_ = conversation->value( "title", std::string{} );
        conversation_created_ = conversation->value( "created", std::string{} );
        conversation_project_ = project.has_value() ? project->value( "id", std::string{} ) : std::string{};
        conversation_messages_ = conversation->value( "messages", nlohmann::json::array() );
        conversation_links_ = conversation->value( "links_opened", nlohmann::json::array() );
        offerTools();

        // A chat's call ids are its own, since an offer is answered by its call's id: numbering goes on past the highest
        // this chat holds, which an earlier run of the session may have given.
        for ( const nlohmann::json& message : conversation_messages_ )
        {
            const nlohmann::json calls = message.is_object() ? message.value( "tool_calls", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

            for ( const nlohmann::json& call : calls.is_array() ? calls : nlohmann::json::array() )
            {
                const std::string id = call.is_object() ? call.value( "call_id", std::string{} ) : std::string{};
                int number = 0;

                if ( id.starts_with( "call-" ) && std::from_chars( id.data() + 5, id.data() + id.size(), number ).ec == std::errc{} )
                {
                    call_number_ = std::max( call_number_, number );
                }
            }
        }

        send( "conversation.loaded", { { "id", id }, { "title", conversation_title_ },
            { "model", conversation->value( "model", std::string{} ) },
            { "folder", conversationFolder() },
            { "project", conversation_project_.empty() ? nlohmann::json( nullptr ) : nlohmann::json( conversation_project_ ) },
            { "messages", conversation_messages_ } } );

        sendState();
    }

    void SessionProcess::renameConversation( const std::string& id, std::string title )
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

    void SessionProcess::deleteConversation( const std::string& id )
    {
        if ( !store_.remove( id ) )
        {
            return;
        }

        forgetEverywhere( id );
        clearNotesFrom( id );

        if ( id == conversation_id_ )
        {
            startNewConversation();
            sendState();
        }

        sendProjects();
        sendConversations();
        sendActivity( {} );
    }

    bool SessionProcess::conversationHasUserTurn() const
    {
        return std::ranges::any_of( session().conversation(), []( const Turn& turn ) { return turn.role == Role::User; } );
    }

    void SessionProcess::useChat( const std::string& folder, Mila::Agents::FolderAccess access, bool from_project )
    {
        conversation_folder_.reset();
        conversation_folder_access_ = access;
        conversation_folder_from_project_ = from_project && !folder.empty();

        if ( !folder.empty() && !folders_enforced_ )
        {
            send( "info", { { "text", std::format(
                "This chat was started in the folder {}, and Mila cannot work in folders on this computer yet.", folder ) } } );
        }
        else if ( !folder.empty() )
        {
            conversation_folder_ = folder;
        }

        try
        {
            grantChat();
        }
        catch ( const std::invalid_argument& )
        {
            send( "info", { { "text", std::format( "The folder {} is not there any more, so this chat has none.", folder ) } } );
            conversation_folder_.reset();
            conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
            conversation_folder_from_project_ = false;
            grantChat();
        }

        usePrompt( prompts_ / "general.json" );
        session().withLimits( Mila::Agents::AgentLimits{} );

        Mila::Agents::ToolRules rules;
        rules.add( { std::string( TaskPlan::kUpdateName ), Mila::Agents::ToolPermission::Allow, {} } );
        session().withRules( std::move( rules ) );

        plan_.clear();
    }

    void SessionProcess::grantChat()
    {
        std::vector<Mila::Agents::GrantFolder> folders;

        if ( conversation_folder_ )
        {
            folders.push_back( { std::filesystem::path( std::u8string( conversation_folder_->begin(), conversation_folder_->end() ) ),
                conversation_folder_access_ } );
        }

        if ( container_ && !folders.empty() && conversation_folder_access_ == Mila::Agents::FolderAccess::ReadWrite )
        {
            const std::filesystem::path project = folders.front().folder;
            Mila::Agents::ToolGrant grant( std::move( folders ), true, ProgramSetup::forProject( programs_, project ), container_ );
            ProgramSetup::writeGitIdentity( grant.folders().front().folder );
            session().withGrant( std::move( grant ) );

            return;
        }

        session().withGrant( Mila::Agents::ToolGrant( std::move( folders ), true ) );
    }

    std::string SessionProcess::grantChosenFolder()
    {
        if ( conversation_folder_ )
        {
            return std::format( "This chat already works in the folder {}; paths are relative to it.", *conversation_folder_ );
        }

        std::optional<std::string> chosen;

        {
            std::lock_guard<std::mutex> lock( approval_mutex_ );
            chosen = std::exchange( chosen_folder_, std::nullopt );
        }

        if ( !chosen )
        {
            throw std::runtime_error( "no folder was chosen" );
        }

        // One of the project's folders is held at the project's access; any other is this chat's alone, read first.
        const std::optional<nlohmann::json> project = projectOf( conversation_project_ );
        const std::optional<Mila::Agents::FolderAccess> held = project.has_value() ? projectAccess( *project, *chosen ) : std::nullopt;

        conversation_folder_ = *chosen;
        conversation_folder_access_ = held.value_or( Mila::Agents::FolderAccess::Read );
        conversation_folder_from_project_ = held.has_value();

        try
        {
            grantChat();
        }
        catch ( const std::invalid_argument& refused )
        {
            conversation_folder_.reset();
            conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
            conversation_folder_from_project_ = false;

            throw std::runtime_error( std::format( "the folder {} cannot be used: {}", *chosen, refused.what() ) );
        }

        // The file tools are offered from the next round, now that there is a folder for them to work in (5.5).
        offerToolsWhenSafe();
        sendState();

        if ( conversation_folder_access_ == Mila::Agents::FolderAccess::ReadWrite )
        {
            return std::format( "The user gave you the folder {} to read and change. Paths are relative to it, as "
                "list_folder shows them.", *chosen );
        }

        return std::format( "The user gave you the folder {} to read. Paths are relative to it, as list_folder shows "
            "them. The user is asked the first time you change a file in it.", *chosen );
    }

    nlohmann::json SessionProcess::placesFor( const std::string& arguments ) const
    {
        const nlohmann::json parsed = nlohmann::json::parse( arguments, nullptr, false );
        const std::string words = parsed.is_object() ? parsed.value( "place", std::string{} ) : std::string{};
        const std::vector<nlohmann::json> projects = projects_.list();

        return findPlaces( words, projects, store_.list(), conversation_id_ );
    }

    std::string SessionProcess::openPlace( const std::string& arguments )
    {
        std::optional<nlohmann::json> chosen;

        {
            std::lock_guard<std::mutex> lock( approval_mutex_ );
            chosen = std::exchange( chosen_place_, std::nullopt );
        }

        const bool names_allowed = session().exposure().empty();
        nlohmann::json place;

        if ( chosen )
        {
            place = std::move( *chosen );
        }
        else
        {
            const nlohmann::json places = placesFor( arguments );

            if ( places.empty() )
            {
                return "Nothing in the workspace is called that, and nothing was opened. The places are workspace, "
                    "projects, chats, activity and reach, or a project or chat by its name.";
            }

            if ( places.size() > 1 )
            {
                // In a chat that has read from outside the person was asked; one that has not asks them through the model.
                return names_allowed ? severalPlaces( places )
                    : std::format( "{} places fit, none was opened, and the user was not shown them.", places.size() );
            }

            place = places.front();
        }

        const std::string line = wentLine( place );
        const nlohmann::json where = { { "view", place[ "view" ] }, { "id", place[ "id" ] } };

        send( "turn.went", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", announced_call_id_ },
            { "place", where }, { "text", line } } );

        // Kept with the call, so the chat shows where Mila went when it is opened again.
        if ( !announced_call_id_.empty() )
        {
            storedCall( announced_call_id_ )[ "went" ] = { { "place", where }, { "text", line } };
        }

        const std::vector<nlohmann::json> projects = projects_.list();

        return describePlace( place, projects, store_.list(), conversation_id_, names_allowed );
    }

    Mila::Agents::ToolResult SessionProcess::findRecordsIn( const std::string& arguments ) const
    {
        const nlohmann::json parsed = nlohmann::json::parse( arguments, nullptr, false );
        const std::string words = parsed.is_object() ? parsed.value( "words", std::string{} ) : std::string{};
        const std::vector<nlohmann::json> conversations = store_.loadAll();
        const std::vector<nlohmann::json> projects = projects_.list();

        return findRecords( words, conversations, projects, conversation_id_ );
    }

    void SessionProcess::usePrompt( const std::filesystem::path& path )
    {
        prompt_ = PromptFile::load( path );
        prompt_tools_ = prompt_.tools;

        for ( PromptTool& named : prompt_tools_ )
        {
            if ( !named.description.empty() )
            {
                continue;
            }

            const auto tool = std::ranges::find( milo_tools_, named.name, &Mila::Agents::Tool::name );

            if ( tool == milo_tools_.end() )
            {
                const std::u8string file = path.u8string();

                throw std::runtime_error( std::format( "the prompt {} names a tool Mila does not have: {}",
                    std::string( file.begin(), file.end() ), named.name ) );
            }

            named.description = tool->description;
            named.parameters = tool->parameters;
            named.summary = tool->summary;
        }

        offerTools();
    }

    void SessionProcess::refreshTools()
    {
        grantChat();
        offerTools();
    }

    void SessionProcess::registerTool( std::string name, Mila::Agents::ToolNeeds needs, Handler handler,
        Mila::Agents::ResultOrigin origin )
    {
        handlers_.insert_or_assign( std::move( name ), RegisteredTool{ needs, std::move( handler ), origin } );
    }

    void SessionProcess::useMiloTool( Mila::Agents::Tool tool )
    {
        registerTool( tool.name, tool.needs,
            [run = tool.run]( const std::string& arguments, const Mila::Agents::ToolGrant& grant, std::stop_token stop )
            {
                return run( arguments, grant, stop );
            }, tool.result_origin );

        milo_tools_.push_back( std::move( tool ) );
    }

    void SessionProcess::offerTools()
    {
        Mila::Agents::Session& offering = session();
        offering.withoutTools();

        std::set<std::string, std::less<>> offered;

        for ( const PromptTool& described : prompt_tools_ )
        {
            const auto handler = handlers_.find( described.name );

            if ( handler == handlers_.end() )
            {
                continue;
            }

            const Mila::Agents::ToolNeeds& needs = handler->second.needs;
            const bool works_in_folder = needs.reads_files || needs.writes_files || needs.runs_programs;
            const bool needs_folder = works_in_folder || described.name == kRequestFolderName;

            if ( needs_folder && !folders_enforced_ )
            {
                continue;
            }

            // A tool that works in a folder can do nothing in a chat that holds none, so it is offered once the chat has
            // one; until then the chat has request_folder (AssistantInterface.md 5.5).
            if ( works_in_folder && !conversation_folder_ )
            {
                continue;
            }

            // A project's note is a project's, so a chat in none is offered neither (phase 7's decisions).
            if ( ( described.name == kNoteUnderstandingName || described.name == kReadProjectNotesName ) && conversation_project_.empty() )
            {
                continue;
            }

            offered.insert( described.name );

            Mila::Agents::Tool tool{
                described.name,
                described.description,
                described.parameters,
                [run = handler->second.run]( std::string_view arguments, const Mila::Agents::ToolGrant& grant, std::stop_token stop )
                {
                    return run( std::string( arguments ), grant, stop );
                } };

            tool.needs = handler->second.needs;
            tool.result_origin = handler->second.origin;
            offering.withTool( std::move( tool ) );
        }

        offering.withSystemPrompt( systemPrompt( offered ) );
        applyTemplateVariables();
    }

    std::vector<ToolCall> SessionProcess::placedCalls() const
    {
        if ( conversation_project_.empty() || conversationHasUserTurn()
            || session().gate().find( kReadProjectNotesName ) == nullptr )
        {
            return {};
        }

        if ( !projects_.understanding( conversation_project_ ).value( "current", nlohmann::json( nullptr ) ).is_object() )
        {
            return {};
        }

        return { ToolCall{ std::string( kReadProjectNotesName ), "{}" } };
    }

    Mila::Agents::ToolResult SessionProcess::readProjectNotes()
    {
        const std::optional<nlohmann::json> project = projectOf( conversation_project_ );

        if ( !project.has_value() )
        {
            return Mila::Agents::ToolResult( "This chat is in no project, so there are no notes to read.",
                Mila::Agents::ResultOrigin::Application );
        }

        const std::string name = project->value( "name", std::string{} );
        const nlohmann::json current = projects_.understanding( conversation_project_ ).value( "current", nlohmann::json( nullptr ) );

        // The app's call has no question before it, so it is announced here, as the model's calls are when asked.
        if ( announced_call_id_.empty() )
        {
            announceCall( kReadProjectNotesName, "{}" );
        }

        nlohmann::json& stored = storedCall( announced_call_id_ );

        if ( std::exchange( placing_notes_, false ) )
        {
            stored[ "placed" ] = true;
        }

        if ( !current.is_object() )
        {
            stored[ "given" ] = nullptr;

            return Mila::Agents::ToolResult( std::format( "The project \"{}\" has no notes yet.", name ),
                Mila::Agents::ResultOrigin::Application );
        }

        // What a note written here rests on, so deleting any chat it was drawn from clears that note too (8).
        stored[ "given" ] = { { "id", current.value( "id", std::string{} ) },
            { "derived_from", current.value( "derived_from", nlohmann::json::array() ) } };

        const nlohmann::json source = current.value( "source", nlohmann::json::object() );
        const nlohmann::json reference = source.is_object() ? source.value( "ref", nlohmann::json::object() ) : nlohmann::json::object();
        const std::string from = reference.is_object() ? reference.value( "conversation", std::string{} ) : std::string{};
        const std::optional<nlohmann::json> written_in = from.empty() ? std::nullopt : store_.load( from );

        // A note drawn from a chat that had read from outside may not direct Mila, accepted or not (5.6).
        return Mila::Agents::ToolResult(
            notesText( current, name, written_in.has_value() ? written_in->value( "title", std::string{} ) : std::string{} ),
            current.value( "exposed", true ) ? Mila::Agents::ResultOrigin::Outside : Mila::Agents::ResultOrigin::Application );
    }

    std::string SessionProcess::noteUnderstanding( const std::string& arguments )
    {
        const std::optional<nlohmann::json> project = projectOf( conversation_project_ );

        if ( !project.has_value() )
        {
            return "This chat is in no project, so there is nothing to note.";
        }

        // A note replaces the project's whole, so a chat that never read the current one would drop what it holds: it
        // reads it first (phase 7's decisions, Todd, 2026-11-21).
        const nlohmann::json current = projects_.understanding( conversation_project_ ).value( "current", nlohmann::json( nullptr ) );

        if ( current.is_object() && !noteGivenIn( conversation_messages_, turn_calls_, current.value( "id", std::string{} ) ) )
        {
            return "Not noted: this chat has not read the project's current notes. Call read_project_notes, then write the "
                "whole note again, keeping what still holds.";
        }

        const std::string call_id = announced_call_id_;
        const nlohmann::json written = noteFromArguments( arguments, newStoreIdentifier( "note" ), conversation_id_, call_id,
            givenDerivedFrom( conversation_messages_, turn_calls_ ), !session().exposure().empty(), model_.name(), timestamp() );

        if ( !written.contains( "note" ) )
        {
            return written[ "result" ].get<std::string>();
        }

        const std::string note_id = written[ "note" ][ "id" ].get<std::string>();
        nlohmann::json understanding = projects_.understanding( conversation_project_ );
        const std::string where = admitNote( understanding, written[ "note" ] );

        try
        {
            projects_.saveUnderstanding( conversation_project_, understanding );
        }
        catch ( const std::exception& failure )
        {
            return std::format( "Not noted: the note could not be kept ({}).", failure.what() );
        }

        recordEvent( { { "kind", where == "waiting" ? "understanding.waiting" : "understanding.noted" }, { "who", "mila" },
            { "project", conversation_project_ }, { "conversation", conversation_id_ }, { "name", project->value( "name", std::string{} ) } } );

        if ( !call_id.empty() )
        {
            storedCall( call_id )[ "note" ] = { { "id", note_id }, { "state", where } };
        }

        sendProjects();

        if ( where == "waiting" )
        {
            return std::format( "Noted. This chat has read from outside, so the note waits for the user to accept it on the "
                "project {} before new chats there are given it.", project->value( "name", std::string{} ) );
        }

        return std::format( "Noted for the project {}; new chats in it are given this note.", project->value( "name", std::string{} ) );
    }

    void SessionProcess::decideNote( const nlohmann::json& request )
    {
        const std::string id = request.value( "project", std::string{} );
        const std::optional<nlohmann::json> project = projectOf( id );

        if ( !project.has_value() )
        {
            sendError( "That project is no longer stored.", nullptr );

            return;
        }

        const bool accept = request.value( "decision", std::string{} ) == "accept";
        nlohmann::json understanding = projects_.understanding( id );

        if ( !decideWaiting( understanding, request.value( "id", std::string{} ), accept, timestamp() ) )
        {
            sendError( "That note is no longer waiting.", nullptr );
            sendProjects();

            return;
        }

        projects_.saveUnderstanding( id, understanding );
        recordEvent( { { "kind", accept ? "understanding.accepted" : "understanding.declined" }, { "project", id },
            { "name", project->value( "name", std::string{} ) } } );

        sendProjects();
        sendActivity( {} );
    }

    void SessionProcess::changeUnderstanding( const std::string& project_id, const nlohmann::json& value )
    {
        const std::optional<nlohmann::json> project = projectOf( project_id );
        nlohmann::json understanding = projects_.understanding( project_id );

        if ( value.is_null() )
        {
            if ( !understanding[ "current" ].is_object() )
            {
                return;
            }

            understanding[ "current" ] = nullptr;
            projects_.saveUnderstanding( project_id, understanding );
            recordEvent( { { "kind", "understanding.cleared" }, { "project", project_id },
                { "name", project->value( "name", std::string{} ) } } );

            return;
        }

        std::variant<nlohmann::json, std::string> edited = editedNote( understanding[ "current" ], value, newStoreIdentifier( "note" ), timestamp() );

        if ( std::holds_alternative<std::string>( edited ) )
        {
            sendError( std::get<std::string>( edited ), nullptr );

            return;
        }

        understanding[ "current" ] = std::move( std::get<nlohmann::json>( edited ) );
        projects_.saveUnderstanding( project_id, understanding );
        recordEvent( { { "kind", "understanding.edited" }, { "project", project_id },
            { "name", project->value( "name", std::string{} ) } } );
    }

    void SessionProcess::clearNotesFrom( const std::string& conversation_id )
    {
        for ( const nlohmann::json& project : projects_.list() )
        {
            const std::string id = project.value( "id", std::string{} );
            nlohmann::json understanding = projects_.understanding( id );

            if ( clearDerivedFrom( understanding, conversation_id ).empty() )
            {
                continue;
            }

            try
            {
                projects_.saveUnderstanding( id, understanding );
            }
            catch ( const std::exception& failure )
            {
                sendError( std::format( "A note drawn from the deleted chat could not be cleared: {}", failure.what() ), nullptr );

                continue;
            }

            // The person deleted what it was drawn from, so the clearing is theirs.
            recordEvent( { { "kind", "understanding.cleared" }, { "project", id }, { "name", project.value( "name", std::string{} ) },
                { "because", "chat.deleted" } } );
        }
    }

    std::string SessionProcess::systemPrompt( const std::set<std::string, std::less<>>& offered ) const
    {
        std::string prompt = prompt_.systemPromptFor( offered );
        const std::optional<nlohmann::json> project = projectOf( conversation_project_ );

        if ( !project.has_value() )
        {
            return prompt;
        }

        // Only the person's own words go here, and last: the project's name and purpose, which they wrote or accepted, and
        // their instructions. Mila's notes reach a chat as a result, never as the prompt (5.6).
        const std::string purpose = trimmed( project->value( "purpose", std::string{} ) );
        const std::string instructions = trimmed( project->value( "instructions", std::string{} ) );

        prompt += std::format( "\n\nThis chat is in the user's project \"{}\"", project->value( "name", std::string{} ) );
        prompt += purpose.empty() ? std::string( "." ) : std::format( ": {}", purpose );

        if ( !purpose.empty() && std::string_view( ".!?" ).find( purpose.back() ) == std::string_view::npos )
        {
            prompt += '.';
        }

        if ( !instructions.empty() )
        {
            prompt += std::format( " The user's instructions for it:\n{}", instructions );
        }

        return prompt;
    }

    DeploymentRequest SessionProcess::requestFor( std::optional<WeightQuantization> quantization ) const
    {
        DeploymentRequest request;
        request.withAllowedDevices( devices_ )
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

    void SessionProcess::openRemembered()
    {
        const DeploymentRequest request = requestFor( std::nullopt );

        if ( auto refusal = Mila::Agents::Agent::check( settings_.model, request ) )
        {
            no_model_reason_ = refusalText( *refusal );

            return;
        }

        work_ = Work::Load;
        open( settings_.model, std::nullopt, request );
        work_ = Work::None;
    }

    void SessionProcess::loadModel( const nlohmann::json& request )
    {
        if ( !licenseAccepted() )
        {
            sendError( std::string( kLicenseFirst ), "load" );

            return;
        }

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

    void SessionProcess::switchModel( const std::string& name, std::optional<WeightQuantization> quantization )
    {
        const bool had_model = model_.resident();
        const std::string previous = model_.name();
        const std::optional<WeightQuantization> previous_quantization = loaded_quantization_;
        const DeploymentRequest previous_request = had_model ? model_.request() : DeploymentRequest{};

        const DeploymentRequest request = requestFor( quantization );

        if ( auto refusal = Mila::Agents::Agent::check( name, request ) )
        {
            sendError( refusalText( *refusal ), workName() );

            return;
        }

        if ( !open( name, quantization, request ) )
        {
            if ( had_model && open( previous, previous_quantization, previous_request ) )
            {
                send( "info", { { "text", std::format( "{} is loaded again.", previous ) }, { "during", workName() } } );
            }

            return;
        }

        startNewConversation();

        // Said during the load or install, so a conversation that has not begun keeps its home screen.
        send( "info", { { "text", had_model ? "Model switched. Conversation history cleared." : "Model loaded." },
            { "during", workName() } } );

        if ( !settings_.context_length )
        {
            send( "info", { { "text", std::format( "Context {} (auto).", model_.agent().plan().contextLength() ) },
                { "during", workName() } } );
        }
    }

    bool SessionProcess::open( const std::string& name, std::optional<WeightQuantization> quantization, const DeploymentRequest& request )
    {
        const bool applied_at_load = quantization && *quantization != WeightQuantization::None;

        // The quantization is quoted only when it is applied as the weights load: a pre-quantized model already
        // carries it in its name. The model is named as a person reads it.
        const std::string shown = displayName( name );
        load_label_ = applied_at_load
            ? std::format( "Loading {} ({})", shown, quantizationWord( *quantization ) )
            : std::format( "Loading {}", shown );

        loading_ = name;
        sendState();
        send( "progress", { { "task", loadTask() }, { "label", load_label_ }, { "done", nullptr }, { "total", nullptr } } );

        std::string failure;

        try
        {
            model_.open( name, request );
        }
        catch ( const Mila::Deployment::OpenRefusedError& error )
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

        loading_.clear();
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

        // A model just loaded is about to be used: held from now, so the first prompt starts at once.
        memory_holder_.use( model_.agent().model().model().getDeviceId() );
        memory_holder_.idle();

        return true;
    }

    void SessionProcess::installModel( const std::string& name )
    {
        if ( !licenseAccepted() )
        {
            sendError( std::string( kLicenseFirst ), "install" );

            return;
        }

        const auto published = published_display_names_.find( name );
        install_name_ = published != published_display_names_.end() ? published->second : name;
        install_total_ = 0;
        install_percent_ = -1;

        work_ = Work::Install;
        sendState();

        bool installed = false;

        try
        {
            for ( const std::string& line : Mila::Desktop::installModel( name,
                [this]( std::uint64_t received, std::uint64_t total ) { installProgress( received, total ); } ) )
            {
                send( "info", { { "text", line }, { "during", "install" } } );
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

    void SessionProcess::installProgress( std::uint64_t received, std::uint64_t total )
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

    void SessionProcess::removeModel( const std::string& name )
    {
        try
        {
            for ( const std::string& line : Mila::Desktop::removeModel( name ) )
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

    void SessionProcess::reopen( const DeploymentRequest& request, std::string_view done )
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

    void SessionProcess::changeSetting( const nlohmann::json& request )
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
        else if ( request.contains( "run_commands" ) )
        {
            changeRunCommands( request[ "run_commands" ] );
        }
        else
        {
            sendError( "That is not a setting this session has.", "settings" );
        }
    }

    void SessionProcess::changeContext( const nlohmann::json& value )
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

        Mila::Agents::Agent& agent = model_.agent();

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

    void SessionProcess::changeThinking( const nlohmann::json& value )
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

    void SessionProcess::changeEffort( const nlohmann::json& value )
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

    void SessionProcess::changeSampling( const nlohmann::json& change )
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

        Mila::Agents::SamplingOverride& sampling = settings_.sampling;

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

    std::string_view SessionProcess::samplingBounds( std::string_view key ) noexcept
    {
        if ( key == "temperature" )
            return "Temperature is from 0 to 5; 0 always picks the likeliest word.";

        if ( key == "top_k" )
            return "top_k is a whole number, 0 or more; 0 turns it off.";

        if ( key == "top_p" )
            return "top_p is from 0 to 1; 1 turns it off.";

        return "presence_penalty is from -2 to 2; 0 turns it off.";
    }

    void SessionProcess::changeSeed( const nlohmann::json& value )
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

    void SessionProcess::changeSearchKey( const nlohmann::json& value )
    {
        const std::string key = value.is_string() ? value.get<std::string>() : std::string{};
        const auto first = key.find_first_not_of( " \t\r\n" );
        const auto last = key.find_last_not_of( " \t\r\n" );
        const std::string trimmed = first == std::string::npos ? std::string{} : key.substr( first, last - first + 1 );

        if ( trimmed.empty() )
        {
            search_key_store_.remove();
            search_key_file_.remove();
            web_search_.reset();
            handlers_.erase( std::string( WebSearch::kName ) );
            refreshTools();

            return;
        }

        search_key_store_.write( trimmed );
        search_key_file_.remove();
        useSearchKey( trimmed );
        refreshTools();
    }

    void SessionProcess::useSearchKey( const std::string& key )
    {
        web_search_.emplace( key );

        Mila::Agents::ToolNeeds network;
        network.uses_network = true;

        // Registered once; a later key replaces the search it runs.
        registerTool( std::string( WebSearch::kName ), network,
            [this]( const std::string& arguments, const Mila::Agents::ToolGrant& grant, std::stop_token )
            {
                if ( !web_search_ )
                {
                    throw std::runtime_error( "web search is not set up" );
                }

                return web_search_->search( arguments, grant );
            } );
    }

    void SessionProcess::changeRunCommands( const nlohmann::json& value )
    {
        if ( !value.is_boolean() )
        {
            sendError( "Running commands is on or off.", "settings" );

            return;
        }

        settings_.run_commands = value.get<bool>();
        saveSettings();

        if ( !settings_.run_commands )
        {
            // Off at once, in the open conversation too: no prompt names the tool and nothing runs it.
            container_.reset();
            container_problem_.clear();
            handlers_.erase( std::string( CommandTool::kName ) );

            refreshTools();

            return;
        }

        useCommands();

        if ( container_ )
        {
            refreshTools();
            send( "info", { { "text", "Mila can run commands in a chat's folder now." } } );
        } else {
            sendError( std::format( "Commands cannot be run here: {}", container_problem_ ), "settings" );
        }
    }

    void SessionProcess::useCommands()
    {
        auto opened = Mila::Agents::MxcProgramContainer::open( prompts_.parent_path() / "mxc_ffi.dll" );

        if ( !opened )
        {
            container_.reset();
            container_problem_ = opened.error();

            return;
        }

        container_ = std::move( *opened );
        container_problem_.clear();
        programs_ = ProgramSetup::find();

        registerTool( std::string( CommandTool::kName ), CommandTool::needs(),
            []( const std::string& arguments, const Mila::Agents::ToolGrant& grant, std::stop_token stop )
            {
                return CommandTool::run( arguments, grant, stop );
            } );
    }

    void SessionProcess::useDetail( const std::string& detail )
    {
        Mila::Logging::Logger::defaultLogger().setLevel( detail == "all"
            ? Mila::Logging::LogLevel::Info
            : Mila::Logging::LogLevel::Warning );
    }

    bool SessionProcess::licenseAccepted() const
    {
        return license_.has_value() && !settings_.license_accepted.empty()
            && settings_.license_accepted == license_->sha256;
    }

    void SessionProcess::sendLicense()
    {
        if ( !license_.has_value() )
        {
            send( "license", { { "text", nullptr }, { "sha256", nullptr }, { "accepted", false },
                { "problem", license_.error() } } );

            return;
        }

        send( "license", { { "text", license_->text }, { "sha256", license_->sha256 },
            { "accepted", licenseAccepted() }, { "problem", nullptr } } );
    }

    void SessionProcess::acceptLicense( const std::string& sha256 )
    {
        if ( !license_.has_value() )
        {
            sendError( license_.error(), "license" );

            return;
        }

        if ( sha256 != license_->sha256 )
        {
            sendError( "The licence shown is not the one Mila carries. Close Mila and open it again.", "license" );
            sendLicense();

            return;
        }

        if ( !licenseAccepted() )
        {
            const auto now = std::chrono::floor<std::chrono::seconds>( std::chrono::system_clock::now() );
            settings_.license_accepted = license_->sha256;
            settings_.license_accepted_at = std::format( "{:%FT%TZ}", now );
            saveSettings();
        }

        sendLicense();
        sendState();

        if ( !settings_.model.empty() && !model_.resident() )
        {
            openRemembered();
            sendState();
        }
    }

    void SessionProcess::saveSettings()
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

    const Mila::Dnn::Conversation::TemplateVariable* SessionProcess::templateVariable( std::string_view name ) const
    {
        if ( !model_.resident() )
        {
            return nullptr;
        }

        const auto variables = model_.agent().model().protocol()->templateVariables();
        const auto found = std::ranges::find( variables, name, &Mila::Dnn::Conversation::TemplateVariable::name );

        return found == variables.end() ? nullptr : &*found;
    }

    bool SessionProcess::hasThinkingSwitch() const
    {
        return model_.resident() && model_.agent().model().capabilities().thinking
            && templateVariable( "enable_thinking" ) != nullptr;
    }

    const Mila::Dnn::Conversation::TemplateVariable* SessionProcess::effortVariable() const
    {
        return hasThinkingSwitch() ? templateVariable( "reasoning_effort" ) : nullptr;
    }

    void SessionProcess::applyTemplateVariables()
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

        // Thinking off: Qwen 3.8 at its default effort ran out of 8192 tokens writing a summary, in 4 of 6
        // (Notebooks/CompactionSummaries.md). 6000 is above the longest thinking-off summary measured, from a small
        // sample (Agents/Compaction.md, decision 8); a quarter of a short context leaves it room to answer.
        session().withCompaction( {
            .summary_tokens = std::min<Mila::Dnn::dim_t>( 6000, model_.agent().plan().contextLength() / 4 ),
            .template_variables = { { "enable_thinking", "false" } } } );
    }

    void SessionProcess::refuseAttachment( const nlohmann::json& request )
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

    void SessionProcess::sendState()
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
            const Mila::Agents::SamplingOverride& chosen = settings_.sampling;

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

        // Counted whole between turns; within one, each round's start says it (onAgentEvent).
        if ( work_ != Work::Turn )
        {
            measureContext();
        }

        const std::optional<dim_t> compacts_at = resident ? model_.agent().compactionThreshold() : std::nullopt;

        send( "session.state", {
            { "model", resident ? nlohmann::ordered_json( model_.name() ) : nlohmann::ordered_json( nullptr ) },
            { "loading", loading_.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( loading_ ) },
            { "no_model_reason", no_model_reason_.empty() || resident
                ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( no_model_reason_ ) },
            { "context_length", context_length },
            { "context_automatic", !settings_.context_length.has_value() },
            { "context_used", context_used_ ? nlohmann::ordered_json( *context_used_ ) : nlohmann::ordered_json( nullptr ) },
            { "compacts_at", compacts_at ? nlohmann::ordered_json( *compacts_at ) : nlohmann::ordered_json( nullptr ) },
            { "thinking_capable", thinking_capable },
            { "thinking", thinking },
            { "effort", std::move( effort_in_effect ) },
            { "effort_levels", std::move( effort_levels ) },
            { "detail", settings_.detail },
            { "sampling", std::move( sampling ) },
            { "modalities", std::move( modalities ) },
            { "folder", conversationFolder() },
            { "folder_access", !conversation_folder_ ? nlohmann::ordered_json( nullptr )
                : nlohmann::ordered_json( conversation_folder_access_ == Mila::Agents::FolderAccess::ReadWrite ? "read_write" : "read" ) },
            { "allowed_for_chat", allowedForChat() },
            { "project", conversation_project_.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( conversation_project_ ) },
            { "folder_from_project", conversation_folder_.has_value() && conversation_folder_from_project_ },
            { "plan", plan_.toJson() },
            { "web_search", web_search_.has_value() },
            { "run_commands", settings_.run_commands },
            { "run_commands_problem", container_problem_.empty()
                ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( container_problem_ ) },
            { "program_folders", programFoldersShown() },
            { "busy", busy },
            { "license_accepted", licenseAccepted() },
            { "streams", nlohmann::ordered_json::array( { stream } ) } } );
    }

    void SessionProcess::measureContext()
    {
        if ( !model_.resident() )
        {
            context_used_.reset();

            return;
        }

        try
        {
            context_used_ = model_.agent().contextTokens();
        }
        catch ( const std::invalid_argument& )
        {
            // A template variable the template refuses is said where it is set; the count waits for it.
            context_used_.reset();
        }
    }

    nlohmann::ordered_json SessionProcess::programFoldersShown() const
    {
        nlohmann::ordered_json shown = nlohmann::ordered_json::array();

        if ( container_ )
        {
            for ( const std::filesystem::path& folder : programs_.folders )
            {
                const std::u8string text = folder.u8string();
                shown.push_back( std::string( text.begin(), text.end() ) );
            }
        }

        return shown;
    }

    nlohmann::json SessionProcess::conversationFolder() const
    {
        return conversation_folder_ ? nlohmann::json( *conversation_folder_ ) : nlohmann::json( nullptr );
    }

    nlohmann::ordered_json SessionProcess::allowedForChat() const
    {
        nlohmann::ordered_json allowed = nlohmann::ordered_json::array();

        for ( const std::string& name : session().gate().allowedForConversation() )
        {
            allowed.push_back( name );
        }

        return allowed;
    }

    void SessionProcess::sendConversations()
    {
        nlohmann::json items = store_.list();

        // Each file Mila wrote is compared with the disk as the list is drawn (AssistantInterface.md 4.3, *Needs you*).
        for ( nlohmann::json& item : items )
        {
            if ( !item.contains( "wrote" ) || !item[ "wrote" ].is_array() )
            {
                continue;
            }

            for ( nlohmann::json& file : item[ "wrote" ] )
            {
                file[ "disk" ] = diskState( file.value( "written", nlohmann::json( nullptr ) ) );
                file.erase( "written" );
            }
        }

        send( "conversations.list", { { "items", std::move( items ) } } );
    }

    void SessionProcess::openCreation( const nlohmann::json& request )
    {
        // Read from the store: between turns the open chat's file holds every message it has.
        const std::string id = request.value( "conversation_id", std::string{} );
        const std::optional<nlohmann::json> conversation = id.empty() ? std::nullopt : store_.load( id );
        std::optional<nlohmann::json> creation;

        if ( conversation.has_value() && request.contains( "path" ) && request[ "path" ].is_string() )
        {
            creation = fileCreation( *conversation, request[ "path" ].get<std::string>() );
        }
        else if ( conversation.has_value() && request.contains( "message" ) && request[ "message" ].is_number_unsigned() )
        {
            creation = replyCreation( *conversation, request[ "message" ].get<std::size_t>() );
        }

        if ( !creation.has_value() )
        {
            sendError( "That is no longer among what Mila made.", nullptr );

            return;
        }

        nlohmann::json content = { { "content", nullptr }, { "content_kind", nullptr } };
        nlohmann::json disk = nullptr;

        if ( ( *creation )[ "kind" ] == "file" )
        {
            const nlohmann::json location = ( *creation )[ "location" ];

            if ( location.is_string() )
            {
                const std::string text = location.get<std::string>();

                content = showableContent( std::filesystem::path( std::u8string( text.begin(), text.end() ) ) );
            }

            // The last write's record, which only the stored calls hold.
            for ( const nlohmann::json& file : summarizeConversation( *conversation )[ "wrote" ] )
            {
                if ( file[ "path" ] == ( *creation )[ "path" ] )
                {
                    disk = diskState( file[ "written" ] );
                }
            }
        }

        send( "creation.loaded", { { "creation", std::move( *creation ) }, { "content", content[ "content" ] },
            { "content_kind", content[ "content_kind" ] }, { "disk", std::move( disk ) } } );
    }

    void SessionProcess::keepReply( const nlohmann::json& request )
    {
        const std::string id = request.value( "conversation_id", std::string{} );
        const nlohmann::json index = request.value( "message", nlohmann::json( nullptr ) );
        const bool keep = request.value( "keep", true );

        if ( id.empty() || !index.is_number_unsigned() )
        {
            return;
        }

        const std::size_t message = index.get<std::size_t>();

        const auto mark = [ & ]( nlohmann::json& messages )
            {
                if ( !messages.is_array() || message >= messages.size() || !messages[ message ].is_object()
                    || messages[ message ].value( "role", std::string{} ) != "assistant" )
                {
                    return false;
                }

                messages[ message ][ "kept" ] = keep ? nlohmann::json( timestamp() ) : nlohmann::json( nullptr );

                return true;
            };

        if ( id == conversation_id_ )
        {
            if ( mark( conversation_messages_ ) )
            {
                saveConversation();
            }
        }
        else if ( std::optional<nlohmann::json> stored = store_.load( id ) )
        {
            if ( stored->contains( "messages" ) && mark( ( *stored )[ "messages" ] ) )
            {
                store_.save( *stored );
            }
        }

        sendConversations();
    }

    void SessionProcess::sendProjects()
    {
        nlohmann::json items = nlohmann::json::array();
        const std::vector<nlohmann::json> projects = projects_.list();

        // Whether a project made from Mila's suggestion can still be undone, which the overview's card offers.
        for ( const nlohmann::json& project : projects )
        {
            nlohmann::json item = project;
            item[ "undo" ] = undoable( project, projects );

            // What Mila understands, and a note waiting for the person, which the project's page shows in full (5.6).
            const nlohmann::json understanding = projects_.understanding( project.value( "id", std::string{} ) );
            item[ "understanding" ] = understanding.value( "current", nlohmann::json( nullptr ) );
            item[ "waiting" ] = understanding.value( "waiting", nlohmann::json( nullptr ) );
            items.push_back( std::move( item ) );
        }

        send( "projects.list", { { "items", std::move( items ) } } );
    }

    void SessionProcess::sendActivity( const std::string& before )
    {
        constexpr std::size_t kMostEvents = 200;

        std::vector<nlohmann::json> events = activity_.read();

        for ( const nlohmann::json& project : projects_.list() )
        {
            for ( nlohmann::json& event : projects_.history( project.value( "id", std::string{} ) ).read() )
            {
                events.push_back( std::move( event ) );
            }
        }

        // A chat moved between two projects is in the history of each, the same event twice.
        std::set<std::string> seen;
        std::vector<nlohmann::json> kept;

        for ( nlohmann::json& event : events )
        {
            const std::string time = event.value( "time", std::string{} );

            if ( ( !before.empty() && !( time < before ) ) || !seen.insert( event.dump() ).second )
            {
                continue;
            }

            kept.push_back( std::move( event ) );
        }

        std::ranges::stable_sort( kept, []( const nlohmann::json& left, const nlohmann::json& right )
            {
                return left.value( "time", std::string{} ) > right.value( "time", std::string{} );
            } );

        const bool more = kept.size() > kMostEvents;
        kept.resize( std::min( kept.size(), kMostEvents ) );

        nlohmann::json items = nlohmann::json::array();

        for ( nlohmann::json& event : kept )
        {
            items.push_back( std::move( event ) );
        }

        const nlohmann::json earliest = more ? nlohmann::json( items.back().value( "time", std::string{} ) ) : nlohmann::json( nullptr );

        send( "activity.list", { { "items", std::move( items ) }, { "before", earliest } } );
    }

    // ---- Projects ---------------------------------------------------------------------------

    void SessionProcess::newConversation( const nlohmann::json& request )
    {
        const nlohmann::json named = request.value( "project", nlohmann::json( nullptr ) );
        const std::optional<nlohmann::json> project = projectOf( named.is_string() ? named.get<std::string>() : std::string{} );

        // A chat begins with no folder, whatever an older page asks for (AssistantInterface.md 5.5), unless its project
        // holds one (10).
        const auto starting = project.has_value() ? startingFolder( *project ) : std::nullopt;

        if ( starting.has_value() )
        {
            useChat( starting->first, starting->second, true );
        }
        else
        {
            useChat( {} );
        }

        startNewConversation();
        conversation_project_ = project.has_value() ? project->value( "id", std::string{} ) : std::string{};

        // The project's instructions and its note's tools, which a chat in no project is not offered.
        offerTools();
        sendState();
    }

    void SessionProcess::newProject( const nlohmann::json& request )
    {
        const std::string name = trimmed( request.value( "name", std::string{} ) );

        if ( name.empty() )
        {
            sendError( "A project needs a name.", nullptr );

            return;
        }

        const std::string id = makeProject( name, trimmed( request.value( "purpose", std::string{} ) ), nullptr );

        for ( const nlohmann::json& moved : request.value( "conversation_ids", nlohmann::json::array() ) )
        {
            if ( moved.is_string() )
            {
                moveOne( moved.get<std::string>(), id );
            }
        }

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
        send( "project.made", { { "id", id } } );
    }

    std::string SessionProcess::makeProject( const std::string& name, const std::string& purpose, const nlohmann::json& suggested )
    {
        const std::string now = timestamp();
        const std::string id = projects_.newIdentifier();

        nlohmann::json project = {
            { "v", 1 },
            { "id", id },
            { "name", name },
            { "purpose", purpose },
            { "instructions", "" },
            { "folders", nlohmann::json::array() },
            { "created", now },
            { "updated", now } };

        nlohmann::json event = { { "kind", "project.made" }, { "project", id }, { "name", name } };

        // Kept so the project can be undone while nothing has changed it (AssistantInterface.md 6).
        if ( suggested.is_object() )
        {
            project[ "suggested" ] = suggested;
            event[ "suggested" ] = true;
            event[ "conversation" ] = suggested.value( "conversation_id", std::string{} );
        }

        projects_.save( project );
        recordEvent( std::move( event ) );

        return id;
    }

    void SessionProcess::updateProject( const nlohmann::json& request )
    {
        const std::string id = request.value( "id", std::string{} );
        std::optional<nlohmann::json> project = projectOf( id );

        if ( !project.has_value() )
        {
            sendError( "That project is no longer stored.", nullptr );

            return;
        }

        std::vector<nlohmann::json> events;

        if ( request.contains( "name" ) )
        {
            const std::string name = trimmed( request[ "name" ].is_string() ? request[ "name" ].get<std::string>() : std::string{} );

            if ( name.empty() )
            {
                sendError( "A project needs a name.", nullptr );

                return;
            }

            ( *project )[ "name" ] = name;
            events.push_back( { { "kind", "project.renamed" }, { "name", name } } );
        }
        else if ( request.contains( "purpose" ) )
        {
            ( *project )[ "purpose" ] = trimmed( request[ "purpose" ].is_string() ? request[ "purpose" ].get<std::string>() : std::string{} );
            events.push_back( { { "kind", "project.purpose" }, { "name", project->value( "name", std::string{} ) } } );
        }
        else if ( request.contains( "instructions" ) )
        {
            ( *project )[ "instructions" ] = request[ "instructions" ].is_string() ? request[ "instructions" ].get<std::string>() : std::string{};
            events.push_back( { { "kind", "project.instructions" }, { "name", project->value( "name", std::string{} ) } } );
        }
        else if ( request.contains( "understanding" ) )
        {
            // The note is its own file and its own events; the project's record is not changed by it.
            changeUnderstanding( id, request[ "understanding" ] );
            sendProjects();
            sendActivity( {} );

            return;
        }
        else if ( request.contains( "folders" ) )
        {
            const std::optional<nlohmann::json> folders = projectFolders( request[ "folders" ] );

            if ( !folders.has_value() )
            {
                sendError( "A project's folders must each be a folder on this computer, kept to read or to read and change.", nullptr );

                return;
            }

            const nlohmann::json before = project->value( "folders", nlohmann::json::array() );

            for ( const nlohmann::json& folder : *folders )
            {
                const std::string path = folder[ "path" ].get<std::string>();
                const std::optional<Mila::Agents::FolderAccess> was = projectAccess( *project, path );

                if ( !was.has_value() )
                {
                    events.push_back( { { "kind", "project.folder_added" }, { "path", path }, { "access", folder[ "access" ] } } );
                }
                else if ( *was != folderAccessOf( folder ) )
                {
                    events.push_back( { { "kind", "project.folder_access" }, { "path", path }, { "access", folder[ "access" ] } } );
                }
            }

            for ( const nlohmann::json& folder : before )
            {
                const std::string path = folder.value( "path", std::string{} );
                const bool kept = std::ranges::any_of( *folders, [&path]( const nlohmann::json& now ) { return now[ "path" ] == path; } );

                if ( !kept )
                {
                    events.push_back( { { "kind", "project.folder_removed" }, { "path", path } } );
                }
            }

            ( *project )[ "folders" ] = *folders;
        }
        else
        {
            return;
        }

        ( *project )[ "updated" ] = timestamp();
        projects_.save( *project );

        for ( nlohmann::json& event : events )
        {
            event[ "project" ] = id;
            recordEvent( std::move( event ) );
        }

        // The open chat reads the project's name, purpose and instructions as they now stand from its next response; its
        // tools and what the person allowed for it stay.
        if ( conversation_project_ == id && ( request.contains( "instructions" ) || request.contains( "name" ) || request.contains( "purpose" ) ) )
        {
            std::set<std::string, std::less<>> offered;

            for ( const Mila::Agents::Tool& tool : session().tools() )
            {
                offered.insert( tool.name );
            }

            session().withSystemPrompt( systemPrompt( offered ) );
        }

        // The open chat loses a folder it held through the project once the project lets it go, and is narrowed with
        // it; a project widened reaches its chats from their next start. Requests wait for any reply being written.
        if ( conversation_project_ == id && conversation_folder_ && conversation_folder_from_project_ )
        {
            const std::optional<Mila::Agents::FolderAccess> held = projectAccess( *project, *conversation_folder_ );

            if ( !held.has_value() )
            {
                conversation_folder_.reset();
                conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
                conversation_folder_from_project_ = false;
                refreshTools();
            }
            else if ( *held == Mila::Agents::FolderAccess::Read && conversation_folder_access_ == Mila::Agents::FolderAccess::ReadWrite )
            {
                conversation_folder_access_ = Mila::Agents::FolderAccess::Read;
                grantChat();
            }

            if ( !conversation_id_.empty() )
            {
                saveConversation();
            }
        }

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
    }

    void SessionProcess::deleteProject( const nlohmann::json& request )
    {
        const std::string id = request.value( "id", std::string{} );

        if ( !projectOf( id ).has_value() )
        {
            sendError( "That project is no longer stored.", nullptr );

            return;
        }

        const bool delete_chats = request.value( "conversations", std::string( "keep" ) ) == "delete";

        for ( const nlohmann::json& item : store_.list() )
        {
            if ( item.value( "project", nlohmann::json( nullptr ) ) != id )
            {
                continue;
            }

            const std::string chat = item.value( "id", std::string{} );

            if ( delete_chats )
            {
                store_.remove( chat );
                forgetEverywhere( chat );
                clearNotesFrom( chat );
            }
            else if ( std::optional<nlohmann::json> stored = store_.load( chat ) )
            {
                ( *stored )[ "project" ] = nullptr;
                ( *stored )[ "folder_from_project" ] = false;
                store_.save( *stored );
            }
        }

        // The open chat goes with the project's chats, or stays in no project keeping its folder as its own.
        if ( conversation_project_ == id )
        {
            if ( delete_chats )
            {
                useChat( {} );
                startNewConversation();
            }
            else
            {
                conversation_project_.clear();
                conversation_folder_from_project_ = false;
            }
        }

        projects_.remove( id );
        forgetEverywhere( id );
        recordEvent( { { "kind", "project.deleted" } } );

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
    }

    void SessionProcess::moveConversation( const nlohmann::json& request )
    {
        const nlohmann::json named = request.value( "project", nlohmann::json( nullptr ) );
        const std::string project = named.is_string() ? named.get<std::string>() : std::string{};

        if ( !project.empty() && !projectOf( project ).has_value() )
        {
            sendError( "That project is no longer stored.", nullptr );

            return;
        }

        moveOne( request.value( "id", std::string{} ), project );

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
    }

    void SessionProcess::moveOne( const std::string& id, const std::string& project, const nlohmann::json& details )
    {
        std::optional<nlohmann::json> stored = store_.load( id );

        if ( !stored.has_value() )
        {
            return;
        }

        const nlohmann::json was = stored->value( "project", nlohmann::json( nullptr ) );
        const std::string from = was.is_string() ? was.get<std::string>() : std::string{};

        if ( from == project )
        {
            return;
        }

        // One field moves it, and the chat's last change stays when it was.
        ( *stored )[ "project" ] = project.empty() ? nlohmann::json( nullptr ) : nlohmann::json( project );
        ( *stored )[ "folder_from_project" ] = false;
        store_.save( *stored );

        if ( id == conversation_id_ )
        {
            conversation_project_ = project;
            conversation_folder_from_project_ = false;

            // The project's instructions from the chat's next round, and no note: the chat has its own context (phase
            // 7's decisions).
            offerToolsWhenSafe();
        }

        nlohmann::json event = { { "kind", "chat.moved" }, { "conversation", id }, { "name", stored->value( "title", std::string{} ) },
            { "project", project.empty() ? nlohmann::json( nullptr ) : nlohmann::json( project ) },
            { "from", from.empty() ? nlohmann::json( nullptr ) : nlohmann::json( from ) } };

        if ( details.is_object() )
        {
            event.update( details );
        }

        recordEvent( std::move( event ) );
    }

    // ---- Offers ---------------------------------------------------------------------------

    std::string SessionProcess::proposeProject( const std::string& arguments )
    {
        const bool names_allowed = session().exposure().empty();
        const std::vector<nlohmann::json> projects = projects_.list();
        const nlohmann::json offered = offerProject( arguments, store_.list(), projects, conversation_id_, names_allowed );

        if ( offered[ "offer" ].is_object() )
        {
            recordEvent( { { "kind", "suggestion.made" }, { "who", "mila" }, { "conversation", conversation_id_ },
                { "name", offered[ "offer" ][ "name" ] } } );
            keepOffer( offered[ "offer" ] );
        }

        return offered[ "result" ].get<std::string>();
    }

    std::string SessionProcess::moveToProject( const std::string& arguments )
    {
        const bool names_allowed = session().exposure().empty();

        // Mila never moves a chat the person filed, nor one it filed before.
        if ( !conversation_project_.empty() )
        {
            const std::optional<nlohmann::json> held = projectOf( conversation_project_ );

            return std::format( "This chat is already in the project {}, so it stays there.",
                held.has_value() ? held->value( "name", std::string{} ) : std::string{} );
        }

        if ( conversation_id_.empty() )
        {
            return "This chat is not stored yet, so it was not moved.";
        }

        const std::vector<nlohmann::json> projects = projects_.list();
        const nlohmann::json resolved = projectForMove( arguments, projects, names_allowed );

        if ( !resolved.contains( "project" ) )
        {
            return resolved[ "result" ].get<std::string>();
        }

        const std::string id = resolved[ "project" ][ "id" ].get<std::string>();
        const std::string name = resolved[ "project" ][ "name" ].get<std::string>();

        // In a chat that has read from outside the person was asked, with the call in view, so allowing it is their yes.
        // Elsewhere it is done and said once they have said yes to one move into this project (question A).
        const bool said = !names_allowed || filingAllowed( projects_.history( id ).read() );

        nlohmann::json offer = { { "kind", "move" }, { "state", said ? "accepted" : "open" }, { "project", id },
            { "name", name }, { "in_window", !names_allowed }, { "said", said } };

        if ( said )
        {
            moveOne( conversation_id_, id, names_allowed
                ? nlohmann::json{ { "who", "mila" } } : nlohmann::json{ { "suggested", "move" } } );
            offer[ "decided" ] = timestamp();
        }
        else
        {
            recordEvent( { { "kind", "suggestion.made" }, { "who", "mila" }, { "conversation", conversation_id_ },
                { "project", id }, { "name", name } } );
        }

        keepOffer( std::move( offer ) );

        if ( !said )
        {
            return std::format( "Offered to move this chat into the project {}. The user answers in their next message; "
                "do not move it yourself or offer it again. In a sentence, say what you suggested.", name );
        }

        return names_allowed
            ? std::format( "Moved this chat into the project {}, and the user was told; they can say keep it separate to "
                "undo it. In a sentence, say that you moved it.", name )
            : std::format( "Moved this chat into the project {}, as the user allowed.", name );
    }

    void SessionProcess::keepOffer( nlohmann::json offer )
    {
        offer[ "time" ] = timestamp();

        // One open offer at a time: a later one in the same reply replaces it.
        for ( nlohmann::json& call : turn_calls_ )
        {
            if ( call.contains( "offer" ) && call[ "offer" ].is_object() && call[ "offer" ].value( "state", std::string{} ) == "open" )
            {
                call[ "offer" ][ "state" ] = "replaced";
            }
        }

        if ( announced_call_id_.empty() )
        {
            return;
        }

        storedCall( announced_call_id_ )[ "offer" ] = offer;

        send( "turn.offer", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", announced_call_id_ },
            { "offer", std::move( offer ) } } );
    }

    bool SessionProcess::answerOffer( const std::string& text )
    {
        if ( conversation_id_.empty() )
        {
            return false;
        }

        const nlohmann::json pending = pendingOffer( conversation_messages_ );

        if ( pending.is_null() )
        {
            return false;
        }

        nlohmann::json& call = conversation_messages_[ pending[ "message" ].get<std::size_t>() ][ "tool_calls" ][ pending[ "call" ].get<std::size_t>() ];
        nlohmann::json& offer = call[ "offer" ];
        const std::string call_id = call.value( "call_id", std::string{} );
        const OfferAnswer answer = readOfferAnswer( text );
        const bool open = offer.value( "state", std::string{} ) == "open";
        const bool words_count = open && !offer.value( "in_window", false );
        std::string line;

        if ( words_count && answer == OfferAnswer::Accept )
        {
            line = takeOffer( offer, conversation_id_, call_id, offer.value( "name", std::string{} ) );
        }
        else if ( words_count && answer == OfferAnswer::Decline )
        {
            line = declineOffer( offer, conversation_id_ );
        }
        else if ( !open && ( answer == OfferAnswer::KeepSeparate || answer == OfferAnswer::Decline ) )
        {
            line = keepSeparate( offer, conversation_id_ );
        }
        else
        {
            // The next thing said closes the offer either way; a project's waits on the overview's card.
            if ( open )
            {
                offer[ "state" ] = "undecided";
            }

            return false;
        }

        const nlohmann::json answered = offer;

        // Kept as the model reads it: the person's words, then the app's line in Mila's place.
        conversation_messages_.push_back( { { "role", "user" }, { "content", text }, { "thought", nullptr },
            { "tool_calls", nullptr }, { "attachments", nullptr }, { "stats", nullptr } } );
        conversation_messages_.push_back( { { "role", "assistant" }, { "content", line }, { "thought", nullptr },
            { "tool_calls", nullptr }, { "attachments", nullptr }, { "stats", nullptr }, { "app", true } } );

        Turn said;
        said.role = Role::User;
        said.content = text;
        session().turns().push_back( std::move( said ) );

        Turn kept;
        kept.role = Role::Assistant;
        kept.content = line;
        session().turns().push_back( std::move( kept ) );

        turn_id_ = std::format( "turn-{}", ++turn_number_ );

        send( "turn.begin", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "conversation_id", conversation_id_ } } );
        send( "turn.answered", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "call_id", call_id },
            { "offer", answered }, { "text", line } } );
        send( "turn.end", { { "stream_id", kStreamId }, { "turn_id", turn_id_ }, { "finish", "stop" }, { "answer", "" },
            { "message", conversation_messages_.size() - 1 } } );

        saveConversation();
        sendProjects();
        sendConversations();
        sendActivity( {} );

        return true;
    }

    void SessionProcess::decideSuggestion( const nlohmann::json& request )
    {
        const std::string id = request.value( "conversation_id", std::string{} );
        const std::string call_id = request.value( "call_id", std::string{} );
        const std::string decision = request.value( "decision", std::string{} );

        if ( decision != "accept" && decision != "decline" )
        {
            return;
        }

        const bool is_open = !id.empty() && id == conversation_id_;
        std::optional<nlohmann::json> stored = is_open || id.empty() ? std::nullopt : store_.load( id );
        nlohmann::json* found = is_open ? offerOnCall( conversation_messages_, call_id )
            : stored.has_value() && stored->contains( "messages" ) ? offerOnCall( ( *stored )[ "messages" ], call_id ) : nullptr;

        const std::string state = found != nullptr ? found->value( "state", std::string{} ) : std::string{};

        if ( found == nullptr || found->value( "kind", std::string{} ) != "project" || ( state != "open" && state != "undecided" ) )
        {
            sendError( "That suggestion has already been answered.", nullptr );
            sendConversations();

            return;
        }

        nlohmann::json offer = *found;

        if ( decision == "accept" )
        {
            const nlohmann::json edited = request.value( "name", nlohmann::json( nullptr ) );
            std::string name = trimmed( edited.is_string() ? edited.get<std::string>() : std::string{} );

            if ( name.empty() )
            {
                name = offer.value( "name", std::string{} );
            }

            if ( name.empty() )
            {
                sendError( "A project needs a name.", nullptr );

                return;
            }

            takeOffer( offer, id, call_id, name );
        }
        else
        {
            declineOffer( offer, id );
        }

        // The chat's own copy is written last: taking the offer moved it, which its stored file now says.
        if ( is_open )
        {
            if ( nlohmann::json* kept = offerOnCall( conversation_messages_, call_id ) )
            {
                *kept = offer;
            }

            saveConversation();
        }
        else
        {
            changeStoredOffer( id, call_id, [&offer]( nlohmann::json& kept ) { kept = offer; } );
        }

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
    }

    std::string SessionProcess::takeOffer( nlohmann::json& offer, const std::string& conversation_id, const std::string& call_id,
        const std::string& name )
    {
        offer[ "decided" ] = timestamp();

        if ( offer.value( "kind", std::string{} ) == "move" )
        {
            const std::string project = offer.value( "project", std::string{} );
            const std::optional<nlohmann::json> held = projectOf( project );

            if ( !held.has_value() )
            {
                offer[ "state" ] = "lapsed";

                return std::format( "The project {} is no longer there, so this chat stays where it is.", name );
            }

            moveOne( conversation_id, project, { { "suggested", "move" } } );
            offer[ "state" ] = "accepted";

            return std::format( "Moved this chat into {}.", held->value( "name", name ) );
        }

        const std::string made = makeProject( name, offer.value( "purpose", std::string{} ),
            { { "conversation_id", conversation_id }, { "call_id", call_id } } );

        std::size_t moved = 0;
        bool asking_moved = false;

        for ( const nlohmann::json& chat : offer.value( "chats", nlohmann::json::array() ) )
        {
            const std::string id = chat.value( "id", std::string{} );
            const std::optional<nlohmann::json> stored = store_.load( id );

            // A chat the person filed or deleted since the offer stays as they left it.
            if ( !stored.has_value() || stored->value( "project", nlohmann::json( nullptr ) ).is_string() )
            {
                continue;
            }

            moveOne( id, made, { { "suggested", "project" } } );
            ++moved;
            asking_moved = asking_moved || id == conversation_id;
        }

        offer[ "state" ] = "accepted";
        offer[ "project" ] = made;
        offer[ "name" ] = name;

        const std::size_t others = moved - ( asking_moved ? 1 : 0 );

        if ( !asking_moved )
        {
            return std::format( "Made the project {} and moved {} chat{} into it.", name, moved, moved == 1 ? "" : "s" );
        }

        return others == 0 ? std::format( "Made the project {} and moved this chat into it.", name )
            : std::format( "Made the project {} and moved this chat and {} other{} into it.", name, others, others == 1 ? "" : "s" );
    }

    std::string SessionProcess::declineOffer( nlohmann::json& offer, const std::string& conversation_id )
    {
        const std::string name = offer.value( "name", std::string{} );
        const bool move = offer.value( "kind", std::string{} ) == "move";

        offer[ "state" ] = "declined";
        offer[ "decided" ] = timestamp();

        nlohmann::json event = { { "kind", "suggestion.declined" }, { "conversation", conversation_id }, { "name", name } };

        if ( move )
        {
            event[ "project" ] = offer.value( "project", std::string{} );
        }

        recordEvent( std::move( event ) );

        return move ? std::format( "Kept this chat out of {}.", name ) : std::format( "Did not make the project {}.", name );
    }

    std::string SessionProcess::keepSeparate( nlohmann::json& offer, const std::string& conversation_id )
    {
        const std::string name = offer.value( "name", std::string{} );

        moveOne( conversation_id, {} );
        offer[ "state" ] = "undone";
        offer[ "decided" ] = timestamp();

        return std::format( "Kept this chat separate from {}.", name );
    }

    void SessionProcess::changeStoredOffer( const std::string& conversation_id, const std::string& call_id,
        const std::function<void( nlohmann::json& )>& change )
    {
        std::optional<nlohmann::json> stored = store_.load( conversation_id );

        if ( !stored.has_value() || !stored->contains( "messages" ) )
        {
            return;
        }

        if ( nlohmann::json* offer = offerOnCall( ( *stored )[ "messages" ], call_id ) )
        {
            change( *offer );
            store_.save( *stored );
        }
    }

    void SessionProcess::undoProject( const nlohmann::json& request )
    {
        const std::string id = request.value( "id", std::string{} );
        const std::optional<nlohmann::json> project = projectOf( id );

        if ( !project.has_value() || !undoable( *project, projects_.list() ) )
        {
            sendError( "That project can no longer be undone. It can be deleted from its page.", nullptr );
            sendProjects();

            return;
        }

        // Its chats go back to no project, where every chat a suggestion gathers came from.
        for ( const nlohmann::json& item : store_.list() )
        {
            if ( item.value( "project", nlohmann::json( nullptr ) ) != id )
            {
                continue;
            }

            if ( std::optional<nlohmann::json> stored = store_.load( item.value( "id", std::string{} ) ) )
            {
                ( *stored )[ "project" ] = nullptr;
                ( *stored )[ "folder_from_project" ] = false;
                store_.save( *stored );
            }
        }

        const nlohmann::json suggested = project->value( "suggested", nlohmann::json::object() );
        const std::string offered_in = suggested.value( "conversation_id", std::string{} );
        const std::string call_id = suggested.value( "call_id", std::string{} );

        if ( conversation_project_ == id )
        {
            conversation_project_.clear();
            conversation_folder_from_project_ = false;
        }

        if ( !offered_in.empty() && offered_in == conversation_id_ )
        {
            if ( nlohmann::json* offer = offerOnCall( conversation_messages_, call_id ) )
            {
                ( *offer )[ "state" ] = "undone";
            }
        }
        else if ( !offered_in.empty() )
        {
            changeStoredOffer( offered_in, call_id, []( nlohmann::json& offer ) { offer[ "state" ] = "undone"; } );
        }

        if ( !conversation_id_.empty() )
        {
            saveConversation();
        }

        const std::string name = project->value( "name", std::string{} );

        projects_.remove( id );
        forgetEverywhere( id );
        recordEvent( { { "kind", "project.undone" }, { "name", name } } );

        sendProjects();
        sendConversations();
        sendState();
        sendActivity( {} );
    }

    bool SessionProcess::undoable( const nlohmann::json& project, const std::vector<nlohmann::json>& projects ) const
    {
        try
        {
            return projectUndoable( project, projects_.history( project.value( "id", std::string{} ) ).read(), projects );
        }
        catch ( const std::invalid_argument& )
        {
            return false;
        }
    }

    std::optional<nlohmann::json> SessionProcess::projectOf( const std::string& id ) const
    {
        if ( id.empty() )
        {
            return std::nullopt;
        }

        return projects_.load( id );
    }

    std::optional<std::pair<std::string, Mila::Agents::FolderAccess>> SessionProcess::startingFolder( const nlohmann::json& project )
    {
        const nlohmann::json folders = project.value( "folders", nlohmann::json::array() );

        if ( !folders.is_array() || folders.size() != 1 || !folders[ 0 ].is_object() || !folders[ 0 ].contains( "path" )
            || !folders[ 0 ][ "path" ].is_string() )
        {
            return std::nullopt;
        }

        return std::pair{ folders[ 0 ][ "path" ].get<std::string>(), folderAccessOf( folders[ 0 ] ) };
    }

    std::optional<Mila::Agents::FolderAccess> SessionProcess::projectAccess( const nlohmann::json& project, std::string_view folder )
    {
        if ( folder.empty() )
        {
            return std::nullopt;
        }

        const nlohmann::json folders = project.value( "folders", nlohmann::json::array() );

        if ( !folders.is_array() )
        {
            return std::nullopt;
        }

        for ( const nlohmann::json& entry : folders )
        {
            if ( entry.is_object() && entry.contains( "path" ) && entry[ "path" ].is_string() && entry[ "path" ].get<std::string>() == folder )
            {
                return folderAccessOf( entry );
            }
        }

        return std::nullopt;
    }

    void SessionProcess::recordEvent( nlohmann::json event )
    {
        if ( !event.contains( "time" ) )
        {
            event[ "time" ] = timestamp();
        }

        if ( !event.contains( "who" ) )
        {
            event[ "who" ] = "you";
        }

        const auto storedProject = [this, &event]( const char* field )
            {
                return event.contains( field ) && event[ field ].is_string() && projectOf( event[ field ].get<std::string>() ).has_value()
                    ? event[ field ].get<std::string>() : std::string{};
            };

        const std::string to = storedProject( "project" );
        const std::string from = storedProject( "from" );

        try
        {
            if ( !to.empty() )
            {
                projects_.history( to ).append( event );
            }

            if ( !from.empty() && from != to )
            {
                projects_.history( from ).append( event );
            }

            if ( to.empty() && from.empty() )
            {
                activity_.append( event );
            }
        }
        catch ( const std::exception& failure )
        {
            sendError( std::format( "This could not be kept in Activity: {}", failure.what() ), nullptr );
        }
    }

    void SessionProcess::forgetEverywhere( const std::string& id )
    {
        try
        {
            activity_.forget( id );

            for ( const nlohmann::json& project : projects_.list() )
            {
                projects_.history( project.value( "id", std::string{} ) ).forget( id );
            }
        }
        catch ( const std::exception& failure )
        {
            sendError( std::format( "Activity could not forget what was deleted: {}", failure.what() ), nullptr );
        }
    }

    void SessionProcess::sendModels( bool online )
    {
        nlohmann::json published = nullptr;
        nlohmann::json suggested = nullptr;

        if ( online )
        {
            try
            {
                // With a model resident the list is planned against the whole card, as a listing is (Deployment.md
                // 12.5), so the memory that model holds does not count against the others.
                DeploymentRequest request = requestFor( std::nullopt );

                if ( model_.resident() )
                {
                    request.withWholeDeviceMemory();
                }

                nlohmann::json listed = publishedModels( request );
                published = std::move( listed[ "models" ] );
                published_display_names_.clear();

                for ( const auto& model : published )
                {
                    const std::string display_name = model.value( "display_name", std::string{} );

                    if ( !display_name.empty() )
                    {
                        published_display_names_[ model.value( "name", std::string{} ) ] = display_name;
                    }
                }
                suggested = std::move( listed[ "suggested" ] );
            }
            catch ( const std::exception& failure )
            {
                published = nullptr;
                sendError( std::format( "Could not list the models that can be installed: {}", failure.what() ), "models" );
            }
        }

        send( "models.list", { { "installed", installedModels() }, { "published", std::move( published ) },
            { "suggested", std::move( suggested ) } } );
    }

    void SessionProcess::sendError( std::string_view text, nlohmann::ordered_json during )
    {
        send( "error", { { "text", text }, { "during", std::move( during ) } } );
    }

    void SessionProcess::send( std::string_view type, nlohmann::ordered_json fields )
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

    nlohmann::ordered_json SessionProcess::workName() const
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

    std::string_view SessionProcess::loadTask() const
    {
        return work_ == Work::Settings ? "rebuild" : "load";
    }

    std::string_view SessionProcess::finishName( GenerateStatus status )
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

    std::string_view SessionProcess::answerName( ToolAnswer answer )
    {
        switch ( answer )
        {
            case ToolAnswer::Once:            return "allow_once";
            case ToolAnswer::ForConversation: return "allow_conversation";
            case ToolAnswer::Deny:            break;
        }

        return "deny";
    }

    std::string SessionProcess::titleFor( const std::string& text )
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

    std::string SessionProcess::timestamp()
    {
        const auto now = std::chrono::floor<std::chrono::milliseconds>( std::chrono::system_clock::now() );

        return std::format( "{:%FT%TZ}", now );
    }

    std::string_view SessionProcess::roleName( Role role )
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

    nlohmann::json SessionProcess::turnToJson( const Turn& turn )
    {
        nlohmann::json calls = nlohmann::json::array();

        for ( const ToolCall& call : turn.tool_calls )
        {
            calls.push_back( { { "name", call.name }, { "arguments", call.arguments } } );
        }

        return { { "role", roleName( turn.role ) }, { "content", turn.content }, { "tool_calls", std::move( calls ) },
            { "tool_name", turn.tool_name }, { "reasoning", turn.reasoning } };
    }

    Turn SessionProcess::turnFromJson( const nlohmann::json& stored )
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
}
