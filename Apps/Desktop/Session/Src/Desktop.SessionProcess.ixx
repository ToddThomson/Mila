/**
 * @file Desktop.SessionProcess.ixx
 * @brief Mila Desktop's session process: an agent and its session, spoken to as JSON lines on stdio
 *        (MilaDesktop.md 6.1).
 */

export module Desktop.SessionProcess;

import std;
import nlohmann.json;
import Desktop.CommandTool;
import Desktop.ConversationStore;
import Desktop.EventLog;
import Desktop.ProjectStore;
import Desktop.WorkspaceFolder;
import Desktop.ExposureRecord;
import Desktop.MemoryHolder;
import Desktop.MilaLicense;
import Desktop.Models;
import Desktop.Places;
import Desktop.PromptFile;
import Desktop.ProgramSetup;
import Desktop.PromptTool;
import Desktop.RecordSearch;
import Desktop.ResidentAgent;
import Desktop.RoundTimer;
import Desktop.SearchKeyFile;
import Desktop.SearchKeyStore;
import Desktop.Settings;
import Desktop.SettingsFile;
import Desktop.TaskPlan;
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
     * @brief Mila Desktop's session: section 4.2's requests in, section 4.1's events out.
     *
     * One agent at a time over one session, which holds the chat's prompt, tools, rules, grant and limits and
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
         * @param license The licence the package carries, or why it could not be read. Until the person accepts this
         *                text the session installs, loads and answers nothing (AssistantOnboarding.md 4.1).
         * @param prompt_directory Holds `general.json`, every chat's prompt and tools (AssistantInterface.md 5.5).
         */
        SessionProcess( Settings settings, SettingsFile settings_file, std::vector<std::string> problems,
            std::expected<MilaLicense, std::string> license, Mila::Deployment::AllowedDevices devices,
            std::filesystem::path workspace_directory, std::filesystem::path prompt_directory, std::ostream& protocol_out );

        /**
         * @brief Open the model the settings name, then answer requests until @p requests ends.
         *
         * @return The process's exit code.
         */
        int run( std::istream& requests );

    private:

        static constexpr std::string_view kStreamId = "stream-1";

        /// The app's own tool that asks the person for a folder and widens the chat's grant by it (AssistantInterface.md 5.5).
        static constexpr std::string_view kRequestFolderName = "request_folder";

        /// The app's own tool that takes the pane somewhere in the workspace (AssistantInterface.md 5.2).
        static constexpr std::string_view kOpenPlaceName = "open_place";

        /// The app's own tool that finds what the person's past chats said (AssistantInterface.md 5.2).
        static constexpr std::string_view kFindRecordsName = "find_records";

        /// The app's own tool that offers a project gathering chats; the person decides (AssistantInterface.md 6).
        static constexpr std::string_view kProposeProjectName = "propose_project";

        /// The app's own tool that files this chat into a project: offered until the person's first yes for that project,
        /// then done and said (AssistantInterface.md 6, question A).
        static constexpr std::string_view kMoveToProjectName = "move_to_project";

        /// What a request the licence holds back is told; the page asks for the licence before it sends one.
        static constexpr std::string_view kLicenseFirst = "Accept Mila's licence first. It is shown when Mila starts.";

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

            /// False for a call asked every time, which an answer for the conversation does not settle.
            bool conversation_offered{ true };

            /// True for request_folder, whose allowing answer must carry the folder the person chose.
            bool folder_asked{ false };

            /// For open_place in a chat that has read from outside: the places the person chooses among, whose
            /// allowing answer must name one.
            nlohmann::json places;
        };

        using Handler = std::function<Mila::Agents::ToolResult( const std::string& arguments, const Mila::Agents::ToolGrant& grant, std::stop_token stop )>;

        /// The code that runs a tool, what it touches and where its result comes from.
        struct RegisteredTool
        {
            Mila::Agents::ToolNeeds needs;
            Handler run;
            Mila::Agents::ResultOrigin origin{ Mila::Agents::ResultOrigin::Outside };
        };

        Mila::Agents::Session& session() noexcept;

        const Mila::Agents::Session& session() const noexcept;

        // ---- The reading thread -----------------------------------------------------------

        /**
         * @brief Read one request per line until the input ends, which ends the session.
         *
         * A stop and a tool decision are acted on here, because the session's thread is inside the reply they are
         * about. A second turn while one is running is refused here, so that the refusal does not wait for it.
         */
        void readRequests( std::istream& requests );

        void stopReply();

        void decideToolCall( const nlohmann::json& request );

        // ---- The session's thread ---------------------------------------------------------

        void handle( const nlohmann::json& request );

        // ---- A turn -----------------------------------------------------------------------

        void respond( const nlohmann::json& request );

        void onAgentEvent( const AgentEvent& event, RoundTimer& timer, const std::stop_token& stop );

        void sendText( std::string_view channel, std::string_view text );

        /// A round's measurements, and its reasoning and text for the turn as stored.
        void endRound( const RoundTimer& timer, const AgentEvent& event, const std::stop_token& stop );

        /// How the person is asked: the call is shown, and the reply waits for their answer from the reading thread.
        ToolAnswer askAboutCall( const Mila::Agents::ToolQuestion& question );

        /// The person allowed a change: the chat's grant now holds its folder to read and write. The app widens it, on
        /// the person's answer; the gate gives the call the grant as it stands once the answer is in.
        void allowChangesInFolder();

        /**
         * @brief What the call would reach if allowed, from its grant as the gate narrowed it to the tool: each
         *        folder and its access, the program folders, and whether the network -- for a command, only when
         *        the call asks for it, as it runs.
         */
        static nlohmann::json reaches( const Mila::Agents::ToolQuestion& question );

        /// In a chat that has read from outside, what it read, the newest first and at most three, for the question's
        /// *This chat has read* (AssistantInterface.md 5.6); null in a chat that has not.
        nlohmann::json hasRead( const Mila::Agents::ToolQuestion& question ) const;

        /**
         * @brief In a chat that has read from outside, the address or search words a call sends when the person's own
         *        messages never wrote them, compared literally; null otherwise (AssistantInterface.md 5.6).
         */
        nlohmann::json notInPersonsWords( const Mila::Agents::ToolQuestion& question ) const;

        void callAnswered( const ToolCall& call, std::string_view result, bool is_error, bool exposes );

        std::string announceCall( std::string_view name, std::string_view arguments );

        /// The stored record of a call this turn announced.
        nlohmann::json& storedCall( const std::string& call_id );

        /**
         * @brief How the turn ended, from its last round.
         *
         * A reply stopped while a call waited ends on a round that finished its call, so the stop is what says it
         * was cancelled. A turn with no round had no room in the context to start one.
         */
        std::string_view turnFinish() const;

        /// The reply joins the stored conversation, after the message stored on send, as the page shows it and as
        /// the model read it.
        void recordTurn();

        // ---- Conversations ----------------------------------------------------------------

        /// Store the conversation as it stands; `sent` is a message the agent has not taken in yet, stored after it.
        void saveConversation( const Turn* sent = nullptr );

        void startNewConversation();

        /// The session's conversation no longer belongs to a stored one; the next turn starts one.
        void forgetConversation();

        /// A link the person opened from this chat, kept with it for Activity (AssistantInterface.md 5.7).
        void recordLinkOpened( const std::string& url );

        /// `creation.loaded` for a file a chat wrote (`conversation_id`, `path`) or a reply kept (`conversation_id`,
        /// `message`): where it came from, its text where the page can show it, and how it compares with the disk.
        void openCreation( const nlohmann::json& request );

        /// The person kept a reply as made, or let it go (`keep` false); it stays with its chat (AssistantInterface.md 8).
        void keepReply( const nlohmann::json& request );

        /// The person took back something Mila could reach (AssistantInterface.md 4.12): a chat's folder, or what they
        /// allowed for the open chat. A request waits for any reply being written, so it applies from the next.
        void removeReach( const nlohmann::json& request );

        void openConversation( const std::string& id );

        void renameConversation( const std::string& id, std::string title );

        void deleteConversation( const std::string& id );

        bool conversationHasUserTurn() const;

        // ---- Projects (AssistantInterface.md phase 4) ---------------------------------------

        /// A new chat, in the project the request names when it names one that is stored.
        void newConversation( const nlohmann::json& request );

        /// `project.new`: a name, a purpose that may be empty, and chats to move in.
        void newProject( const nlohmann::json& request );

        /// `project.update`: one change of a project's name, purpose, instructions or folders.
        void updateProject( const nlohmann::json& request );

        /// `project.delete`: the project, its history and its record go; its chats are kept in no project or deleted,
        /// as the request says. No folder of the person's is touched.
        void deleteProject( const nlohmann::json& request );

        /// `conversation.move`: a chat into a project, or into none. It keeps the folder it holds.
        void moveConversation( const nlohmann::json& request );

        /// Move one chat, stored or open, and record it in the histories of the projects it left and joined; @p details
        /// join the event, such as who moved it and whether Mila suggested it.
        void moveOne( const std::string& id, const std::string& project, const nlohmann::json& details = nlohmann::json::object() );

        /// A new project's record, stored, and its event; @p suggested names the chat and call it was offered on, or is
        /// null. @return Its id.
        std::string makeProject( const std::string& name, const std::string& purpose, const nlohmann::json& suggested );

        // ---- Offers (AssistantInterface.md 5.3, 5.4 and 6) ----------------------------------

        /// propose_project's work: the offer, kept on its call and shown in the chat, and what the model is told.
        std::string proposeProject( const std::string& arguments );

        /// move_to_project's work: this chat offered into a project, or moved and said once the person has said yes to
        /// one for it; in a chat that has read from outside the person was asked first.
        std::string moveToProject( const std::string& arguments );

        /// The offer on the call being run: kept on it, an open one this reply made before it replaced, and shown.
        void keepOffer( nlohmann::json offer );

        /**
         * @brief The person's message read by app code as an answer to the offer Mila's last reply left (5.4).
         *
         * A bare acceptance or refusal takes or declines an open offer made where words may answer it, and *keep it
         * separate* undoes a move Mila made and said; the message and the app's line are then kept in the chat, and no
         * reply is written. Anything else closes an open offer undecided.
         *
         * @return True when the message answered, so Mila does not reply to it.
         */
        bool answerOffer( const std::string& text );

        /// `suggestion.decide`: a project offer answered on the overview's card, or in the transcript of a chat that has
        /// read from outside, with the name as the person edited it.
        void decideSuggestion( const nlohmann::json& request );

        /**
         * @brief Take an offer: the project made and its chats moved in, or the chat moved into the project.
         *
         * @param offer Changed in place to say it was accepted.
         * @param name The project's name as the person left it.
         * @return The line the chat keeps of it.
         */
        std::string takeOffer( nlohmann::json& offer, const std::string& conversation_id, const std::string& call_id,
            const std::string& name );

        /// Decline an offer, @p offer changed in place, and keep the event. @return The line the chat keeps.
        std::string declineOffer( nlohmann::json& offer, const std::string& conversation_id );

        /// Undo a move Mila made and said: the chat goes back to no project. @return The line the chat keeps.
        std::string keepSeparate( nlohmann::json& offer, const std::string& conversation_id );

        /// The offer kept on @p call_id in a stored chat, changed by @p change and stored again, the chat read fresh so a
        /// move @p change made is kept.
        void changeStoredOffer( const std::string& conversation_id, const std::string& call_id,
            const std::function<void( nlohmann::json& )>& change );

        /// `project.undo`: a project made from a suggestion, while nothing has changed it, goes, and its chats return to
        /// no project; its folders are never touched.
        void undoProject( const nlohmann::json& request );

        /// Whether a stored project can still be undone (Desktop.Offers' projectUndoable).
        bool undoable( const nlohmann::json& project, const std::vector<nlohmann::json>& projects ) const;

        /// The stored project, or nothing for an empty id or one that names none.
        std::optional<nlohmann::json> projectOf( const std::string& id ) const;

        /**
         * @brief The folder a chat in this project starts with: the project's one folder, at the project's access.
         *
         * @return The folder in UTF-8 and its access, or nothing for a project with none or several, whose chats ask
         *         for one with request_folder.
         */
        static std::optional<std::pair<std::string, Mila::Agents::FolderAccess>> startingFolder( const nlohmann::json& project );

        /// The access the project holds a folder at, or nothing when the folder is not one of the project's.
        static std::optional<Mila::Agents::FolderAccess> projectAccess( const nlohmann::json& project, std::string_view folder );

        /// An event, by the person unless it says otherwise: in the history of each stored project it names, as
        /// `project` or `from`, else in the workspace's activity.
        void recordEvent( nlohmann::json event );

        /// Forget a deleted project or chat in every event that names it (AssistantInterface.md 8, *Deleting*).
        void forgetEverywhere( const std::string& id );

        // ---- The chat and its tools --------------------------------------------------------

        /**
         * @brief Give the session a chat's system prompt, tools, limits, rules and grant (AssistantInterface.md 5.5).
         *
         * Every chat is offered every tool whose work can be done here, under one prompt. Its grant holds the network,
         * for reading pages and searching, and the folder given, if any, to read; the person can give it one later
         * through request_folder, and allows changes there when a call first makes one. Its reply is bounded by the
         * context and the person's Stop. Plan updates run without asking; every other call is asked about, and one that
         * uses the network every time.
         *
         * @param folder The folder the chat was stored with, in UTF-8, or empty. One that cannot be granted -- gone, or
         *               on a computer whose grant holds no folder yet -- is said, and the chat opens without it.
         * @param access How far the chat starts in it: read, unless the folder is its project's and the person chose
         *               *Read and change* there (AssistantInterface.md 10).
         * @param from_project True for a folder the chat holds because its project does, which goes from the chat when
         *                     the project lets it go.
         */
        void useChat( const std::string& folder, Mila::Agents::FolderAccess access = Mila::Agents::FolderAccess::Read,
            bool from_project = false );

        /**
         * @brief The chat's grant, from its folder and whether commands are on: the network, the folder at the access the
         *        person has given, and, once it may be changed and the container is open, what commands run with.
         *
         * Commands get the network only by asking, call by call (ContainedPrograms.md decision 11): the grant holds it and
         * run_command narrows it away from every other call.
         *
         * @throws std::invalid_argument for a folder the grant refuses; the grant is unchanged then.
         */
        void grantChat();

        /**
         * @brief request_folder's work, once the person has allowed it: the folder they chose joins the chat's grant for
         *        the rest of the chat (AssistantInterface.md 5.5).
         *
         * The grant is the app's to widen and is never widened by a call (ToolCalling.md 3.3): the call only asks, and the
         * person's choice in the picker is what widens it.
         */
        std::string grantChosenFolder();

        /**
         * @brief open_place's work: the place the person's words name, or the one they chose among several, opened in
         *        the pane by `turn.went`, and what it shows said to the model (AssistantInterface.md 5.2).
         *
         * Several that fit go to the model to ask about; in a chat that has read from outside they went to the person
         * when the call was asked about, and the model is told none of their names.
         */
        std::string openPlace( const std::string& arguments );

        /// The places open_place's words name, from the store as it stands.
        nlohmann::json placesFor( const std::string& arguments ) const;

        /// find_records' work over every other stored chat; from outside when a match is from a chat that read from outside.
        Mila::Agents::ToolResult findRecordsIn( const std::string& arguments ) const;

        /// The prompt's passages and its tools, from the next response on; the model and conversation stay.
        void usePrompt( const std::filesystem::path& path );

        /// A key given or removed, or commands turned on or off, changes the open chat's prompt and tools from its next
        /// response.
        void refreshTools();

        /// The code that runs a tool, offered whenever the chat's prompt describes it.
        void registerTool( std::string name, Mila::Agents::ToolNeeds needs, Handler handler,
            Mila::Agents::ResultOrigin origin = Mila::Agents::ResultOrigin::Outside );

        /// A tool Mila::Agents provides, which a prompt names rather than describes.
        void useMiloTool( Mila::Agents::Tool tool );

        /**
         * @brief Offer exactly the tools the prompt describes and the session can run here, and the system prompt's
         *        passages about them.
         *
         * Describing a tool nothing runs primes the model to call it, so one with no handler is left out, and so is
         * every tool that needs a folder where the grant can hold none (AssistantInterface.md 5.5, *On a Mac*), and every
         * tool that works in a folder while the chat holds none: it is offered once the chat is given one.
         */
        void offerTools();

        // ---- Models -----------------------------------------------------------------------

        /// The request a load sends: these devices, the context fixed or the longest that fits, and a quantization to
        /// apply as the weights load when one is asked for.
        DeploymentRequest requestFor( std::optional<WeightQuantization> quantization ) const;

        /// The model the settings remember, opened as the session starts; one that will not open is said in the
        /// page's no-model state, where the person chooses another.
        void openRemembered();

        void loadModel( const nlohmann::json& request );

        /**
         * @brief Change model, clearing the conversation, as the person asked: a model's name, checked before the
         *        one resident is released, so a name that will not open leaves it working.
         *
         * A model that does not fit is found only once the card is free; the one it replaced is then loaded again.
         */
        void switchModel( const std::string& name, std::optional<WeightQuantization> quantization );

        /**
         * @brief Open a model on a request, saying how it goes: a load from start to end, and why it failed if it did.
         *
         * @return False, having said why, when nothing could be opened; nothing is resident then, and the session is
         *         held.
         */
        bool open( const std::string& name, std::optional<WeightQuantization> quantization, const DeploymentRequest& request );

        void installModel( const std::string& name );

        void installProgress( std::uint64_t received, std::uint64_t total );

        void removeModel( const std::string& name );

        /**
         * @brief Open the resident model again on another request, the conversation kept: a context sizes the
         *        model's buffers as it is built, so a new one is a new agent over the same session.
         *
         * A request that does not fit puts the model back at the context it had.
         */
        void reopen( const DeploymentRequest& request, std::string_view done );

        // ---- Settings ---------------------------------------------------------------------

        /// One change per request (MilaDesktop.md 4.2); what the session says about it arrives as its own events.
        void changeSetting( const nlohmann::json& request );

        void changeContext( const nlohmann::json& value );

        void changeThinking( const nlohmann::json& value );

        void changeEffort( const nlohmann::json& value );

        void changeSampling( const nlohmann::json& change );

        static std::string_view samplingBounds( std::string_view key ) noexcept;

        void changeSeed( const nlohmann::json& value );

        /// A key is kept and searching offered from the next response; null or empty removes both.
        void changeSearchKey( const nlohmann::json& value );

        void useSearchKey( const std::string& key );

        void changeRunCommands( const nlohmann::json& value );

        /**
         * @brief Opens MXC from the session's own folder and finds the build tools, so a chat with a folder can offer
         *        run_command; what failed is kept for Settings to show.
         */
        void useCommands();

        void useDetail( const std::string& detail );

        // ---- The licence ------------------------------------------------------------------

        /// Whether the person accepted the very text this package carries; a package with another text asks again.
        bool licenseAccepted() const;

        void sendLicense();

        /**
         * @brief The person accepted the licence the page showed them, named by its digest, so an acceptance is
         *        always of the text that was read. Kept for the next launch; the remembered model then opens.
         */
        void acceptLicense( const std::string& sha256 );

        /// Kept for the next launch; a file that cannot be written is said, and the setting still applies now.
        void saveSettings();

        // ---- The template's own variables -------------------------------------------------

        const Mila::Dnn::Conversation::TemplateVariable* templateVariable( std::string_view name ) const;

        /// Whether the resident model reasons, and its template can turn that off.
        bool hasThinkingSwitch() const;

        /// The template's effort variable, where its thinking has levels and is on.
        const Mila::Dnn::Conversation::TemplateVariable* effortVariable() const;

        /**
         * @brief The settings as the resident model's template variables, each only where the template has it.
         *
         * Tools go in the system message where the template can place them there. Llama 3.1's template puts them in
         * the first user message by default, under an instruction to answer with a call, and Llama 3.1 8B then called
         * a tool for every question; with them in the system message and a system prompt saying when to call, it
         * answered 10 of 10 questions that needed none (measured 2026-10-25).
         */
        void applyTemplateVariables();

        // ---- Attachments ------------------------------------------------------------------

        /**
         * @brief Every attachment is refused, by name: no model this build runs takes an image or audio yet
         *        (InstalledModel's isSupportedModality), so session.state lists no modality to accept one for.
         */
        void refuseAttachment( const nlohmann::json& request );

        // ---- Events of the process's own --------------------------------------------------

        void sendState();

        /// The context the conversation takes now, counted by the agent; unknown with no model resident.
        void measureContext();

        /// The folders commands are given to read, for Settings to show; none while commands are off.
        nlohmann::ordered_json programFoldersShown() const;

        nlohmann::json conversationFolder() const;

        /// The tools the person allowed for the open chat, by name.
        nlohmann::ordered_json allowedForChat() const;

        void sendConversations();

        /// `projects.list`: every project's record, the most recently changed first.
        void sendProjects();

        /// `activity.list`: the workspace's and every project's events, newest first, the latest 200 before @p before
        /// (an event's time), or the latest 200 when it is empty.
        void sendActivity( const std::string& before );

        /// What is installed, and, when asked for, what the publisher offers.
        void sendModels( bool online );

        // ---- Writing ----------------------------------------------------------------------

        void sendError( std::string_view text, nlohmann::ordered_json during );

        /// One message, one line. Text the model wrote that is not valid UTF-8 is replaced rather than ending the line.
        void send( std::string_view type, nlohmann::ordered_json fields );

        // ---- Small facts ------------------------------------------------------------------

        /// Where an error happened, for the page to show it there.
        nlohmann::ordered_json workName() const;

        /// A context change rebuilds the model it already holds; anything else that loads, loads.
        std::string_view loadTask() const;

        static std::string_view finishName( GenerateStatus status );

        static std::string_view answerName( ToolAnswer answer );

        /// The first user message, cut at a word under 60 characters (MilaDesktop.md 6.3).
        static std::string titleFor( const std::string& text );

        /// To the millisecond: the store lists conversations by it, and two can change within a second.
        static std::string timestamp();

        static std::string_view roleName( Role role );

        static nlohmann::json turnToJson( const Turn& turn );

        static Turn turnFromJson( const nlohmann::json& stored );

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
        std::expected<MilaLicense, std::string> license_;
        Mila::Deployment::AllowedDevices devices_;

        ResidentAgent model_;

        // The quantization asked for when the resident model loaded, applied again when it reloads.
        std::optional<WeightQuantization> loaded_quantization_;

        // Why the model the settings remember did not open; empty once one does.
        std::string no_model_reason_;

        ConversationStore store_;
        ProjectStore projects_;

        // The workspace's events that belong to no project (AssistantInterface.md 8).
        EventLog activity_;
        std::filesystem::path prompts_;
        std::ostream& out_;
        SearchKeyStore search_key_store_;

        /// Where the key was kept before the credential store held it; read only to move it.
        SearchKeyFile search_key_file_;

        // Every tool the session can run, by name, with what it declares it needs; offered when the prompt describes it.
        std::map<std::string, RegisteredTool> handlers_;

        // The chat's prompt as read, its passages kept by the tools offered; and its tools as described, for the line
        // each call is shown with.
        PromptFile prompt_;
        std::vector<PromptTool> prompt_tools_;
        std::vector<Mila::Agents::Tool> milo_tools_;

        // Set while the person has given a search key; the web_search tool is offered only then.
        std::optional<WebSearch> web_search_;

        // Set while commands are on and MXC opened (ContainedPrograms.md 4.11); one container for every
        // conversation, since it keeps what it checked of each folder between runs. Why it did not open, otherwise.
        std::shared_ptr<const Mila::Agents::ProgramContainer> container_;
        std::string container_problem_;
        Mila::Agents::ProgramFolders programs_;
        std::mutex output_mutex_;

        RequestQueue queue_;

        // Locks the model's memory while Mila is in use and lets it go when it is not (Deployment.md 12.10); what it
        // brings back first is the page's "Getting ready".
        MemoryHolder memory_holder_{ [this]( std::size_t done, std::size_t total )
            {
                send( "progress", { { "task", "restore" }, { "label", "Getting ready" }, { "done", done }, { "total", total } } );
            } };

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

        // The model a load is opening, while it does, so the page names it before it is resident.
        std::string loading_;

        // The tokens the conversation takes in the context: counted between turns, said by each round within one.
        std::optional<dim_t> context_used_;

        /// The model an install is bringing in, as a person reads it: its display name when the publisher's list gave one.
        std::string install_name_;

        /// Each published model's display name from the list last sent, so an install can say what it is installing.
        std::map<std::string, std::string, std::less<>> published_display_names_;
        std::uint64_t install_total_{ 0 };
        int install_percent_{ -1 };

        // The folder the chat's grant holds, in UTF-8, once it has one (AssistantInterface.md 5.5).
        std::optional<std::string> conversation_folder_;

        // How far: read until the person allows a change in it, for this chat only.
        Mila::Agents::FolderAccess conversation_folder_access_{ Mila::Agents::FolderAccess::Read };

        // Whether the chat holds its folder because its project does (AssistantInterface.md 10).
        bool conversation_folder_from_project_{ false };

        // The folder the person chose in answer to request_folder, until its call takes it. Under approval_mutex_.
        std::optional<std::string> chosen_folder_;

        // The place the person chose in answer to open_place, until its call takes it. Under approval_mutex_.
        std::optional<nlohmann::json> chosen_place_;

        // Whether a grant can hold a folder on this computer: true on Windows and macOS, false where the grant has no
        // file checks (ToolCalling.md 3.4), when no tool that needs one is offered. Found by asking for one.
        bool folders_enforced_ = []
            {
                try
                {
                    Mila::Agents::ToolGrant probe( { { std::filesystem::temp_directory_path(), Mila::Agents::FolderAccess::Read } }, false );

                    return true;
                }
                catch ( const std::invalid_argument& )
                {
                    return false;
                }
            }();
        TaskPlan plan_;

        // The conversation the session's history belongs to; empty until its first turn.
        std::string conversation_id_;
        std::string conversation_title_;
        std::string conversation_created_;

        // The project the chat is in, or empty for none.
        std::string conversation_project_;
        nlohmann::json conversation_messages_ = nlohmann::json::array();
        nlohmann::json conversation_links_ = nlohmann::json::array();

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
