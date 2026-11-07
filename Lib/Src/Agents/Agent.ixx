/**
 * @file Agent.ixx
 * @brief An installed model at work in a program: the conversation, its tools and the loop that runs their calls.
 *
 * See Specifications/Direction.md section 4 and Specifications/ToolCalling.md.
 */

export module Agents.Agent;

import std;
import nlohmann.json;
import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.GenerateParams;
import Dnn.SamplingParams;
import Dnn.GenerateStatus;
import Dnn.TensorTypes;
import Dnn.Models.Conversation.Protocol;
import Data.BpeTokenizer;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.InstalledModel;
import Distribution.ModelStore;
import Distribution.SamplingRecommendation;
import Distribution.FamilySampling;
import Deployment.ModelHandle;
import Agents.Tool;
import Agents.ToolQuestion;
import Agents.ToolAnswer;
import Agents.ToolDecision;
import Agents.ToolOutcome;
import Agents.ToolGate;
import Agents.AgentEvent;
import Agents.AgentLimits;
import Agents.CompactionOutcome;
import Agents.CompactionPolicy;
import Agents.Response;
import Agents.Session;
import Agents.SamplingOverride;
import Deployment.OpenRefusal;
import Deployment.OpenRefusedError;

namespace Mila::Agents
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;

    /// The time one response may take, which stops the response when it runs out.
    class ResponseClock;

    /**
     * @brief A model named by its store name, and a session: its conversation, and the calls the model makes in it.
     *
     * Through an agent the application never runs a call itself: its session holds the tools and a way to ask the
     * person, and the agent parses each call, asks, runs it and gives the model its result. Family-blind: everything
     * a model does differently comes from its handle. A response is one virtual call into the model per round, and
     * runs within the session's limits (AgentLimits).
     *
     * An agent ends with its model; its session does not. Another model, or the same one at another context, is
     * another agent created from the session `release` gives back (Direction.md 4.3).
     *
     * Names no device and no precision: the handle chooses both when the model opens (ModelHandle.md 10.7).
     */
    export class Agent
    {
    public:

        using Handle = ModelHandle;
        using EventSink = std::function<void( const AgentEvent& )>;

        /**
         * @brief Whether a name would open as an agent, loading nothing: the reason it would not, or nothing when it
         *        would. What create refuses before it reads a file, so a program can ask before it releases a model.
         */
        [[nodiscard]] static std::optional<OpenRefusal> check(
            std::string_view name, const DeploymentRequest& request = {}, const ModelStore& store = ModelStore{} );

        /**
         * @brief Open an installed model on the best plan for the request, ready to converse in the session.
         *
         * The session is moved from only once the model has opened: a create that throws leaves the caller's session
         * as it was, conversation included.
         *
         * @throws OpenRefusedError       for every refusal check gives, before anything is loaded.
         * @throws DeploymentRefusedError as Handle::open.
         */
        static Agent create( std::string_view name, const DeploymentRequest& request = {}, Session&& session = {},
            const ModelStore& store = ModelStore{} );

        /**
         * @brief An agent over a model already opened.
         *
         * The session's template variables the model's template does not accept are dropped from it, and named by
         * droppedTemplateVariables.
         *
         * @throws OpenRefusedError for a base model: it has no chat template to converse in.
         */
        explicit Agent( Handle handle, Session session = {} );

        Agent( Agent&& ) noexcept = default;
        Agent& operator=( Agent&& ) noexcept = default;

        /**
         * @brief Give the session back, the model freed before it returns, so the next model is planned against a
         *        device that no longer holds this one. Nothing is left to call.
         */
        [[nodiscard]] Session release() &&;

        /// What the program gave the agent, and the conversation: change it between responses.
        [[nodiscard]] Session& session() noexcept;

        [[nodiscard]] const Session& session() const noexcept;

        /// The template variables the session asked for that this model's template does not accept, dropped from it
        /// when the agent was created.
        [[nodiscard]] std::span<const std::string> droppedTemplateVariables() const noexcept;

        /**
         * @brief Set one of the chat template's own variables in the session, such as its thinking switch, checked
         *        against this model's template at once.
         *
         * A variable not set takes the template's default. The variables are the model's and differ between
         * models: what one accepts is on `model().protocol()->templateVariables()`.
         *
         * @throws std::invalid_argument for a name or value the template does not accept.
         */
        Agent& withTemplateVariable( std::string name, std::string value ) &;

        Agent&& withTemplateVariable( std::string name, std::string value ) &&;

        /**
         * @brief Set every one of the chat template's variables at once; a variable not named takes its default.
         *
         * @throws std::invalid_argument for a name or value the template does not accept, leaving the variables
         *         as they were.
         */
        Agent& withTemplateVariables( std::map<std::string, std::string> variables ) &;

        Agent&& withTemplateVariables( std::map<std::string, std::string> variables ) &&;

        /**
         * @brief The prompt a response to this message would begin from, the system prompt and tools included.
         *
         * Nothing joins the conversation; what it measures is what the next response would prefill.
         *
         * @throws std::invalid_argument for a template variable the session asks for that the template does not
         *         accept.
         */
        [[nodiscard]] Conversation::Prompt prompt( std::string_view message = {} ) const;

        /**
         * @brief Answer one message, running every call the model makes, until it gives its answer or a limit ends
         *        the response.
         *
         * Blocks on the caller's thread, reporting what happens through on_event as it happens. The message, each
         * round's reply and each call's result join the conversation. A round cut short -- by the context, the reply
         * tokens, the time or stop -- runs none of its calls, and every call the conversation holds keeps a result.
         *
         * The first round reads the conversation as the template renders it. Each round after it continues from
         * the reply as the model generated it, followed by the results as the template frames them, so the model
         * prefills only the results; a reply that left its last call open is rendered again instead.
         *
         * Under the session's compaction policy the conversation is compacted before the response, between its
         * rounds, and before a result would be cut, reported by events. A summary counts against the response's time
         * and not its reply tokens. After one fails, the response does not try again.
         *
         * @param message  What the user says.
         * @param on_event Every event in order; may be empty.
         * @param stop     Stops the reply; a call waiting for the person's answer does not run, and a call running is
         *                 asked to stop.
         */
        Response respond( std::string_view message, const EventSink& on_event = {}, std::stop_token stop = {} );

        /**
         * @brief Compact the conversation now, whatever the trigger, under the session's policy. Between responses.
         *
         * @param on_event Compaction's events; may be empty.
         * @param stop     Stops a summary being written, leaving the conversation as it was.
         * @throws std::logic_error when the session has no policy, which is what says how a summary is written.
         */
        CompactionOutcome compact( const EventSink& on_event = {}, std::stop_token stop = {} );

        /// The plan the model was loaded on.
        [[nodiscard]] const DeploymentPlan& plan() const noexcept;

        /// The model, and from it the typed model and its components.
        [[nodiscard]] Handle& model() noexcept;

        [[nodiscard]] const Handle& model() const noexcept;

        /**
         * @brief Whether the model reasons before it answers, which keys the publisher's sampling.
         *
         * `enable_thinking` is the switch both templates that have one name it by; a model with a reasoning channel
         * and no switch always reasons.
         */
        [[nodiscard]] bool thinks() const;

        /**
         * @brief The sampling the next response runs at: each setting the session asks for, over the publisher's for
         *        the thinking in effect, over the sampler's neutral values.
         */
        [[nodiscard]] SamplingParams sampling() const;

    private:

        static constexpr std::string_view kStoppedResult = "The user stopped the reply before this tool call ran.";
        static constexpr std::string_view kStoppedWhileRunningResult =
            "The user stopped the reply while this tool call ran, so it may not have finished.";
        static constexpr std::string_view kOutOfTimeResult = "The response ran out of time before this tool call ran.";
        static constexpr std::string_view kOutOfTimeWhileRunningResult =
            "The response ran out of time while this tool call ran, so it may not have finished.";

        // How the turn that replaces a history begins, which is also how a later compaction knows it for one.
        static constexpr std::string_view kSummaryHeading = "A summary of the conversation so far:";
        static constexpr std::string_view kKeptHeading = "Keep to these, word for word:";
        static constexpr std::string_view kMessageHeading = "The message being answered:";
        static constexpr std::string_view kTurnsDroppedNote =
            "Earlier turns of this conversation are gone, because no summary of them could be written.";

        [[nodiscard]] bool thinksWith( const std::map<std::string, std::string>& asked ) const;

        [[nodiscard]] SamplingParams samplingWith( const std::map<std::string, std::string>& variables ) const;

        static OpenRefusal baseModelRefusal( std::string_view name );

        void setTemplateVariable( std::string name, std::string value );

        void setTemplateVariables( std::map<std::string, std::string> variables );

        /// Each variable the session asks for is checked alone, so one the template refuses takes no other with it.
        void dropUnacceptedTemplateVariables();

        /// The most tokens a round may generate: the reply tokens the response has left, within the context left.
        std::optional<dim_t> roundBudget( const std::vector<std::int32_t>& tokens, dim_t generated ) const;

        /// A prompt and its tokens, which within a response are not always the encoding of its text.
        struct RoundPrompt
        {
            Conversation::Prompt prompt;
            std::vector<std::int32_t> tokens;
        };

        /**
         * @brief The prompt the next round begins from.
         *
         * After a reply that made calls, the reply as generated followed by the results answered so far, as the
         * template frames them after it: the model holds every token before the results, so nothing it read or
         * wrote is prefilled again. Otherwise the conversation as the template renders it.
         */
        RoundPrompt nextRound() const;

        /**
         * @brief Whether a reply's last call is closed as the template closes it, so the reply can stand in the
         *        next prompt as written. One left open is rendered again: the model never reads a call it did not
         *        finish followed by its result.
         */
        bool closesItsCalls( std::string_view reply ) const;

        Conversation::PromptOptions promptOptions() const;

        Conversation::Prompt render( std::span<const Conversation::Turn> history ) const;

        Conversation::Prompt render( std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const;

        /// The tools as OpenAI-style function schemas, which every template reads and declares its own way.
        std::string toolsJson() const;

        std::vector<std::int32_t> encode( const Conversation::Prompt& prompt ) const;

        // ---- Compaction (Specifications/Agents/Compaction.md) --------------------------------

        /// One response's compaction: what it answers, what stops it, and whether it has failed once already.
        struct CompactionRun
        {
            std::string_view message;
            std::stop_token stop;
            Response& response;
            bool gave_up{ false };
        };

        /// The prompt length that sets compaction off here. A summary always has its room below the limit.
        [[nodiscard]] dim_t compactionTrigger( const CompactionPolicy& policy ) const;

        [[nodiscard]] bool compactionIsDue( const std::vector<std::int32_t>& tokens, const CompactionRun& run ) const;

        [[nodiscard]] bool messageFitsAlone( std::string_view message ) const;

        /// @return True when the conversation changed. Whatever else happened, this response tries no more.
        template<typename TEmit>
        bool compactNow( CompactionRun& run, const TEmit& emit );

        [[nodiscard]] static bool isSummaryTurn( const Conversation::Turn& turn ) noexcept;

        [[nodiscard]] static std::string messageSection( std::string_view message );

        /// A message that turned out not to fit even beside its summary leaves the conversation as a dropped one does.
        void takeMessageFromSummaryTurn( std::string_view message );

        /// Where the turn in progress begins: the message not yet answered, or the latest reply with calls and its
        /// results. Those stay as they stand.
        [[nodiscard]] std::size_t turnInProgress( bool message_is_pending ) const;

        /// The prompt options a summary is written under: the session's, with the policy's variables over them.
        [[nodiscard]] Conversation::PromptOptions summaryOptions( const CompactionPolicy& policy ) const;

        /// Drop the oldest reply with its results, or the oldest turn; an earlier summary goes last.
        static void dropOldestTurns( std::vector<Conversation::Turn>& turns );

        /**
         * @brief Cut the one turn left so a summary can read its beginning. A reply that ran to the context limit would
         *        otherwise leave nothing a summary could read, and the conversation could never go on.
         *
         * @return False when no text of it would be left.
         */
        [[nodiscard]] bool cutToLeaveRoom( Conversation::Turn& turn, dim_t tokens_over ) const;

        /// @param text Holds something other than space.
        [[nodiscard]] static std::string_view withoutOuterSpace( std::string_view text ) noexcept;

        /**
         * @brief Ask the model for the summary. Generates in place of a round, reporting nothing of it.
         *
         * @param earlier The turn holding the summary before this one, if any. A summary it already holds says nothing of
         *        what came since, and accepting it would lose that without a word.
         *
         * @return The summary, or nothing with failure set to why, and worth_another_try when a second attempt may differ.
         */
        [[nodiscard]] std::optional<std::string> writeSummary( const CompactionPolicy& policy,
            const Conversation::Prompt& prompt, const std::vector<std::int32_t>& tokens,
            const std::map<std::string, std::string>& variables, std::string_view earlier, std::stop_token stop,
            std::string& failure, bool& worth_another_try );

        template<typename TEmit>
        CompactionOutcome compactConversation( std::optional<std::string_view> message, const TEmit& emit, std::stop_token stop );

        /// Anything but the message pending and an earlier summary.
        [[nodiscard]] bool hasTurnsToDrop() const;

        /**
         * @brief The last resort before a response, when no summary could be written and the message does not fit:
         *        drop the oldest turns unread, so a session that cannot summarize is not refused for good.
         *
         * Goes below the trigger and not just below the limit, so the next message does not pay for another failed
         * summary. The oldest go first and an earlier summary stays, and the conversation's first turn tells the model
         * that something is gone. The message pending is the conversation's last turn.
         */
        template<typename TEmit>
        void dropOldestUnread( std::string_view message, const TEmit& emit );

        /// A prompt that leaves the model no position to answer in is not generated from.
        bool fitsContext( const std::vector<std::int32_t>& tokens ) const noexcept;

        template<typename TEmit>
        static void forwardReplyEvent( const Conversation::ReplyEvent& event, Response& response, const TEmit& emit );

        /**
         * @brief Pass a call through the gate, and put its result in the conversation.
         *
         * Models call tools they invent -- Llama 3.1 8B asked for `get_capital` when offered only `get_weather` --
         * and leave out arguments a tool requires: the gate answers both without asking (ToolGate).
         *
         * The time the person takes to answer is not the response's: the clock is paused while they are asked.
         *
         * @return False when the response was stopped, or ran out of time, before the call ran or while it ran: its
         *         result says so, nothing reports it, and the response ends.
         */
        template<typename TEmit>
        bool answerCall( const Conversation::ToolCall& call, const TEmit& emit, std::stop_token halted, ResponseClock& clock,
            CompactionRun& compaction );

        /// A call of the last round a response may run, which is answered without running it.
        template<typename TEmit>
        void answerNotRun( const Conversation::ToolCall& call, const TEmit& emit );

        /// Cut a result to the most tokens its tool may give the model, and tell the model so.
        void cutToResultTokens( ToolOutcome& outcome ) const;

        static std::string_view stoppedResult( const ResponseClock& clock, bool while_running ) noexcept;

        /**
         * @brief Put a call's result in the conversation, cut to the context left, and report it.
         *
         * A result the context cannot hold first has the conversation compacted around it, under a policy. If it
         * still does not fit it is cut to its first tokens that fit, and the model told it was cut and by how much;
         * one with no room at all is refused, and the model told so. Never left to overflow the next prompt, which
         * would end the response with no word of why.
         */
        template<typename TEmit>
        void addResult( const Conversation::ToolCall& call, ToolOutcome outcome, const TEmit& emit, CompactionRun& compaction );

        /// How many tokens the prompt must lose to leave the model a position to answer in.
        dim_t tokensOverContext( const std::vector<std::int32_t>& tokens ) const noexcept;

        /// A cut between tokens can split a character; its leading bytes alone are not text a template can write.
        static void dropIncompleteLastCharacter( std::string& text );

        Handle handle_;
        Session session_;
        std::vector<std::string> dropped_template_variables_;

        // While a round's calls are answered: its prompt and its reply as generated, which the results continue.
        std::optional<RoundPrompt> continued_;
    };

    /// The agent every published model runs as: CUDA, computing in BF16 whatever its weights' format.
}
