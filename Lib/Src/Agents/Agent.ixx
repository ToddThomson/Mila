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

    /**
     * @brief The time one response may take, which stops the response when it runs out.
     *
     * A thread waits on the deadline, since a call that runs or a prefill emits nothing the response could check the
     * clock at. While the person is asked the clock is paused, and the deadline moves by the time they took.
     */
    class ResponseClock
    {
    public:

        ResponseClock( std::optional<std::chrono::milliseconds> limit, std::stop_source halt )
        {
            if ( !limit )
            {
                return;
            }

            deadline_ = std::chrono::steady_clock::now() + *limit;
            watcher_ = std::jthread( [this, halt]( std::stop_token finished ) { watch( finished, halt ); } );
        }

        ResponseClock( const ResponseClock& ) = delete;
        ResponseClock& operator=( const ResponseClock& ) = delete;

        void pause()
        {
            if ( !watcher_.joinable() )
            {
                return;
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            paused_at_ = std::chrono::steady_clock::now();
            paused_ = true;
            changed_.notify_all();
        }

        void resume()
        {
            if ( !watcher_.joinable() )
            {
                return;
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            deadline_ += std::chrono::steady_clock::now() - paused_at_;
            paused_ = false;
            changed_.notify_all();
        }

        [[nodiscard]] bool expired() const noexcept
        {
            return expired_.load();
        }

    private:

        void watch( std::stop_token finished, std::stop_source halt )
        {
            std::unique_lock<std::mutex> lock( mutex_ );

            while ( !finished.stop_requested() )
            {
                if ( paused_ )
                {
                    changed_.wait( lock, finished, [this] { return !paused_; } );
                    continue;
                }

                const auto deadline = deadline_;

                if ( changed_.wait_until( lock, finished, deadline, [this, deadline] { return paused_ || deadline_ != deadline; } ) )
                {
                    continue;
                }

                if ( !finished.stop_requested() )
                {
                    expired_ = true;
                    halt.request_stop();
                }

                return;
            }
        }

        std::mutex mutex_;
        std::condition_variable_any changed_;
        std::chrono::steady_clock::time_point deadline_{};
        std::chrono::steady_clock::time_point paused_at_{};
        bool paused_{ false };
        std::atomic<bool> expired_{ false };

        // Last, so it is joined before the members it reads are destroyed.
        std::jthread watcher_;
    };

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
     * Device and precision are the program's, fixed at compile time; Agent is the pair every published model runs.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class BasicAgent
    {
    public:

        using Handle = ModelHandle<TDeviceType, TPrecision>;
        using EventSink = std::function<void( const AgentEvent& )>;

        /**
         * @brief Whether a name would open as an agent, loading nothing: the reason it would not, or nothing when it
         *        would. What create refuses before it reads a file, so a program can ask before it releases a model.
         */
        [[nodiscard]] static std::optional<OpenRefusal> check(
            std::string_view name, const DeploymentRequest& request = {}, const ModelStore& store = ModelStore{} )
        {
            const std::optional<InstalledModel> installed = InstalledModel::find( name, store );

            if ( installed && !installed->capabilities().instruct )
            {
                return baseModelRefusal( installed->name() );
            }

            return Handle::check( name, request, store );
        }

        /**
         * @brief Open an installed model on the best plan for the request, ready to converse in the session.
         *
         * The session is moved from only once the model has opened: a create that throws leaves the caller's session
         * as it was, conversation included.
         *
         * @throws OpenRefusedError       for every refusal check gives, before anything is loaded.
         * @throws DeploymentRefusedError as Handle::open.
         */
        static BasicAgent create( std::string_view name, const DeploymentRequest& request = {}, Session&& session = {},
            const ModelStore& store = ModelStore{} )
        {
            if ( auto refusal = check( name, request, store ) )
            {
                throw OpenRefusedError( std::move( *refusal ) );
            }

            Handle handle = Handle::open( name, request, store );

            return BasicAgent( std::move( handle ), std::move( session ) );
        }

        /**
         * @brief An agent over a model already opened.
         *
         * The session's template variables the model's template does not accept are dropped from it, and named by
         * droppedTemplateVariables.
         *
         * @throws OpenRefusedError for a base model: it has no chat template to converse in.
         */
        explicit BasicAgent( Handle handle, Session session = {} )
            : handle_( std::move( handle ) ), session_( std::move( session ) )
        {
            if ( handle_.protocol() == nullptr )
            {
                throw OpenRefusedError( baseModelRefusal( handle_.name() ) );
            }

            dropUnacceptedTemplateVariables();
        }

        BasicAgent( BasicAgent&& ) noexcept = default;
        BasicAgent& operator=( BasicAgent&& ) noexcept = default;

        /**
         * @brief Give the session back, the model freed before it returns, so the next model is planned against a
         *        device that no longer holds this one. Nothing is left to call.
         */
        [[nodiscard]] Session release() &&
        {
            Session session = std::move( session_ );

            {
                const Handle released = std::move( handle_ );
            }

            return session;
        }

        /// What the program gave the agent, and the conversation: change it between responses.
        [[nodiscard]] Session& session() noexcept
        {
            return session_;
        }

        [[nodiscard]] const Session& session() const noexcept
        {
            return session_;
        }

        /// The template variables the session asked for that this model's template does not accept, dropped from it
        /// when the agent was created.
        [[nodiscard]] std::span<const std::string> droppedTemplateVariables() const noexcept
        {
            return dropped_template_variables_;
        }

        /**
         * @brief Set one of the chat template's own variables in the session, such as its thinking switch, checked
         *        against this model's template at once.
         *
         * A variable not set takes the template's default. The variables are the model's and differ between
         * models: what one accepts is on `model().protocol()->templateVariables()`.
         *
         * @throws std::invalid_argument for a name or value the template does not accept.
         */
        BasicAgent& withTemplateVariable( std::string name, std::string value ) &
        {
            setTemplateVariable( std::move( name ), std::move( value ) );

            return *this;
        }

        BasicAgent&& withTemplateVariable( std::string name, std::string value ) &&
        {
            setTemplateVariable( std::move( name ), std::move( value ) );

            return std::move( *this );
        }

        /**
         * @brief Set every one of the chat template's variables at once; a variable not named takes its default.
         *
         * @throws std::invalid_argument for a name or value the template does not accept, leaving the variables
         *         as they were.
         */
        BasicAgent& withTemplateVariables( std::map<std::string, std::string> variables ) &
        {
            setTemplateVariables( std::move( variables ) );

            return *this;
        }

        BasicAgent&& withTemplateVariables( std::map<std::string, std::string> variables ) &&
        {
            setTemplateVariables( std::move( variables ) );

            return std::move( *this );
        }

        /**
         * @brief The prompt a response to this message would begin from, the system prompt and tools included.
         *
         * Nothing joins the conversation; what it measures is what the next response would prefill.
         *
         * @throws std::invalid_argument for a template variable the session asks for that the template does not
         *         accept.
         */
        [[nodiscard]] Conversation::Prompt prompt( std::string_view message = {} ) const
        {
            const auto conversation = session_.conversation();
            std::vector<Conversation::Turn> turns( conversation.begin(), conversation.end() );
            turns.push_back( { Conversation::Role::User, std::string( message ) } );

            return render( turns );
        }

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
        Response respond( std::string_view message, const EventSink& on_event = {}, std::stop_token stop = {} )
        {
            const auto emit = [&]( const AgentEvent& event )
                {
                    if ( on_event )
                    {
                        on_event( event );
                    }
                };

            Response response;

            // Set on the session since the agent was created, a variable its template refuses is refused before the
            // message joins.
            Conversation::validateTemplateVariables(
                handle_.protocol()->templateVariables(), session_.templateVariables() );

            std::vector<Conversation::Turn>& history = session_.turns();
            const AgentLimits& limits = session_.limits();

            history.push_back( { Conversation::Role::User, std::string( message ) } );
            continued_.reset();

            RoundPrompt next = nextRound();
            Conversation::Prompt& prompt = next.prompt;
            std::vector<std::int32_t>& tokens = next.tokens;

            // One stop for everything the response runs: the caller's, or its own time running out.
            std::stop_source halt;
            const std::stop_callback forward_stop( stop, [&halt] { halt.request_stop(); } );
            ResponseClock clock( limits.time, halt );
            const std::stop_token halted = halt.get_token();

            CompactionRun compaction{ message, halted, response };

            // A message too long for the context even alone is refused below, and costs the conversation nothing.
            if ( compactionIsDue( tokens, compaction ) && messageFitsAlone( message )
                && compactNow( compaction, emit ) )
            {
                next = nextRound();
            }

            if ( !fitsContext( tokens ) && compaction.gave_up && !halted.stop_requested() && hasTurnsToDrop() )
            {
                dropOldestUnread( message, emit );
                ++response.compactions;
                next = nextRound();
            }

            if ( !fitsContext( tokens ) )
            {
                // Never answered, so it never joined: the conversation stays one the context holds.
                if ( history.size() == 1 && isSummaryTurn( history.front() ) )
                {
                    takeMessageFromSummaryTurn( message );
                } else {
                    history.pop_back();
                }

                response.status = GenerateStatus::ContextOverflow;

                return response;
            }

            GenerateParams params;
            params.sampling = sampling();

            dim_t generated = 0;

            for ( int round = 0;; ++round )
            {
                const std::optional<dim_t> budget = roundBudget( tokens, generated );

                if ( budget && *budget <= 0 )
                {
                    // Every reply token is spent, and the last round's calls already have their results.
                    response.status = GenerateStatus::MaxNewTokensReached;

                    return response;
                }

                params.max_new_tokens = budget.transform( []( dim_t tokens_allowed )
                    {
                        return static_cast<int>( std::min<dim_t>( tokens_allowed, std::numeric_limits<int>::max() ) );
                    } );

                AgentEvent began{ AgentEvent::Kind::RoundBegan };
                began.budget = budget;
                emit( began );
                ++response.rounds;

                // Read against the prompt it continues, which may leave the reply inside the reasoning channel.
                Conversation::ReplyReader reader = handle_.protocol()->readReply( prompt );

                const auto forward = [&]( std::span<const Conversation::ReplyEvent> events )
                    {
                        for ( const auto& event : events )
                        {
                            forwardReplyEvent( event, response, emit );
                        }
                    };

                std::vector<std::int32_t> reply_tokens;
                std::string reply_text;

                const GenerateStatus status = handle_.model().generate(
                    tokens,
                    [&]( std::int32_t token )
                    {
                        const std::string piece = handle_.tokenizer().decode( std::span<const std::int32_t>( &token, 1 ) );

                        ++generated;
                        reply_tokens.push_back( token );
                        reply_text += piece;
                        emit( { AgentEvent::Kind::Token, piece } );
                        forward( reader.feed( piece ) );
                    },
                    params,
                    halted );

                // The round's end settles what the reader still held, and reads a call left open as written.
                forward( reader.finish() );

                const Conversation::Reply& reply = reader.reply();

                response.status = status;
                response.time_limit_reached = clock.expired();

                AgentEvent ended{ AgentEvent::Kind::RoundEnded };
                ended.status = status;
                ended.reply = &reply;
                emit( ended );

                for ( const auto& span : reply.malformed_tool_calls )
                {
                    emit( { AgentEvent::Kind::MalformedToolCall, span } );
                }

                const bool runs_calls = !reply.tool_calls.empty() && !session_.tools().empty()
                    && status == GenerateStatus::Success && !halted.stop_requested();

                if ( !runs_calls )
                {
                    history.push_back( { Conversation::Role::Assistant, reply.content, {}, {}, reply.reasoning } );

                    return response;
                }

                history.push_back(
                    { Conversation::Role::Assistant, reply.content, reply.tool_calls, {}, reply.reasoning } );

                if ( limits.rounds && round + 1 >= *limits.rounds )
                {
                    // Run now, their results would reach no round; the next response can make the calls again.
                    for ( const auto& call : reply.tool_calls )
                    {
                        answerNotRun( call, emit );
                    }

                    response.round_limit_reached = true;

                    return response;
                }

                if ( closesItsCalls( reply_text ) )
                {
                    RoundPrompt as_generated{ { prompt.text + reply_text, false }, tokens };
                    as_generated.tokens.insert( as_generated.tokens.end(), reply_tokens.begin(), reply_tokens.end() );
                    continued_ = std::move( as_generated );
                }

                for ( std::size_t index = 0; index < reply.tool_calls.size(); ++index )
                {
                    if ( !answerCall( reply.tool_calls[ index ], emit, halted, clock, compaction ) )
                    {
                        for ( std::size_t rest = index + 1; rest < reply.tool_calls.size(); ++rest )
                        {
                            history.push_back( { Conversation::Role::Tool,
                                std::string( stoppedResult( clock, false ) ), {}, reply.tool_calls[ rest ].name } );
                        }

                        response.status = GenerateStatus::ClientCancelled;
                        response.time_limit_reached = clock.expired();

                        return response;
                    }
                }

                next = nextRound();
                continued_.reset();

                if ( compactionIsDue( tokens, compaction ) && compactNow( compaction, emit ) )
                {
                    next = nextRound();
                }

                if ( !fitsContext( tokens ) )
                {
                    response.status = GenerateStatus::ContextOverflow;

                    return response;
                }
            }
        }

        /**
         * @brief Compact the conversation now, whatever the trigger, under the session's policy. Between responses.
         *
         * @param on_event Compaction's events; may be empty.
         * @param stop     Stops a summary being written, leaving the conversation as it was.
         * @throws std::logic_error when the session has no policy, which is what says how a summary is written.
         */
        CompactionOutcome compact( const EventSink& on_event = {}, std::stop_token stop = {} )
        {
            if ( !session_.compaction() )
            {
                throw std::logic_error( "a session with no compaction policy cannot compact" );
            }

            const auto emit = [&]( const AgentEvent& event )
                {
                    if ( on_event )
                    {
                        on_event( event );
                    }
                };

            continued_.reset();

            return compactConversation( std::nullopt, emit, stop );
        }

        /// The plan the model was loaded on.
        [[nodiscard]] const DeploymentPlan& plan() const noexcept
        {
            return handle_.plan();
        }

        /// The model, and from it the typed model and its components.
        [[nodiscard]] Handle& model() noexcept
        {
            return handle_;
        }

        [[nodiscard]] const Handle& model() const noexcept
        {
            return handle_;
        }

        /**
         * @brief Whether the model reasons before it answers, which keys the publisher's sampling.
         *
         * `enable_thinking` is the switch both templates that have one name it by; a model with a reasoning channel
         * and no switch always reasons.
         */
        [[nodiscard]] bool thinks() const
        {
            return thinksWith( session_.templateVariables() );
        }

        /**
         * @brief The sampling the next response runs at: each setting the session asks for, over the publisher's for
         *        the thinking in effect, over the sampler's neutral values.
         */
        [[nodiscard]] SamplingParams sampling() const
        {
            return samplingWith( session_.templateVariables() );
        }

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

        [[nodiscard]] bool thinksWith( const std::map<std::string, std::string>& asked ) const
        {
            if ( !handle_.capabilities().thinking )
            {
                return false;
            }

            const auto variables = handle_.protocol()->templateVariables();

            if ( std::ranges::find( variables, "enable_thinking", &Conversation::TemplateVariable::name ) == variables.end() )
            {
                return true;
            }

            return Conversation::resolveTemplateVariable( variables, asked, "enable_thinking" ) == "true";
        }

        [[nodiscard]] SamplingParams samplingWith( const std::map<std::string, std::string>& variables ) const
        {
            SamplingParams sampling;

            // The family's published settings where the manifest carries none, as every application opens at.
            const std::vector<SamplingRecommendation> recommendations = recommendedSampling( handle_.installed().stored().record );

            if ( const auto* recommended = matchingSampling( recommendations, thinksWith( variables ) ) )
            {
                sampling.temperature = recommended->temperature.value_or( sampling.temperature );
                sampling.top_k = recommended->top_k.value_or( sampling.top_k );
                sampling.top_p = recommended->top_p.value_or( sampling.top_p );
                sampling.presence_penalty = recommended->presence_penalty.value_or( sampling.presence_penalty );
            }

            const SamplingOverride& asked = session_.sampling();

            sampling.temperature = asked.temperature.value_or( sampling.temperature );
            sampling.top_k = asked.top_k.value_or( sampling.top_k );
            sampling.top_p = asked.top_p.value_or( sampling.top_p );
            sampling.presence_penalty = asked.presence_penalty.value_or( sampling.presence_penalty );

            return sampling;
        }

        static OpenRefusal baseModelRefusal( std::string_view name )
        {
            return OpenRefusal{ OpenRefusal::Reason::BaseModel, std::string( name ),
                std::format( "'{}' is a base model: it has no chat template to converse in", name ) };
        }

        void setTemplateVariable( std::string name, std::string value )
        {
            Conversation::validateTemplateVariables(
                handle_.protocol()->templateVariables(), { { name, value } } );

            session_.templateVariables()[ std::move( name ) ] = std::move( value );
        }

        void setTemplateVariables( std::map<std::string, std::string> variables )
        {
            Conversation::validateTemplateVariables( handle_.protocol()->templateVariables(), variables );

            session_.templateVariables() = std::move( variables );
        }

        /// Each variable the session asks for is checked alone, so one the template refuses takes no other with it.
        void dropUnacceptedTemplateVariables()
        {
            const auto accepted = handle_.protocol()->templateVariables();
            std::map<std::string, std::string>& asked = session_.templateVariables();

            for ( auto variable = asked.begin(); variable != asked.end(); )
            {
                try
                {
                    Conversation::validateTemplateVariables( accepted, { { variable->first, variable->second } } );
                    ++variable;
                }
                catch ( const std::invalid_argument& )
                {
                    dropped_template_variables_.push_back( variable->first );
                    variable = asked.erase( variable );
                }
            }
        }

        /// The most tokens a round may generate: the reply tokens the response has left, within the context left.
        std::optional<dim_t> roundBudget( const std::vector<std::int32_t>& tokens, dim_t generated ) const
        {
            const AgentLimits& limits = session_.limits();

            if ( !limits.reply_tokens )
            {
                return std::nullopt;
            }

            const dim_t context_left = handle_.plan().contextLength() - static_cast<dim_t>( tokens.size() );

            return std::min( *limits.reply_tokens - generated, context_left );
        }

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
        RoundPrompt nextRound() const
        {
            const std::span<const Conversation::Turn> history = session_.conversation();

            if ( !continued_ )
            {
                Conversation::Prompt prompt = render( history );
                std::vector<std::int32_t> tokens = encode( prompt );

                return { std::move( prompt ), std::move( tokens ) };
            }

            std::size_t first_result = history.size();

            while ( first_result > 0 && history[ first_result - 1 ].role == Conversation::Role::Tool )
            {
                --first_result;
            }

            const Conversation::Prompt after = handle_.protocol()->continueWithToolResults(
                history.subspan( first_result ), promptOptions() );

            RoundPrompt next = *continued_;
            next.prompt.text += after.text;
            next.prompt.reply_begins_in_reasoning = after.reply_begins_in_reasoning;

            const std::vector<std::int32_t> added = encode( after );
            next.tokens.insert( next.tokens.end(), added.begin(), added.end() );

            return next;
        }

        /**
         * @brief Whether a reply's last call is closed as the template closes it, so the reply can stand in the
         *        next prompt as written. One left open is rendered again: the model never reads a call it did not
         *        finish followed by its result.
         */
        bool closesItsCalls( std::string_view reply ) const
        {
            const std::string& close = handle_.protocol()->replyGrammar().tool_call_close;

            if ( close.empty() )
            {
                // A call with no close runs to the end of the reply.
                return true;
            }

            const std::size_t end = reply.find_last_not_of( " \t\r\n" );

            return end != std::string_view::npos && reply.substr( 0, end + 1 ).ends_with( close );
        }

        Conversation::PromptOptions promptOptions() const
        {
            Conversation::PromptOptions options;
            options.template_variables = session_.templateVariables();

            if ( !session_.tools().empty() )
            {
                options.tools_json = toolsJson();
            }

            return options;
        }

        Conversation::Prompt render( std::span<const Conversation::Turn> history ) const
        {
            return render( history, promptOptions() );
        }

        Conversation::Prompt render( std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const
        {
            std::vector<Conversation::Turn> turns;
            turns.reserve( history.size() + 1 );

            if ( !session_.systemPrompt().empty() )
            {
                turns.push_back( { Conversation::Role::System, session_.systemPrompt() } );
            }

            turns.insert( turns.end(), history.begin(), history.end() );

            return handle_.protocol()->formatPrompt( turns, options );
        }

        /// The tools as OpenAI-style function schemas, which every template reads and declares its own way.
        std::string toolsJson() const
        {
            nlohmann::ordered_json array = nlohmann::ordered_json::array();

            for ( const auto& tool : session_.tools() )
            {
                nlohmann::ordered_json function = nlohmann::ordered_json::object();
                function[ "name" ] = tool.name;
                function[ "description" ] = tool.description;
                function[ "parameters" ] = nlohmann::ordered_json::parse( tool.parameters );

                nlohmann::ordered_json entry = nlohmann::ordered_json::object();
                entry[ "type" ] = "function";
                entry[ "function" ] = std::move( function );
                array.push_back( std::move( entry ) );
            }

            return array.dump();
        }

        std::vector<std::int32_t> encode( const Conversation::Prompt& prompt ) const
        {
            return handle_.tokenizer().encode( prompt.text );
        }

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
        [[nodiscard]] dim_t compactionTrigger( const CompactionPolicy& policy ) const
        {
            const dim_t limit = handle_.plan().contextLength();

            return std::min( policy.trigger_tokens.value_or( limit ), limit - policy.summary_tokens );
        }

        [[nodiscard]] bool compactionIsDue( const std::vector<std::int32_t>& tokens, const CompactionRun& run ) const
        {
            const std::optional<CompactionPolicy>& policy = session_.compaction();

            if ( !policy || run.gave_up || policy->summary_tokens >= handle_.plan().contextLength() )
            {
                return false;
            }

            return static_cast<dim_t>( tokens.size() ) >= compactionTrigger( *policy );
        }

        [[nodiscard]] bool messageFitsAlone( std::string_view message ) const
        {
            const std::array<Conversation::Turn, 1> alone{ Conversation::Turn{ Conversation::Role::User, std::string( message ) } };

            return fitsContext( encode( render( alone ) ) );
        }

        /// @return True when the conversation changed. Whatever else happened, this response tries no more.
        template<typename TEmit>
        bool compactNow( CompactionRun& run, const TEmit& emit )
        {
            const CompactionOutcome outcome = compactConversation( run.message, emit, run.stop );

            if ( outcome == CompactionOutcome::Compacted || outcome == CompactionOutcome::ReasoningCleared )
            {
                ++run.response.compactions;

                return true;
            }

            run.gave_up = true;

            return false;
        }

        [[nodiscard]] static bool isSummaryTurn( const Conversation::Turn& turn ) noexcept
        {
            return turn.role == Conversation::Role::User && turn.content.starts_with( kSummaryHeading );
        }

        [[nodiscard]] static std::string messageSection( std::string_view message )
        {
            return std::format( "\n\n{}\n\n{}", kMessageHeading, message );
        }

        /// A message that turned out not to fit even beside its summary leaves the conversation as a dropped one does.
        void takeMessageFromSummaryTurn( std::string_view message )
        {
            std::string& content = session_.turns().front().content;
            content.resize( content.size() - messageSection( message ).size() );
        }

        /// Where the turn in progress begins: the message not yet answered, or the latest reply with calls and its
        /// results. Those stay as they stand.
        [[nodiscard]] std::size_t turnInProgress( bool message_is_pending ) const
        {
            const std::span<const Conversation::Turn> history = session_.conversation();

            if ( message_is_pending )
            {
                return history.size() - 1;
            }

            std::size_t first = history.size();

            while ( first > 0 && history[ first - 1 ].role == Conversation::Role::Tool )
            {
                --first;
            }

            if ( first > 0 && history[ first - 1 ].role == Conversation::Role::Assistant && !history[ first - 1 ].tool_calls.empty() )
            {
                return first - 1;
            }

            return first;
        }

        /// The prompt options a summary is written under: the session's, with the policy's variables over them.
        [[nodiscard]] Conversation::PromptOptions summaryOptions( const CompactionPolicy& policy ) const
        {
            Conversation::PromptOptions options = promptOptions();

            for ( const auto& [name, value] : policy.template_variables )
            {
                try
                {
                    Conversation::validateTemplateVariables( handle_.protocol()->templateVariables(), { { name, value } } );
                    options.template_variables[ name ] = value;
                }
                catch ( const std::invalid_argument& )
                {
                    // Another model's switch, such as a thinking switch on a model that always reasons.
                }
            }

            return options;
        }

        /// Drop the oldest reply with its results, or the oldest turn; an earlier summary goes last.
        static void dropOldestTurns( std::vector<Conversation::Turn>& turns )
        {
            const std::size_t first = turns.size() > 1 && isSummaryTurn( turns.front() ) ? 1 : 0;

            turns.erase( turns.begin() + static_cast<std::ptrdiff_t>( first ) );

            while ( turns.size() > first && turns[ first ].role == Conversation::Role::Tool )
            {
                turns.erase( turns.begin() + static_cast<std::ptrdiff_t>( first ) );
            }
        }

        /**
         * @brief Cut the one turn left so a summary can read its beginning. A reply that ran to the context limit would
         *        otherwise leave nothing a summary could read, and the conversation could never go on.
         *
         * @return False when no text of it would be left.
         */
        [[nodiscard]] bool cutToLeaveRoom( Conversation::Turn& turn, dim_t tokens_over ) const
        {
            // The note below is itself tokens, and decoding a cut can round either way.
            static constexpr dim_t kMarginTokens = 32;

            const std::vector<std::int32_t> tokens = handle_.tokenizer().encode( turn.content );
            const dim_t kept = static_cast<dim_t>( tokens.size() ) - tokens_over - kMarginTokens;

            if ( kept <= 0 )
            {
                return false;
            }

            turn.content = handle_.tokenizer().decode(
                std::span<const std::int32_t>( tokens.data(), static_cast<std::size_t>( kept ) ) );

            dropIncompleteLastCharacter( turn.content );
            turn.content += "\n[The rest of this turn was too long to read.]";

            return true;
        }

        /// @param text Holds something other than space.
        [[nodiscard]] static std::string_view withoutOuterSpace( std::string_view text ) noexcept
        {
            static constexpr std::string_view kSpace = " \t\r\n";

            text.remove_prefix( text.find_first_not_of( kSpace ) );
            text.remove_suffix( text.size() - text.find_last_not_of( kSpace ) - 1 );

            return text;
        }

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
            std::string& failure, bool& worth_another_try )
        {
            worth_another_try = false;

            GenerateParams params;
            params.sampling = samplingWith( variables );
            params.max_new_tokens = static_cast<int>( std::min<dim_t>( policy.summary_tokens, std::numeric_limits<int>::max() ) );

            Conversation::ReplyReader reader = handle_.protocol()->readReply( prompt );

            const GenerateStatus status = handle_.model().generate(
                tokens,
                [&]( std::int32_t token )
                {
                    // What the reader returns is the summary's own text and reasoning, which are no reply of the model's.
                    (void)reader.feed( handle_.tokenizer().decode( std::span<const std::int32_t>( &token, 1 ) ) );
                },
                params,
                stop );

            (void)reader.finish();

            const Conversation::Reply& reply = reader.reply();

            if ( status == GenerateStatus::MaxNewTokensReached )
            {
                failure = std::format( "the summary did not finish in {} tokens", policy.summary_tokens );
            } else if ( status != GenerateStatus::Success ) {
                failure = "the summary was stopped";
            } else if ( !reply.tool_calls.empty() || !reply.malformed_tool_calls.empty() ) {
                failure = "the model called a tool instead of writing the summary";
                worth_another_try = true;
            } else if ( reply.content.find_first_not_of( " \t\r\n" ) == std::string::npos ) {
                failure = "the model wrote no summary";
            } else if ( earlier.contains( withoutOuterSpace( reply.content ) ) ) {
                failure = "the summary added nothing to the earlier one";
                worth_another_try = true;
            }

            if ( !failure.empty() )
            {
                return std::nullopt;
            }

            return reply.content;
        }

        template<typename TEmit>
        CompactionOutcome compactConversation( std::optional<std::string_view> message, const TEmit& emit, std::stop_token stop )
        {
            const CompactionPolicy& policy = *session_.compaction();
            std::vector<Conversation::Turn>& history = session_.turns();

            const bool message_is_pending = message && !history.empty() && history.back().role == Conversation::Role::User
                && history.back().content == *message;
            const std::size_t turn_in_progress = turnInProgress( message_is_pending );

            if ( turn_in_progress == 0 || ( turn_in_progress == 1 && isSummaryTurn( history.front() ) ) )
            {
                return CompactionOutcome::NothingToReplace;
            }

            const dim_t tokens_before = static_cast<dim_t>( encode( render( history ) ).size() );

            AgentEvent began{ AgentEvent::Kind::CompactionBegan };
            began.tokens_before = tokens_before;
            emit( began );

            const auto clearReasoning = [&]( std::vector<Conversation::Turn>& turns )
                {
                    const std::size_t end = std::min( turn_in_progress, turns.size() );

                    for ( std::size_t index = 0; index < end; ++index )
                    {
                        turns[ index ].reasoning.clear();
                    }
                };

            // Where a template renders past reasoning, clearing it changes the prompt from its first turn that has
            // any. So it is done alone only when that is enough; otherwise the summary reads what the model holds.
            {
                std::vector<Conversation::Turn> cleared = history;
                clearReasoning( cleared );

                const dim_t tokens_cleared = static_cast<dim_t>( encode( render( cleared ) ).size() );

                if ( tokens_cleared < tokens_before && tokens_cleared < compactionTrigger( policy ) )
                {
                    std::vector<Conversation::Turn> replaced = std::move( cleared );
                    replaced.swap( history );
                    continued_.reset();

                    AgentEvent compacted{ AgentEvent::Kind::Compacted };
                    compacted.replaced_turns = std::span<const Conversation::Turn>( replaced ).first( turn_in_progress );
                    compacted.tokens_before = tokens_before;
                    compacted.tokens_after = tokens_cleared;
                    emit( compacted );

                    return CompactionOutcome::ReasoningCleared;
                }
            }

            std::vector<Conversation::Turn> read( history.begin(), history.end() - ( message_is_pending ? 1 : 0 ) );
            const std::size_t turns_to_replace = turn_in_progress;
            std::size_t dropped = 0;
            std::string failure;
            std::optional<std::string> summary;

            if ( policy.summarizer )
            {
                try
                {
                    read.resize( turns_to_replace );
                    summary = policy.summarizer( read, stop );

                    if ( !summary )
                    {
                        failure = "the program's summarizer wrote nothing";
                    }
                }
                catch ( const std::exception& thrown )
                {
                    failure = std::format( "the program's summarizer failed: {}", thrown.what() );
                }
            } else {
                const Conversation::PromptOptions options = summaryOptions( policy );
                const Conversation::Turn instruction{ Conversation::Role::User,
                    policy.instruction.empty() ? defaultCompactionInstruction( policy.summary_tokens ) : policy.instruction };

                Conversation::Prompt prompt;
                std::vector<std::int32_t> tokens;

                const auto fits = [&]()
                    {
                        std::vector<Conversation::Turn> asked = read;
                        asked.push_back( instruction );
                        prompt = render( asked, options );
                        tokens = encode( prompt );

                        return static_cast<dim_t>( tokens.size() ) + policy.summary_tokens <= handle_.plan().contextLength();
                    };

                // The turn in progress stays word for word anyway, then reasoning goes, then the oldest turns: what
                // is dropped reaches no summary, and still reaches the application.
                if ( !fits() )
                {
                    read.resize( turns_to_replace );

                    if ( !fits() )
                    {
                        clearReasoning( read );

                        while ( !read.empty() && !fits() )
                        {
                            if ( read.size() > 1 )
                            {
                                const std::size_t held = read.size();
                                dropOldestTurns( read );
                                dropped += held - read.size();
                            } else if ( !cutToLeaveRoom( read.front(), tokens.size() + policy.summary_tokens - handle_.plan().contextLength() ) ) {
                                read.clear();
                            }
                        }
                    }
                }

                if ( read.empty() )
                {
                    failure = "there is no room to write a summary";
                } else {
                    const std::string_view earlier = isSummaryTurn( history.front() ) ? std::string_view( history.front().content ) : std::string_view{};
                    bool worth_another_try = false;
                    summary = writeSummary( policy, prompt, tokens, options.template_variables, earlier, stop, failure, worth_another_try );

                    // Keeping the tools spares a re-read of the whole conversation, but the instruction not to call
                    // one does not always hold, and a model may copy an earlier summary. Once it has done either, the
                    // re-read is the smaller cost.
                    if ( worth_another_try )
                    {
                        Conversation::PromptOptions no_tools = options;
                        no_tools.tools_json.clear();

                        std::vector<Conversation::Turn> asked = read;
                        asked.push_back( instruction );
                        prompt = render( asked, no_tools );
                        tokens = encode( prompt );
                        failure.clear();
                        summary = writeSummary( policy, prompt, tokens, no_tools.template_variables, earlier, stop, failure, worth_another_try );
                    }
                }
            }

            if ( !summary )
            {
                AgentEvent failed{ AgentEvent::Kind::CompactionFailed, failure };
                emit( failed );

                return stop.stop_requested() ? CompactionOutcome::Stopped : CompactionOutcome::Failed;
            }

            std::string text = std::format( "{}\n\n{}", kSummaryHeading, *summary );

            if ( !policy.kept.empty() )
            {
                text += std::format( "\n\n{}", kKeptHeading );

                for ( const std::string& entry : policy.kept )
                {
                    text += std::format( "\n\n{}", entry );
                }
            }

            if ( message )
            {
                text += messageSection( *message );
            }

            // The vector the response holds a reference to stays the same object.
            std::vector<Conversation::Turn> replaced;
            replaced.swap( history );

            history.push_back( { Conversation::Role::User, std::move( text ) } );
            history.insert( history.end(),
                replaced.begin() + static_cast<std::ptrdiff_t>( turns_to_replace + ( message_is_pending ? 1 : 0 ) ), replaced.end() );

            continued_.reset();

            AgentEvent compacted{ AgentEvent::Kind::Compacted, *summary };
            compacted.replaced_turns = std::span<const Conversation::Turn>( replaced ).first( turns_to_replace );
            compacted.dropped_turns = dropped;
            compacted.tokens_before = tokens_before;
            compacted.tokens_after = static_cast<dim_t>( encode( render( history ) ).size() );
            emit( compacted );

            return CompactionOutcome::Compacted;
        }

        /// Anything but the message pending and an earlier summary.
        [[nodiscard]] bool hasTurnsToDrop() const
        {
            const std::span<const Conversation::Turn> history = session_.conversation();

            return history.size() > ( isSummaryTurn( history.front() ) ? 2u : 1u );
        }

        /**
         * @brief The last resort before a response, when no summary could be written and the message does not fit:
         *        drop the oldest turns unread, so a session that cannot summarize is not refused for good.
         *
         * Goes below the trigger and not just below the limit, so the next message does not pay for another failed
         * summary. The oldest go first and an earlier summary stays, and the conversation's first turn tells the model
         * that something is gone. The message pending is the conversation's last turn.
         */
        template<typename TEmit>
        void dropOldestUnread( std::string_view message, const TEmit& emit )
        {
            const CompactionPolicy& policy = *session_.compaction();
            std::vector<Conversation::Turn>& history = session_.turns();

            const dim_t tokens_before = static_cast<dim_t>( encode( render( history ) ).size() );
            const std::size_t first_droppable = isSummaryTurn( history.front() ) ? 1 : 0;
            const std::size_t pending = history.size() - 1;

            std::string opening = first_droppable == 1 ? history.front().content : std::string( kSummaryHeading );

            if ( !opening.contains( kTurnsDroppedNote ) )
            {
                opening += std::format( "\n\n{}", kTurnsDroppedNote );
            }

            if ( first_droppable == 0 && !policy.kept.empty() )
            {
                opening += std::format( "\n\n{}", kKeptHeading );

                for ( const std::string& entry : policy.kept )
                {
                    opening += std::format( "\n\n{}", entry );
                }
            }

            // The opening joins the first turn kept, or the message as a summary's turn does, so that a template that
            // wants turns to alternate is given what it always is.
            const auto keepingFrom = [&]( std::size_t first )
                {
                    std::vector<Conversation::Turn> remaining( history.begin() + static_cast<std::ptrdiff_t>( first ), history.end() );

                    for ( Conversation::Turn& turn : remaining )
                    {
                        turn.reasoning.clear();
                    }

                    remaining.front().content = first == pending
                        ? opening + messageSection( message )
                        : std::format( "{}\n\n{}", opening, remaining.front().content );

                    return remaining;
                };

            std::size_t first = first_droppable;
            std::vector<Conversation::Turn> remaining = keepingFrom( first );

            while ( first < pending
                && ( history[ first ].role != Conversation::Role::User
                    || static_cast<dim_t>( encode( render( remaining ) ).size() ) >= compactionTrigger( policy ) ) )
            {
                ++first;

                if ( history[ first ].role == Conversation::Role::User )
                {
                    remaining = keepingFrom( first );
                }
            }

            std::vector<Conversation::Turn> replaced = std::move( remaining );
            replaced.swap( history );

            continued_.reset();

            AgentEvent compacted{ AgentEvent::Kind::Compacted };
            compacted.replaced_turns = std::span<const Conversation::Turn>( replaced ).subspan( first_droppable, first - first_droppable );
            compacted.dropped_turns = first - first_droppable;
            compacted.tokens_before = tokens_before;
            compacted.tokens_after = static_cast<dim_t>( encode( render( history ) ).size() );
            emit( compacted );
        }

        /// A prompt that leaves the model no position to answer in is not generated from.
        bool fitsContext( const std::vector<std::int32_t>& tokens ) const noexcept
        {
            return std::cmp_less( tokens.size(), handle_.plan().contextLength() );
        }

        template<typename TEmit>
        static void forwardReplyEvent( const Conversation::ReplyEvent& event, Response& response, const TEmit& emit )
        {
            using Kind = Conversation::ReplyEvent::Kind;

            switch ( event.kind )
            {
                case Kind::ReasoningOpened:
                    emit( { AgentEvent::Kind::ReasoningOpened } );
                    break;

                case Kind::Reasoning:
                    response.reasoning += event.text;
                    emit( { AgentEvent::Kind::Reasoning, event.text } );
                    break;

                case Kind::ReasoningClosed:
                    emit( { AgentEvent::Kind::ReasoningClosed } );
                    break;

                case Kind::Answer:
                    response.text += event.text;
                    emit( { AgentEvent::Kind::Answer, event.text } );
                    break;

                case Kind::ToolCallOpened:
                    emit( { AgentEvent::Kind::ToolCallOpened } );
                    break;

                case Kind::ToolCallClosed:
                    // A call is reported once it has been answered.
                    break;
            }
        }

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
            CompactionRun& compaction )
        {
            Session::Asker ask;

            if ( session_.asker() )
            {
                ask = [this, &clock]( const ToolQuestion& question )
                    {
                        clock.pause();
                        const ToolAnswer answer = session_.asker()( question );
                        clock.resume();

                        return answer;
                    };
            }

            ToolOutcome outcome = session_.gate().answer( call.name, call.arguments, ask, halted, session_.limits() );

            if ( outcome.stopped )
            {
                // Every call in the conversation keeps a result, so the next prompt renders.
                session_.turns().push_back(
                    { Conversation::Role::Tool, std::string( stoppedResult( clock, outcome.ran ) ), {}, call.name } );

                return false;
            }

            if ( outcome.ran && outcome.result_tokens )
            {
                cutToResultTokens( outcome );
            }

            addResult( call, std::move( outcome ), emit, compaction );

            return true;
        }

        /// A call of the last round a response may run, which is answered without running it.
        template<typename TEmit>
        void answerNotRun( const Conversation::ToolCall& call, const TEmit& emit )
        {
            const std::string result = std::format(
                "This tool call was not run: the response reached its limit of {} rounds.", *session_.limits().rounds );

            AgentEvent called{ AgentEvent::Kind::ToolCalled, result };
            called.call = &call;
            called.is_error = true;
            called.decision = ToolDecision::NotRun;
            emit( called );

            session_.turns().push_back( { Conversation::Role::Tool, result, {}, call.name } );
        }

        /// Cut a result to the most tokens its tool may give the model, and tell the model so.
        void cutToResultTokens( ToolOutcome& outcome ) const
        {
            const std::vector<std::int32_t> tokens = handle_.tokenizer().encode( outcome.result );
            const dim_t total = static_cast<dim_t>( tokens.size() );
            const dim_t kept = *outcome.result_tokens;

            if ( total <= kept )
            {
                return;
            }

            outcome.result = handle_.tokenizer().decode(
                std::span<const std::int32_t>( tokens.data(), static_cast<std::size_t>( kept ) ) );

            dropIncompleteLastCharacter( outcome.result );
            outcome.result += std::format(
                "\n[The result was cut to its first {} of {} tokens, the most this tool may give.]", kept, total );
        }

        static std::string_view stoppedResult( const ResponseClock& clock, bool while_running ) noexcept
        {
            if ( clock.expired() )
            {
                return while_running ? kOutOfTimeWhileRunningResult : kOutOfTimeResult;
            }

            return while_running ? kStoppedWhileRunningResult : kStoppedResult;
        }

        /**
         * @brief Put a call's result in the conversation, cut to the context left, and report it.
         *
         * A result the context cannot hold first has the conversation compacted around it, under a policy. If it
         * still does not fit it is cut to its first tokens that fit, and the model told it was cut and by how much;
         * one with no room at all is refused, and the model told so. Never left to overflow the next prompt, which
         * would end the response with no word of why.
         */
        template<typename TEmit>
        void addResult( const Conversation::ToolCall& call, ToolOutcome outcome, const TEmit& emit, CompactionRun& compaction )
        {
            std::string& result = outcome.result;
            bool is_error = outcome.is_error;
            std::vector<Conversation::Turn>& history = session_.turns();

            history.push_back( { Conversation::Role::Tool, result, {}, call.name } );

            std::vector<std::int32_t> prompt_tokens = nextRound().tokens;

            if ( !fitsContext( prompt_tokens ) && compactionIsDue( prompt_tokens, compaction ) && compactNow( compaction, emit ) )
            {
                prompt_tokens = nextRound().tokens;
            }

            if ( !fitsContext( prompt_tokens ) )
            {
                const std::vector<std::int32_t> result_tokens = handle_.tokenizer().encode( result );
                const dim_t total = static_cast<dim_t>( result_tokens.size() );
                dim_t kept = total - tokensOverContext( prompt_tokens );

                result = std::format(
                    "Error: the result of '{}' was {} tokens, and the context has no room left for it.", call.name, total );
                is_error = true;

                while ( kept > 0 )
                {
                    std::string cut = handle_.tokenizer().decode(
                        std::span<const std::int32_t>( result_tokens.data(), static_cast<std::size_t>( kept ) ) );

                    dropIncompleteLastCharacter( cut );
                    cut += std::format(
                        "\n[The result was cut to its first {} of {} tokens to fit the context.]", kept, total );

                    history.back().content = cut;
                    prompt_tokens = nextRound().tokens;

                    if ( fitsContext( prompt_tokens ) )
                    {
                        result = std::move( cut );
                        is_error = false;

                        break;
                    }

                    kept -= tokensOverContext( prompt_tokens );
                }

                history.back().content = result;
            }

            AgentEvent called{ AgentEvent::Kind::ToolCalled, history.back().content };
            called.call = &call;
            called.is_error = is_error;
            called.decision = outcome.decision;
            called.duration = outcome.duration;
            called.returned_bytes = outcome.returned_bytes;
            emit( called );
        }

        /// How many tokens the prompt must lose to leave the model a position to answer in.
        dim_t tokensOverContext( const std::vector<std::int32_t>& tokens ) const noexcept
        {
            return static_cast<dim_t>( tokens.size() ) - handle_.plan().contextLength() + 1;
        }

        /// A cut between tokens can split a character; its leading bytes alone are not text a template can write.
        static void dropIncompleteLastCharacter( std::string& text )
        {
            std::size_t start = text.size();
            int continuation = 0;

            while ( start > 0 && continuation < 3 && ( static_cast<unsigned char>( text[ start - 1 ] ) & 0xC0 ) == 0x80 )
            {
                --start;
                ++continuation;
            }

            if ( start == 0 )
            {
                return;
            }

            const auto lead = static_cast<unsigned char>( text[ start - 1 ] );
            const int length = ( lead & 0x80 ) == 0 ? 1 : ( lead & 0xE0 ) == 0xC0 ? 2 : ( lead & 0xF0 ) == 0xE0 ? 3 : 4;

            if ( continuation + 1 < length )
            {
                text.erase( start - 1 );
            }
        }

        Handle handle_;
        Session session_;
        std::vector<std::string> dropped_template_variables_;

        // While a round's calls are answered: its prompt and its reply as generated, which the results continue.
        std::optional<RoundPrompt> continued_;
    };

    /// The agent every published model runs as: CUDA, computing in BF16 whatever its weights' format.
    export using Agent = BasicAgent<DeviceType::Cuda, TensorDataType::BF16>;
}
