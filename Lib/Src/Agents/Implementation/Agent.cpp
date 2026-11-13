/**
 * @file Agent.cpp
 * @brief The bodies of Agent's members and the clock a response runs against; the interface is Agent.ixx.
 */

module Agents.Agent;

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

    std::optional<OpenRefusal> Agent::check( std::string_view name, const DeploymentRequest& request, const ModelStore& store )
    {
        const std::optional<InstalledModel> installed = InstalledModel::find( name, store );

        if ( installed && !installed->capabilities().instruct )
        {
            return baseModelRefusal( installed->name() );
        }

        return Handle::check( name, request, store );
    }

    Agent Agent::create( std::string_view name, const DeploymentRequest& request, Session&& session, const ModelStore& store )
    {
        if ( auto refusal = check( name, request, store ) )
        {
            throw OpenRefusedError( std::move( *refusal ) );
        }

        Handle handle = Handle::open( name, request, store );

        return Agent( std::move( handle ), std::move( session ) );
    }

    Agent::Agent( Handle handle, Session session )
        : handle_( std::move( handle ) ), session_( std::move( session ) )
    {
        if ( handle_.protocol() == nullptr )
        {
            throw OpenRefusedError( baseModelRefusal( handle_.name() ) );
        }

        dropUnacceptedTemplateVariables();
    }

    Session Agent::release() &&
    {
        Session session = std::move( session_ );

        {
            const Handle released = std::move( handle_ );
        }

        return session;
    }

    Session& Agent::session() noexcept
    {
        return session_;
    }

    const Session& Agent::session() const noexcept
    {
        return session_;
    }

    std::span<const std::string> Agent::droppedTemplateVariables() const noexcept
    {
        return dropped_template_variables_;
    }

    Agent& Agent::withTemplateVariable( std::string name, std::string value ) &
    {
        setTemplateVariable( std::move( name ), std::move( value ) );

        return *this;
    }

    Agent&& Agent::withTemplateVariable( std::string name, std::string value ) &&
    {
        setTemplateVariable( std::move( name ), std::move( value ) );

        return std::move( *this );
    }

    Agent& Agent::withTemplateVariables( std::map<std::string, std::string> variables ) &
    {
        setTemplateVariables( std::move( variables ) );

        return *this;
    }

    Agent&& Agent::withTemplateVariables( std::map<std::string, std::string> variables ) &&
    {
        setTemplateVariables( std::move( variables ) );

        return std::move( *this );
    }

    Conversation::Prompt Agent::prompt( std::string_view message ) const
    {
        const auto conversation = session_.conversation();
        std::vector<Conversation::Turn> turns( conversation.begin(), conversation.end() );
        turns.push_back( { Conversation::Role::User, std::string( message ) } );

        return render( turns );
    }

    Response Agent::respond( std::string_view message, const EventSink& on_event, std::stop_token stop )
    {
        const auto conversation = session_.conversation();
        std::vector<Conversation::Turn> before( conversation.begin(), conversation.end() );

        try
        {
            return respondToMessage( message, on_event, stop );
        }
        catch ( ... )
        {
            session_.turns() = std::move( before );
            continued_.reset();

            throw;
        }
    }

    Response Agent::respondToMessage( std::string_view message, const EventSink& on_event, std::stop_token stop )
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

    CompactionOutcome Agent::compact( const EventSink& on_event, std::stop_token stop )
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

    const DeploymentPlan& Agent::plan() const noexcept
    {
        return handle_.plan();
    }

    auto Agent::model() noexcept -> Handle&
    {
        return handle_;
    }

    auto Agent::model() const noexcept -> const Handle&
    {
        return handle_;
    }

    bool Agent::thinks() const
    {
        return thinksWith( session_.templateVariables() );
    }

    SamplingParams Agent::sampling() const
    {
        return samplingWith( session_.templateVariables() );
    }

    bool Agent::thinksWith( const std::map<std::string, std::string>& asked ) const
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

    SamplingParams Agent::samplingWith( const std::map<std::string, std::string>& variables ) const
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

    OpenRefusal Agent::baseModelRefusal( std::string_view name )
    {
        return OpenRefusal{ OpenRefusal::Reason::BaseModel, std::string( name ),
            std::format( "'{}' is a base model: it has no chat template to converse in", name ) };
    }

    void Agent::setTemplateVariable( std::string name, std::string value )
    {
        Conversation::validateTemplateVariables(
            handle_.protocol()->templateVariables(), { { name, value } } );

        session_.templateVariables()[ std::move( name ) ] = std::move( value );
    }

    void Agent::setTemplateVariables( std::map<std::string, std::string> variables )
    {
        Conversation::validateTemplateVariables( handle_.protocol()->templateVariables(), variables );

        session_.templateVariables() = std::move( variables );
    }

    void Agent::dropUnacceptedTemplateVariables()
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

    std::optional<dim_t> Agent::roundBudget( const std::vector<std::int32_t>& tokens, dim_t generated ) const
    {
        const AgentLimits& limits = session_.limits();

        if ( !limits.reply_tokens )
        {
            return std::nullopt;
        }

        const dim_t context_left = handle_.plan().contextLength() - static_cast<dim_t>( tokens.size() );

        return std::min( *limits.reply_tokens - generated, context_left );
    }

    auto Agent::nextRound() const -> RoundPrompt
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

    bool Agent::closesItsCalls( std::string_view reply ) const
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

    Conversation::PromptOptions Agent::promptOptions() const
    {
        Conversation::PromptOptions options;
        options.template_variables = session_.templateVariables();

        if ( !session_.tools().empty() )
        {
            options.tools_json = toolsJson();
        }

        return options;
    }

    Conversation::Prompt Agent::render( std::span<const Conversation::Turn> history ) const
    {
        return render( history, promptOptions() );
    }

    Conversation::Prompt Agent::render( std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const
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

    std::string Agent::toolsJson() const
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

    std::vector<std::int32_t> Agent::encode( const Conversation::Prompt& prompt ) const
    {
        return handle_.tokenizer().encode( prompt.text );
    }

    dim_t Agent::compactionTrigger( const CompactionPolicy& policy ) const
    {
        const dim_t limit = handle_.plan().contextLength();

        return std::min( policy.trigger_tokens.value_or( limit ), limit - policy.summary_tokens );
    }

    bool Agent::compactionIsDue( const std::vector<std::int32_t>& tokens, const CompactionRun& run ) const
    {
        const std::optional<CompactionPolicy>& policy = session_.compaction();

        if ( !policy || run.gave_up || policy->summary_tokens >= handle_.plan().contextLength() )
        {
            return false;
        }

        return static_cast<dim_t>( tokens.size() ) >= compactionTrigger( *policy );
    }

    bool Agent::messageFitsAlone( std::string_view message ) const
    {
        const std::array<Conversation::Turn, 1> alone{ Conversation::Turn{ Conversation::Role::User, std::string( message ) } };

        return fitsContext( encode( render( alone ) ) );
    }

    template<typename TEmit>
    bool Agent::compactNow( CompactionRun& run, const TEmit& emit )
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

    bool Agent::isSummaryTurn( const Conversation::Turn& turn ) noexcept
    {
        return turn.role == Conversation::Role::User && turn.content.starts_with( kSummaryHeading );
    }

    std::string Agent::messageSection( std::string_view message )
    {
        return std::format( "\n\n{}\n\n{}", kMessageHeading, message );
    }

    void Agent::takeMessageFromSummaryTurn( std::string_view message )
    {
        std::string& content = session_.turns().front().content;
        content.resize( content.size() - messageSection( message ).size() );
    }

    std::size_t Agent::turnInProgress( bool message_is_pending ) const
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

    Conversation::PromptOptions Agent::summaryOptions( const CompactionPolicy& policy ) const
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

    void Agent::dropOldestTurns( std::vector<Conversation::Turn>& turns )
    {
        const std::size_t first = turns.size() > 1 && isSummaryTurn( turns.front() ) ? 1 : 0;

        turns.erase( turns.begin() + static_cast<std::ptrdiff_t>( first ) );

        while ( turns.size() > first && turns[ first ].role == Conversation::Role::Tool )
        {
            turns.erase( turns.begin() + static_cast<std::ptrdiff_t>( first ) );
        }
    }

    bool Agent::cutToLeaveRoom( Conversation::Turn& turn, dim_t tokens_over ) const
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

    std::string_view Agent::withoutOuterSpace( std::string_view text ) noexcept
    {
        static constexpr std::string_view kSpace = " \t\r\n";

        text.remove_prefix( text.find_first_not_of( kSpace ) );
        text.remove_suffix( text.size() - text.find_last_not_of( kSpace ) - 1 );

        return text;
    }

    std::optional<std::string> Agent::writeSummary( const CompactionPolicy& policy, const Conversation::Prompt& prompt, const std::vector<std::int32_t>& tokens, const std::map<std::string, std::string>& variables, std::string_view earlier, std::stop_token stop, std::string& failure, bool& worth_another_try )
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
    CompactionOutcome Agent::compactConversation( std::optional<std::string_view> message, const TEmit& emit, std::stop_token stop )
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

    bool Agent::hasTurnsToDrop() const
    {
        const std::span<const Conversation::Turn> history = session_.conversation();

        return history.size() > ( isSummaryTurn( history.front() ) ? 2u : 1u );
    }

    template<typename TEmit>
    void Agent::dropOldestUnread( std::string_view message, const TEmit& emit )
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

    bool Agent::fitsContext( const std::vector<std::int32_t>& tokens ) const noexcept
    {
        return std::cmp_less( tokens.size(), handle_.plan().contextLength() );
    }

    template<typename TEmit>
    void Agent::forwardReplyEvent( const Conversation::ReplyEvent& event, Response& response, const TEmit& emit )
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

    template<typename TEmit>
    bool Agent::answerCall( const Conversation::ToolCall& call, const TEmit& emit, std::stop_token halted, ResponseClock& clock, CompactionRun& compaction )
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

    template<typename TEmit>
    void Agent::answerNotRun( const Conversation::ToolCall& call, const TEmit& emit )
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

    void Agent::cutToResultTokens( ToolOutcome& outcome ) const
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

    std::string_view Agent::stoppedResult( const ResponseClock& clock, bool while_running ) noexcept
    {
        if ( clock.expired() )
        {
            return while_running ? kOutOfTimeWhileRunningResult : kOutOfTimeResult;
        }

        return while_running ? kStoppedWhileRunningResult : kStoppedResult;
    }

    template<typename TEmit>
    void Agent::addResult( const Conversation::ToolCall& call, ToolOutcome outcome, const TEmit& emit, CompactionRun& compaction )
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

    dim_t Agent::tokensOverContext( const std::vector<std::int32_t>& tokens ) const noexcept
    {
        return static_cast<dim_t>( tokens.size() ) - handle_.plan().contextLength() + 1;
    }

    void Agent::dropIncompleteLastCharacter( std::string& text )
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
}
