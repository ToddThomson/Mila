/**
 * @file Agent.ixx
 * @brief An installed model at work in a program: the conversation, its tools and the loop that runs their calls.
 *
 * See Specifications/Direction.md section 4 and Specifications/ToolCalling.md.
 */

export module AI.Agent;

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
import AI.ModelHandle;
import AI.Tool;
import AI.ToolQuestion;
import AI.ToolAnswer;
import AI.AgentEvent;
import AI.AgentLimits;
import AI.Response;

namespace Mila::AI
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
     * @brief A model named by its store name, holding one conversation and running the calls the model makes.
     *
     * Through an agent the application never runs a call itself: it registers tools and a way to ask the person, and
     * the agent parses each call, asks, runs it and gives the model its result. Family-blind: everything a model
     * does differently comes from its handle. A response is one virtual call into the model per round, and runs
     * within the program's limits (AgentLimits).
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
        using Asker = std::function<ToolAnswer( const ToolQuestion& )>;

        /**
         * @brief Open an installed model on the best plan for the request, ready to converse.
         *
         * @throws std::invalid_argument for a base model, before its weights are read, and as Handle::open.
         * @throws std::runtime_error and DeploymentRefusedError as Handle::open.
         */
        static BasicAgent create(
            std::string_view name, const DeploymentRequest& request = {}, const ModelStore& store = ModelStore{} )
        {
            if ( const auto installed = InstalledModel::find( name, store ); installed && !installed->capabilities().instruct )
            {
                throw std::invalid_argument( baseModelRefusal( installed->name() ) );
            }

            return BasicAgent( Handle::open( name, request, store ) );
        }

        /**
         * @brief An agent over a model already opened.
         *
         * @throws std::invalid_argument for a base model: it has no chat template to converse in.
         */
        explicit BasicAgent( Handle handle )
            : handle_( std::move( handle ) )
        {
            if ( handle_.protocol() == nullptr )
            {
                throw std::invalid_argument( baseModelRefusal( handle_.name() ) );
            }
        }

        BasicAgent( BasicAgent&& ) noexcept = default;
        BasicAgent& operator=( BasicAgent&& ) noexcept = default;

        /**
         * @brief Offer the model a tool.
         *
         * @throws std::invalid_argument for a tool with no name or function, a name already registered, or
         *         parameters that are not a JSON object.
         */
        BasicAgent& withTool( Tool tool ) &
        {
            addTool( std::move( tool ) );

            return *this;
        }

        BasicAgent&& withTool( Tool tool ) &&
        {
            addTool( std::move( tool ) );

            return std::move( *this );
        }

        /**
         * @brief How the person is asked whether a call may run. With none, no one can be asked, so every call of
         *        a registered tool is declined.
         */
        BasicAgent& onAsk( Asker ask ) &
        {
            ask_ = std::move( ask );

            return *this;
        }

        BasicAgent&& onAsk( Asker ask ) &&
        {
            ask_ = std::move( ask );

            return std::move( *this );
        }

        /// The system message every prompt begins with; empty for none.
        BasicAgent& withSystemPrompt( std::string text ) &
        {
            system_prompt_ = std::move( text );

            return *this;
        }

        BasicAgent&& withSystemPrompt( std::string text ) &&
        {
            system_prompt_ = std::move( text );

            return std::move( *this );
        }

        /**
         * @brief Set one of the chat template's own variables, such as its thinking switch.
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

        /// Sample with these settings instead of what the model's publisher recommends.
        BasicAgent& withSampling( SamplingParams sampling ) &
        {
            sampling_ = sampling;

            return *this;
        }

        BasicAgent&& withSampling( SamplingParams sampling ) &&
        {
            sampling_ = sampling;

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
         * @brief Limit how far each response may run. None is set until this is called.
         *
         * @throws std::invalid_argument for a limit set to zero or less.
         */
        BasicAgent& withLimits( AgentLimits limits ) &
        {
            setLimits( limits );

            return *this;
        }

        BasicAgent&& withLimits( AgentLimits limits ) &&
        {
            setLimits( limits );

            return std::move( *this );
        }

        /**
         * @brief The prompt a response to this message would begin from, the system prompt and tools included.
         *
         * Nothing joins the conversation; what it measures is what the next response would prefill.
         */
        [[nodiscard]] Conversation::Prompt prompt( std::string_view message = {} ) const
        {
            std::vector<Conversation::Turn> turns( history_.begin(), history_.end() );
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

            history_.push_back( { Conversation::Role::User, std::string( message ) } );

            Conversation::Prompt prompt = render( history_ );
            std::vector<std::int32_t> tokens = encode( prompt );

            if ( !fitsContext( tokens ) )
            {
                // Never answered, so it never joined: the conversation stays one the context holds.
                history_.pop_back();
                response.status = GenerateStatus::ContextOverflow;

                return response;
            }

            // One stop for everything the response runs: the caller's, or its own time running out.
            std::stop_source halt;
            const std::stop_callback forward_stop( stop, [&halt] { halt.request_stop(); } );
            ResponseClock clock( limits_.time, halt );
            const std::stop_token halted = halt.get_token();

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

                const GenerateStatus status = handle_.model().generate(
                    tokens,
                    [&]( std::int32_t token )
                    {
                        const std::string piece = handle_.tokenizer().decode( std::span<const std::int32_t>( &token, 1 ) );

                        ++generated;
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

                const bool runs_calls = !reply.tool_calls.empty() && !tools_.empty()
                    && status == GenerateStatus::Success && !halted.stop_requested();

                if ( !runs_calls )
                {
                    history_.push_back( { Conversation::Role::Assistant, reply.content, {}, {}, reply.reasoning } );

                    return response;
                }

                history_.push_back(
                    { Conversation::Role::Assistant, reply.content, reply.tool_calls, {}, reply.reasoning } );

                if ( limits_.rounds && round + 1 >= *limits_.rounds )
                {
                    // Run now, their results would reach no round; the next response can make the calls again.
                    for ( const auto& call : reply.tool_calls )
                    {
                        answerNotRun( call, emit );
                    }

                    response.round_limit_reached = true;

                    return response;
                }

                for ( std::size_t index = 0; index < reply.tool_calls.size(); ++index )
                {
                    if ( !answerCall( reply.tool_calls[ index ], emit, halted, clock ) )
                    {
                        for ( std::size_t rest = index + 1; rest < reply.tool_calls.size(); ++rest )
                        {
                            history_.push_back( { Conversation::Role::Tool,
                                std::string( stoppedResult( clock, false ) ), {}, reply.tool_calls[ rest ].name } );
                        }

                        response.status = GenerateStatus::ClientCancelled;
                        response.time_limit_reached = clock.expired();

                        return response;
                    }
                }

                prompt = render( history_ );
                tokens = encode( prompt );

                if ( !fitsContext( tokens ) )
                {
                    response.status = GenerateStatus::ContextOverflow;

                    return response;
                }
            }
        }

        /// Forget the conversation, and every tool allowed for it. Tools and settings stay.
        void newConversation()
        {
            history_.clear();
            allowed_for_conversation_.clear();
        }

        /// Continue a conversation saved earlier, as `conversation()` gave it; no tool is allowed for it yet.
        void openConversation( std::vector<Conversation::Turn> turns )
        {
            history_ = std::move( turns );
            allowed_for_conversation_.clear();
        }

        /// Every message, reply and result so far, without the system prompt.
        [[nodiscard]] std::span<const Conversation::Turn> conversation() const noexcept
        {
            return history_;
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

    private:

        static constexpr std::string_view kDeclinedResult = "The user declined to run this tool call.";
        static constexpr std::string_view kStoppedResult = "The user stopped the reply before this tool call ran.";
        static constexpr std::string_view kStoppedWhileRunningResult =
            "The user stopped the reply while this tool call ran, so it may not have finished.";
        static constexpr std::string_view kOutOfTimeResult = "The response ran out of time before this tool call ran.";
        static constexpr std::string_view kOutOfTimeWhileRunningResult =
            "The response ran out of time while this tool call ran, so it may not have finished.";

        static std::string baseModelRefusal( std::string_view name )
        {
            return std::format( "'{}' is a base model: it has no chat template to converse in", name );
        }

        void addTool( Tool tool )
        {
            if ( tool.name.empty() )
            {
                throw std::invalid_argument( "a tool needs a name" );
            }

            if ( !tool.run )
            {
                throw std::invalid_argument( std::format( "tool '{}' has no function to run it", tool.name ) );
            }

            if ( findTool( tool.name ) != nullptr )
            {
                throw std::invalid_argument( std::format( "a tool named '{}' is already registered", tool.name ) );
            }

            const auto parameters = nlohmann::ordered_json::parse( tool.parameters, nullptr, false );

            if ( parameters.is_discarded() || !parameters.is_object() )
            {
                throw std::invalid_argument( std::format(
                    "tool '{}': its parameters must be a JSON Schema object", tool.name ) );
            }

            tools_.push_back( std::move( tool ) );
        }

        void setTemplateVariable( std::string name, std::string value )
        {
            Conversation::validateTemplateVariables(
                handle_.protocol()->templateVariables(), { { name, value } } );

            template_variables_[ std::move( name ) ] = std::move( value );
        }

        void setTemplateVariables( std::map<std::string, std::string> variables )
        {
            Conversation::validateTemplateVariables( handle_.protocol()->templateVariables(), variables );

            template_variables_ = std::move( variables );
        }

        void setLimits( const AgentLimits& limits )
        {
            if ( limits.rounds && *limits.rounds <= 0 )
            {
                throw std::invalid_argument( std::format( "a response needs at least one round, not {}", *limits.rounds ) );
            }

            if ( limits.time && limits.time->count() <= 0 )
            {
                throw std::invalid_argument( std::format( "a response needs some time, not {}", *limits.time ) );
            }

            if ( limits.reply_tokens && *limits.reply_tokens <= 0 )
            {
                throw std::invalid_argument(
                    std::format( "a response needs at least one reply token, not {}", *limits.reply_tokens ) );
            }

            limits_ = limits;
        }

        /// The most tokens a round may generate: the reply tokens the response has left, within the context left.
        std::optional<dim_t> roundBudget( const std::vector<std::int32_t>& tokens, dim_t generated ) const
        {
            if ( !limits_.reply_tokens )
            {
                return std::nullopt;
            }

            const dim_t context_left = handle_.plan().contextLength() - static_cast<dim_t>( tokens.size() );

            return std::min( *limits_.reply_tokens - generated, context_left );
        }

        const Tool* findTool( std::string_view name ) const noexcept
        {
            const auto tool = std::ranges::find( tools_, name, &Tool::name );

            return tool == tools_.end() ? nullptr : &*tool;
        }

        /**
         * @brief Whether the model reasons before it answers, which keys the publisher's sampling.
         *
         * `enable_thinking` is the switch both templates that have one name it by; a model with a reasoning channel
         * and no switch always reasons.
         */
        bool thinks() const
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

            return Conversation::resolveTemplateVariable( variables, template_variables_, "enable_thinking" ) == "true";
        }

        /// The program's sampling, else the publisher's for the thinking in effect, else the sampler's neutral values.
        SamplingParams sampling() const
        {
            if ( sampling_ )
            {
                return *sampling_;
            }

            SamplingParams sampling;

            if ( const auto* recommended = matchingSampling( handle_.installed().stored().record.sampling, thinks() ) )
            {
                sampling.temperature = recommended->temperature.value_or( sampling.temperature );
                sampling.top_k = recommended->top_k.value_or( sampling.top_k );
                sampling.top_p = recommended->top_p.value_or( sampling.top_p );
                sampling.presence_penalty = recommended->presence_penalty.value_or( sampling.presence_penalty );
            }

            return sampling;
        }

        Conversation::Prompt render( std::span<const Conversation::Turn> history ) const
        {
            Conversation::PromptOptions options;
            options.template_variables = template_variables_;

            if ( !tools_.empty() )
            {
                options.tools_json = toolsJson();
            }

            std::vector<Conversation::Turn> turns;
            turns.reserve( history.size() + 1 );

            if ( !system_prompt_.empty() )
            {
                turns.push_back( { Conversation::Role::System, system_prompt_ } );
            }

            turns.insert( turns.end(), history.begin(), history.end() );

            return handle_.protocol()->formatPrompt( turns, options );
        }

        /// The tools as OpenAI-style function schemas, which every template reads and declares its own way.
        std::string toolsJson() const
        {
            nlohmann::ordered_json array = nlohmann::ordered_json::array();

            for ( const auto& tool : tools_ )
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
         * @brief Ask whether a call may run, run it or decline it, and put its result in the conversation.
         *
         * A call of a tool the model was not offered is never asked about, since nothing will run: the model reads
         * that there is no such tool. Models call tools they invent -- Llama 3.1 8B asked for `get_capital` when
         * offered only `get_weather`.
         *
         * The time the person takes to answer is not the response's: the clock is paused while they are asked.
         *
         * @return False when the response was stopped, or ran out of time, before the call ran or while it ran: its
         *         result says so, nothing reports it, and the response ends.
         */
        template<typename TEmit>
        bool answerCall( const Conversation::ToolCall& call, const TEmit& emit, std::stop_token halted, ResponseClock& clock )
        {
            if ( halted.stop_requested() )
            {
                // Every call in the conversation keeps a result, so the next prompt renders.
                history_.push_back( { Conversation::Role::Tool, std::string( stoppedResult( clock, false ) ), {}, call.name } );

                return false;
            }

            const Tool* tool = findTool( call.name );

            ToolAnswer answer = ToolAnswer::Once;

            if ( tool != nullptr && !allowed_for_conversation_.contains( call.name ) )
            {
                if ( ask_ )
                {
                    clock.pause();
                    answer = ask_( ToolQuestion{ tool->name, tool->description, call.arguments } );
                    clock.resume();
                }
                else
                {
                    answer = ToolAnswer::Deny;
                }
            }

            if ( halted.stop_requested() )
            {
                history_.push_back( { Conversation::Role::Tool, std::string( stoppedResult( clock, false ) ), {}, call.name } );

                return false;
            }

            if ( answer == ToolAnswer::ForConversation )
            {
                allowed_for_conversation_.insert( call.name );
            }

            std::string result;
            bool is_error = true;

            if ( tool == nullptr )
            {
                result = notOfferedResult( call.name );
            }
            else if ( answer == ToolAnswer::Deny )
            {
                result = std::string( kDeclinedResult );
            }
            else
            {
                try
                {
                    result = tool->run( call.arguments, halted );
                    is_error = false;
                }
                catch ( const std::exception& error )
                {
                    result = std::format( "Error: tool '{}' failed: {}", call.name, error.what() );
                }
                catch ( ... )
                {
                    result = std::format( "Error: tool '{}' failed", call.name );
                }

                if ( halted.stop_requested() )
                {
                    // What a tool returns once it was asked to stop may be partial, so the model is not given it.
                    history_.push_back( { Conversation::Role::Tool, std::string( stoppedResult( clock, true ) ), {}, call.name } );

                    return false;
                }
            }

            addResult( call, std::move( result ), is_error, emit );

            return true;
        }

        /// A call of the last round a response may run, which is answered without running it.
        template<typename TEmit>
        void answerNotRun( const Conversation::ToolCall& call, const TEmit& emit )
        {
            const std::string result = std::format(
                "This tool call was not run: the response reached its limit of {} rounds.", *limits_.rounds );

            AgentEvent called{ AgentEvent::Kind::ToolCalled, result };
            called.call = &call;
            called.is_error = true;
            emit( called );

            history_.push_back( { Conversation::Role::Tool, result, {}, call.name } );
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
         * A result the context cannot hold is cut to its first tokens that fit, and the model told it was cut and by
         * how much; one with no room at all is refused, and the model told so. Never left to overflow the next
         * prompt, which would end the response with no word of why.
         */
        template<typename TEmit>
        void addResult( const Conversation::ToolCall& call, std::string result, bool is_error, const TEmit& emit )
        {
            history_.push_back( { Conversation::Role::Tool, result, {}, call.name } );

            std::vector<std::int32_t> prompt_tokens = encode( render( history_ ) );

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

                    history_.back().content = cut;
                    prompt_tokens = encode( render( history_ ) );

                    if ( fitsContext( prompt_tokens ) )
                    {
                        result = std::move( cut );
                        is_error = false;

                        break;
                    }

                    kept -= tokensOverContext( prompt_tokens );
                }

                history_.back().content = result;
            }

            AgentEvent called{ AgentEvent::Kind::ToolCalled, history_.back().content };
            called.call = &call;
            called.is_error = is_error;
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

        std::string notOfferedResult( std::string_view name ) const
        {
            std::string offered;

            for ( const auto& tool : tools_ )
            {
                offered += std::format( "{}{}", offered.empty() ? "" : ", ", tool.name );
            }

            return std::format( "Error: there is no tool named '{}'. The tools are: {}.", name, offered );
        }

        Handle handle_;
        std::vector<Tool> tools_;
        Asker ask_;
        std::string system_prompt_;
        std::map<std::string, std::string> template_variables_;
        std::optional<SamplingParams> sampling_;
        AgentLimits limits_;
        std::vector<Conversation::Turn> history_;
        std::set<std::string, std::less<>> allowed_for_conversation_;
    };

    /// The agent every published model runs as: CUDA, computing in BF16 whatever its weights' format.
    export using Agent = BasicAgent<DeviceType::Cuda, TensorDataType::BF16>;
}
