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
import AI.Response;

namespace Mila::AI
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;

    /**
     * @brief A model named by its store name, holding one conversation and running the calls the model makes.
     *
     * Through an agent the application never runs a call itself: it registers tools and a way to ask the person, and
     * the agent parses each call, asks, runs it and gives the model its result. Family-blind: everything a model
     * does differently comes from its handle. A response is one virtual call into the model per round.
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
         * @brief Answer one message, running every call the model makes, until it gives its answer.
         *
         * Blocks on the caller's thread, reporting what happens through on_event as it happens. The message, each
         * round's reply and each call's result join the conversation. A round cut short -- by the context, or by
         * stop -- runs none of its calls.
         *
         * @param message  What the user says.
         * @param on_event Every event in order; may be empty.
         * @param stop     Stops the reply; a call still waiting for the person's answer does not run.
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

            Conversation::Prompt prompt = render();
            std::vector<std::int32_t> tokens = encode( prompt );

            if ( !fitsContext( tokens ) )
            {
                // Never answered, so it never joined: the conversation stays one the context holds.
                history_.pop_back();
                response.status = GenerateStatus::ContextOverflow;

                return response;
            }

            GenerateParams params;
            params.sampling = sampling();

            for ( int round = 0; round < kMaximumRounds; ++round )
            {
                emit( { AgentEvent::Kind::RoundBegan } );
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
                        forward( reader.feed( handle_.tokenizer().decode( std::span<const std::int32_t>( &token, 1 ) ) ) );
                    },
                    params,
                    stop );

                // The round's end settles what the reader still held, and reads a call left open as written.
                forward( reader.finish() );

                const Conversation::Reply& reply = reader.reply();

                response.status = status;

                AgentEvent ended{ AgentEvent::Kind::RoundEnded };
                ended.status = status;
                emit( ended );

                for ( const auto& span : reply.malformed_tool_calls )
                {
                    emit( { AgentEvent::Kind::MalformedToolCall, span } );
                }

                const bool runs_calls = !reply.tool_calls.empty() && !tools_.empty()
                    && status == GenerateStatus::Success && !stop.stop_requested();

                if ( !runs_calls )
                {
                    history_.push_back( { Conversation::Role::Assistant, reply.content, {}, {}, reply.reasoning } );

                    return response;
                }

                history_.push_back(
                    { Conversation::Role::Assistant, reply.content, reply.tool_calls, {}, reply.reasoning } );

                for ( const auto& call : reply.tool_calls )
                {
                    if ( !answerCall( call, emit, stop ) )
                    {
                        response.status = GenerateStatus::ClientCancelled;

                        return response;
                    }
                }

                prompt = render();
                tokens = encode( prompt );

                if ( !fitsContext( tokens ) )
                {
                    response.status = GenerateStatus::ContextOverflow;

                    return response;
                }
            }

            return response;
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

        /// Rounds in one response, as Chat has run them.
        static constexpr int kMaximumRounds = 4;

        static constexpr std::string_view kDeclinedResult = "The user declined to run this tool call.";
        static constexpr std::string_view kStoppedResult = "The user stopped the reply before this tool call ran.";

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

        Conversation::Prompt render() const
        {
            Conversation::PromptOptions options;
            options.template_variables = template_variables_;

            if ( !tools_.empty() )
            {
                options.tools_json = toolsJson();
            }

            if ( system_prompt_.empty() )
            {
                return handle_.protocol()->formatPrompt( history_, options );
            }

            std::vector<Conversation::Turn> turns;
            turns.reserve( history_.size() + 1 );
            turns.push_back( { Conversation::Role::System, system_prompt_ } );
            turns.insert( turns.end(), history_.begin(), history_.end() );

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
         * @return False when the reply was stopped while the call waited: it does not run, and the response ends.
         */
        template<typename TEmit>
        bool answerCall( const Conversation::ToolCall& call, const TEmit& emit, std::stop_token stop )
        {
            const Tool* tool = findTool( call.name );

            ToolAnswer answer = ToolAnswer::Once;

            if ( tool != nullptr && !allowed_for_conversation_.contains( call.name ) )
            {
                answer = ask_ ? ask_( ToolQuestion{ tool->name, tool->description, call.arguments } ) : ToolAnswer::Deny;
            }

            if ( stop.stop_requested() )
            {
                // Every call in the conversation keeps a result, so the next prompt renders; nothing reports one.
                history_.push_back( { Conversation::Role::Tool, std::string( kStoppedResult ), {}, call.name } );

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
                    result = tool->run( call.arguments );
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
            }

            AgentEvent called{ AgentEvent::Kind::ToolCalled, result };
            called.call = &call;
            called.is_error = is_error;
            emit( called );

            history_.push_back( { Conversation::Role::Tool, std::move( result ), {}, call.name } );

            return true;
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
        std::vector<Conversation::Turn> history_;
        std::set<std::string, std::less<>> allowed_for_conversation_;
    };

    /// The agent every published model runs as: CUDA, computing in BF16 whatever its weights' format.
    export using Agent = BasicAgent<DeviceType::Cuda, TensorDataType::BF16>;
}
